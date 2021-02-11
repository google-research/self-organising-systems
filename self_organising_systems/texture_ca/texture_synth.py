# Lint as: python3
"""
Texture synth experiments.
"""
from absl import app
from absl import flags
from absl import logging
FLAGS = flags.FLAGS

from self_organising_systems.texture_ca.config import cfg
tcfg = cfg.texture_ca
from self_organising_systems.texture_ca.losses import StyleModel, Inception
from self_organising_systems.shared.video import VideoWriter
from self_organising_systems.shared.util import tile2d, Bunch
from self_organising_systems.texture_ca.ca import CAModel, to_rgb

import tensorflow as tf
# TF voodoo during migration period...
tf.compat.v1.enable_v2_behavior()
import numpy as np

def main(_):
  texture_synth_trainer = TextureSynthTrainer()
  texture_synth_trainer.train()

class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, **slots):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._slot_names = slots.keys()
    self._size = None
    for k, v in slots.items():
      if self._size is None:
        self._size = len(v)
      assert self._size == len(v)
      setattr(self, k, np.asarray(v))

  def sample(self, n):
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

def create_loss_model():
  loss_type, loss_params = tcfg.objective.split(':', 1)
  if loss_type == "style":
    texture_fn = loss_params
    input_texture_path = "%s/%s"%(tcfg.texture_dir, texture_fn)
    loss_model = StyleModel(input_texture_path)
  elif loss_type == "inception":
    layer_name, ch = loss_params.split(':')
    loss_model = Inception(layer_name, int(ch))
  return loss_model

class TextureSynthTrainer:
  def __init__(self, loss_model=None):
    self.experiment_log_dir = "%s/%s"%(cfg.logdir, cfg.experiment_name)
    self.writer = tf.summary.create_file_writer(self.experiment_log_dir)

    if loss_model is None:
      loss_model = create_loss_model()
    self.loss_model = loss_model

    self.ca = CAModel()
    if tcfg.ancestor_npy:
      self.ancestor_ca = CAModel()
      ancestor_fn = "%s/%s" % (tcfg.ancestor_dir, tcfg.ancestor_npy)
      self.ancestor_ca.load_params(ancestor_fn)
      self.ca.load_params(ancestor_fn)
      logging.info("loaded pre-trained model %s" % tcfg.ancestor_npy)
    self.loss_log = []
    self.pool = SamplePool(x=self.seed_fn(tcfg.pool_size))
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [1000], [tcfg.lr, tcfg.lr*0.1])
    self.trainer = tf.keras.optimizers.Adam(lr_sched)

  def visualize_batch_tf(self, x0, x, step_num):
    vis0 = np.hstack(to_rgb(x0))
    vis1 = np.hstack(to_rgb(x))
    vis = np.vstack([vis0, vis1])
    tf.summary.image("batch_vis", vis[None, ...])

  def train(self):
    with self.writer.as_default():
      for _ in range(tcfg.train_steps+1):
        step_num = len(self.loss_log)
        step = self.train_step()
        if step_num%50 == 0 or step_num == tcfg.train_steps:
          self.visualize_batch_tf(step.x0, step.batch.x, step_num)
          self.ca.save_params("%s/%s.npy" % (cfg.logdir, cfg.experiment_name))
        logging.info('step: %d, log10(loss): %s, loss: %s'%(len(self.loss_log), np.log10(step.loss), step.loss.numpy()))
      self.save_video("%s/%s.mp4" % (cfg.logdir, cfg.experiment_name), self.ca.embody)

  def train_step(self):
    step_num = len(self.loss_log)
    tf.summary.experimental.set_step(step_num)
    batch = self.pool.sample(tcfg.batch_size)
    x0 = batch.x.copy()
    if step_num%2==0:
      x0[:1] = self.seed_fn(1)
    batch.x[:], loss = self._train_step(x0)
    batch.commit()
    tf.summary.scalar("loss", loss)
    self.loss_log.append(loss.numpy())
    return Bunch(batch=batch, x0=x0, loss=loss, step_num=step_num)

  @tf.function
  def _train_step(self, x):
    iter_n = tf.random.uniform([], tcfg.rollout_len_min, tcfg.rollout_len_max, tf.int32)
    with tf.GradientTape(persistent=False) as g:
      f = self.ca.embody()
      for i in tf.range(iter_n):
        x = f(x)
      loss = self.loss_model(to_rgb(x))
    grads = g.gradient(loss, self.ca.params)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    self.trainer.apply_gradients(zip(grads, self.ca.params))
    return x, loss

  def seed_fn(self, n):
    states = np.zeros([n, tcfg.img_size, tcfg.img_size, tcfg.channel_n], np.float32)
    return states

  def save_video(self, path, f):
    state = self.seed_fn(1)
    f = self.ca.embody()
    if tcfg.ancestor_npy:
      state_ancestor = self.seed_fn(1)
      f_ancestor = self.ancestor_ca.embody()
    with VideoWriter(path, 60.0) as vid:
      for i in range(tcfg.viz_rollout_len):
        # visualize the RGB + hidden states.
        if tcfg.hidden_viz_group:
          padding_channel_len = (3 - state[0].shape[2] % 3) % 3
          splitframe = np.split(np.pad(state[0], ((0,0), (0,0), (0,padding_channel_len)), mode='constant'), (state[0].shape[2] + padding_channel_len)/3, 2)
        else:
          hidden = np.transpose(np.repeat(state[0][..., 3:, None], 3, -1), (2, 0, 1, 3))
          splitframe = np.concatenate([state[0][None, ..., :3], hidden], 0)
        frame = to_rgb(tile2d(splitframe))
        vid.add(frame)
        if tcfg.ancestor_npy:
          c_state = f(state, fire_rate=0.5)
          a_state = f_ancestor(state, fire_rate=0.5)
          progress = max(1.25*(i/tcfg.viz_rollout_len) - 0.25, 0.0)
          state = (1-progress)*c_state + progress*a_state
        else:
          state = f(state, fire_rate=0.5)


if __name__ == '__main__':
  app.run(main)
