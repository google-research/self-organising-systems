"""
Cellular Automata Model
"""
from self_organising_systems.texture_ca.config import cfg
import tensorflow as tf
import numpy as np
import json
import os

def pad_repeat(x, pad):
  x = tf.concat([x[:, -pad:], x, x[:, :pad]], 1)
  x = tf.concat([x[:, :, -pad:], x, x[:, :, :pad]], 2)
  return x

def get_variables(f):
  '''Get all vars involved in computing a function. Userful for'''
  with tf.GradientTape() as g:
    f()
    return g.watched_variables()

def fake_quant(x, min, max):
  y = tf.quantization.fake_quant_with_min_max_vars(x, min=min, max=max)
  return y

def fake_param_quant(w):
  bound = tf.stop_gradient(tf.reduce_max(tf.abs(w)))
  w = fake_quant(w, -bound, bound)
  return w

def to_rgb(x):
  return x[..., :3]/(cfg.texture_ca.q) + 0.5

@tf.function
def perceive(x, angle=0.0, repeat=True):
  chn = tf.shape(x)[-1]
  identify = np.outer([0, 1, 0], [0, 1, 0])
  dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
  dy = dx.T
  laplacian = np.outer([1, 2, 1], [1, 2, 1]) / 8.0
  laplacian[1, 1] -= 2.0
  c, s = tf.cos(angle), tf.sin(angle)
  kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy, laplacian], -1)[:, :, None, :]
  kernel = tf.repeat(kernel, chn, 2)
  pad_mode = 'SAME'
  if repeat:
    x = pad_repeat(x, 1)
    pad_mode = 'VALID'
  y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], pad_mode)
  return y

class DenseLayer:
  def __init__(self, in_n, out_n,
              init_fn=tf.initializers.glorot_uniform()):
    w0 = tf.concat([init_fn([in_n, out_n]), tf.zeros([1, out_n])], 0)
    self.w = tf.Variable(w0)

  def embody(self):
    w = fake_param_quant(self.w)
    w, b = w[:-1], w[-1]
    w = w[None, None, ...]
    def f(x):
      # TFjs matMul doesn't work with non-2d tensors, so using
      # conv2d instead of 'tf.matmul(x, w)+b'
      return tf.nn.conv2d(x, w, 1, 'VALID')+b
    return f

class CAModel:

  def __init__(self, params=None):
    super().__init__()
    self.fire_rate = cfg.texture_ca.fire_rate
    self.channel_n = cfg.texture_ca.channel_n

    init_fn = tf.initializers.glorot_normal(cfg.texture_ca.fixed_seed or None)
    self.layer1 = DenseLayer(self.channel_n*4, cfg.texture_ca.hidden_n, init_fn)
    self.layer2 = DenseLayer(cfg.texture_ca.hidden_n, self.channel_n, tf.zeros)

    self.params = get_variables(self.embody)
    if params is not None:
      self.set_params(params)

  def embody(self, quantized=True):
    layer1 = self.layer1.embody()
    layer2 = self.layer2.embody()

    def noquant(x, min, max):
      return tf.clip_by_value(x, min, max)
    qfunc = fake_quant if quantized else noquant

    @tf.function
    def f(x, fire_rate=None, angle=0.0, step_size=1.0):
      y = perceive(x, angle)
      y = qfunc(y, min=-cfg.texture_ca.q, max=cfg.texture_ca.q)
      y = tf.nn.relu(layer1(y))
      y = qfunc(y, min=0.0, max=cfg.texture_ca.q)
      y = layer2(y)
      dx = y*step_size
      dx = qfunc(dx, min=-cfg.texture_ca.q, max=cfg.texture_ca.q)
      if fire_rate is None:
        fire_rate = self.fire_rate
      update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
      x += dx * tf.cast(update_mask, tf.float32)
      x = qfunc(x, min=-cfg.texture_ca.q, max=cfg.texture_ca.q)
      return x
    return f

  def get_params(self):
    return [p.numpy() for p in self.params]

  def set_params(self, params):
    for v, p in zip(self.params, params):
      v.assign(p)

  def save_params(self, filename):
    with tf.io.gfile.GFile(filename, mode='wb') as f: 
      np.save(f, self.get_params())

  def load_params(self, filename):
    with tf.io.gfile.GFile(filename, mode='rb') as f: 
      params = np.load(f, allow_pickle=True)
      self.set_params(params)
