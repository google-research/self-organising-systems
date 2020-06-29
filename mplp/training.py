"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

class TrainingRegime():

  def __init__(self, network, heldout_weight, hint_loss_ratio=0.7,
               remember_loss_ratio=None, delta_loss=None, stop_grad=False):
    self.network = network
    self.heldout_weight = heldout_weight
    self.hint_loss_ratio = hint_loss_ratio
    self.remember_loss_ratio = remember_loss_ratio
    self.delta_loss = delta_loss
    self.stop_grad = stop_grad

  @tf.function
  def mp_loss(self, pfw, xt, yt, xe, ye, num_loops):
    print("compiling mp_loss")

    network = self.network
    tot_loss = 0.
    hint_train_loss = 0.
    hint_rem_loss = 0.

    print("xt", xt.shape)
    xrem = tf.reshape(xt, [-1, xt.shape[-1]])
    print("xrem", xrem.shape)
    yrem = tf.reshape(yt, [-1, yt.shape[-1]])

    inner_batch_size = xt.shape[1]
    observed_x = 0

    for i in tf.range(num_loops):
      xti, yti = [e[i] for e in [xt, yt]]
      if self.stop_grad:
        pfw = tf.nest.map_structure(tf.stop_gradient, pfw)

      pfw, deltas_i = network.inner_update(pfw, xti, yti)

      if self.delta_loss is not None:
        # not updated
        tot_loss += network.compute_deltas_loss(deltas_i) * self.delta_loss

      if self.hint_loss_ratio is not None:
        # Add hint losses for the training: improve with an inner update on
        # exactly what we gave you.
        result_i, _ = network.forward(pfw, xti)
        loss_i, _ = network.compute_loss(result_i, yti)
        hint_train_loss += tf.reduce_mean(loss_i)

        # A loss that randomly samples already observed data points.
        if self.remember_loss_ratio is not None:
          minval = 0
          maxval = tf.maximum(1, observed_x)
          rem_idx = tf.random.uniform(
              [inner_batch_size], minval=minval, maxval=maxval, dtype=tf.int32)
          observed_x += inner_batch_size

          xremi = tf.gather(xrem, rem_idx)
          yremi = tf.gather(yrem, rem_idx)

          result_i, _ = network.forward(pfw, xremi)
          loss_i, _ = network.compute_loss(result_i, yremi)
          loss_i = tf.reduce_mean(loss_i)
          # mask the first, since it's invalid.
          loss_i = loss_i * tf.cond(i > 0, lambda: 1., lambda: 0.)
          hint_rem_loss += loss_i


    if self.hint_loss_ratio is not None:
      # scale the hint loss to be, in practice, equal to the final loss in
      # magnitude, scaled by the ratio.
      hint_train_loss /= tf.cast(num_loops, tf.float32)
      hint_rem_loss /= tf.cast(num_loops, tf.float32)

      remember_loss_ratio = 0. if self.remember_loss_ratio is None else self.remember_loss_ratio

      tot_loss += hint_train_loss * (1. - remember_loss_ratio)
      tot_loss += hint_rem_loss * remember_loss_ratio
      tot_loss *= self.hint_loss_ratio

    # compute loss at the end.
    final_loss_ratio = 1. - (0. if self.hint_loss_ratio is None else self.hint_loss_ratio)

    if self.heldout_weight is None or self.heldout_weight < 0.99:
      xt_flat = tf.reshape(xt, [-1, xt.shape[-1]])
      yt_flat = tf.reshape(yt,  [-1, yt.shape[-1]])

      # get a random sample:
      idx = tf.random.uniform(
          [inner_batch_size], minval=0, maxval=xt_flat.shape[0], dtype=tf.int32)
      xfinal = tf.gather(xt_flat, idx)
      yfinal = tf.gather(yt_flat, idx)

      result_i, _ = network.forward(pfw, xfinal)
      loss_i, _ = network.compute_loss(result_i, yfinal)

      train_loss_i = tf.reduce_mean(loss_i) * final_loss_ratio
      if self.heldout_weight is not None:
        train_loss_i *= (1 - self.heldout_weight)
      tot_loss += train_loss_i

    if self.heldout_weight is not None and self.heldout_weight > 0.01:
      # add an heldout loss: this is a loss on data the inner_update has no
      # access to!
      # As opposed to the train final loss, we assume xe and ye to be already
      # of the right size: we don't gather.
      result_hi, _ = network.forward(pfw, xe)
      hloss_i, _ = network.compute_loss(result_hi, ye)
      hloss_i = tf.reduce_mean(hloss_i) * final_loss_ratio
      if self.heldout_weight is not None:
        hloss_i *= self.heldout_weight
      tot_loss += hloss_i

    # TODO: add more loss metrics when needed.
    return tot_loss, pfw, (hint_train_loss, hint_rem_loss)

  @tf.function
  def batch_mp_loss(self, pfw_b, xt_b, yt_b, xe_b, ye_b, num_loops,
                    same_pfw=False):
    print("compiling batch_mp_loss")
    task_losses = []
    hint_tr_losses = []
    hint_rem_losses = []
    # Not sure whether you can do something fancier in tf when you have
    # All the code in 1 less dimension.
    next_idx = []
    next_pfw = []
    for i in range(len(xt_b)):
      pfw = pfw_b if same_pfw else pfw_b[i]
      # deserialize. If there is no need to do so, the network needs to return
      # a noop.
      pfw = self.network.deserialize_pfw(pfw)
      xt, yt, xe, ye = [el[i] for el in [xt_b, yt_b, xe_b, ye_b]]
      current_task_loss, curr_t_next_pfw, side_losses = self.mp_loss(
          pfw, xt, yt, xe, ye, num_loops)
      task_losses.append(current_task_loss)
      hint_train_loss_i, hint_rem_loss_i = side_losses
      hint_tr_losses.append(hint_train_loss_i)
      hint_rem_losses.append(hint_rem_loss_i)
      # serialize. If there is no need to do so, the network needs to return
      # a noop.
      curr_t_next_pfw = self.network.serialize_pfw(curr_t_next_pfw)
      next_pfw.append(curr_t_next_pfw)
    return tf.reduce_mean(tf.stack(task_losses)), tf.stack(next_pfw),\
     (tf.reduce_mean(tf.stack(hint_tr_losses)), (tf.reduce_mean(tf.stack(hint_rem_losses))))
