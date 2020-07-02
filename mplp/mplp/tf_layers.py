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

from tensorflow.keras.layers import Dense
from tensorflow.io import gfile
import numpy as np

from .core import Arrow
from .core import Add
from .core import RepeatInputs
from .core import RepeatState
from .core import RepeatForBatch
from .core import PairwiseOp
from .core import StandardizeInputsAndStates
from .core import StandardizeInputsAndStatesLink
from .core import ConcatenateInputsAndStates
from .core import ParamUpdate
from .core import MessageGenerator
from .core import StatefulLearner
from .core import MatMul
from .core import Softmax
from .core import CrossEntropyLoss
from .core import L1Loss
from .core import L2Loss

flatten = lambda l : [e for sublist in l for e in sublist]

class MPLayer():

  def init(self, inner_batch_size=None):
    """Initialize a new network's inner weights and return them.
    If inner_batch_size is not None, we initialize carry states of
    the meta-learners.
    """
    pass

  def setup(self, in_dim, message_size, shared_params=None):
    """Finalize to setup the meta-learner.
    This function would be called by __init__ if in_dim and message_size were
    provided at construction.
    """
    pass

  def forward(self, fw_params, inputs):
    """Synonimous to what predict would be in traditional nets."""
    pass

  def backward(self, fw_params, m_in, side_inputs):
    """Weight update rule."""
    pass

  def get_trainable_weights(self):
    """Return all the trainable meta-learning parameters."""
    pass

  def get_all_weights(self):
    """Return all the meta-learning parameters.
    This is used for storing weights in checkpoints."""
    pass

  def load_weights(self, weights):
    """Manually initialize all weights.
    This is used for loading weights from checkpoints."""
    pass

  def forward_init(self, fw_params, inputs):
    """Performs a custom forward step to initialize specific parameters.
    Defaults to a traditional forward step that also returns its own parameters.
    """
    return self.forward(fw_params, inputs) + (fw_params,)

  def backward_init(self, fw_params, m_in, side_inputs):
    """Performs a custom backward step to initialize specific parameters.
    It's important to not return modified params if they get modified in the
    inner loop.
    Defaults to a traditional backward step that returns unchanged params.
    """
    _, _, m_out, deltas = self.backward(
        fw_params, m_in, side_inputs)
    return fw_params, m_out, deltas

  def update_statistics(self, stats, update_perc=1.):
    """Updates statistics from minibatch runs. This usually means having run
    several backward_inits and aggregated standardization parameters."""
    return


class MPDense(MPLayer):
  def __init__(self, out_dim, in_dim=None, message_size=None, stateful=False,
               stateful_hidden_n=15):
    self.out_dim = out_dim
    # this is important because it is also used in the f functions!
    self.W_std_init_val = 0.05

    self.stateful = stateful
    self.stateful_hidden_n = stateful_hidden_n
    if stateful:
      self.W_cname = "carry"
      self.b_cname = "carry"

    if in_dim is not None and message_size is not None:
      # assumes we don't share learners, yet.
      self.setup(in_dim, message_size)

  def setup(self, in_dim, message_size, shared_params=None):
    """This has to be called either at __init__ time, or by a constructor,
    before the layer can be used."""

    self.in_dim = in_dim
    self.message_size = message_size

    self.W_name = "W"
    self.b_name = "b"

    W_name = self.W_name
    W_b_name = W_name + "_b"
    W_in_name = W_name + "_in"
    W_states = [W_b_name, W_in_name]
    W_in_dim = len(W_states) + message_size
    cname = "all_W_in"
    self.W_fw_arrow = Arrow([
        MatMul(W_name, [in_dim, self.out_dim], self.W_std_init_val)])

    scale_init_val = self.W_std_init_val / 5

    if shared_params is not None:
      assert self.stateful, "not implemented for stateless."
      skeys = shared_params.keys()
      if "W_net" in skeys:
        if "W_out_std" in skeys:
          W_learner = StatefulLearner(
              xname=cname, shared_network=shared_params["W_net"],
              pname=W_name, shared_out_std=shared_params["W_out_std"],
              is_residual=True, out_message_size=message_size)
        else:
          W_learner = StatefulLearner(
              xname=cname, shared_network=shared_params["W_net"],
              pname=W_name, scale_init_val=scale_init_val, is_residual=True,
              out_message_size=message_size)
      else:
        W_learner = StatefulLearner(
            xname=cname, in_dim=W_in_dim, hidden_n=self.stateful_hidden_n,
            pname=W_name, scale_init_val=scale_init_val, is_residual=True,
            out_message_size=message_size)
      if "W_in_std" in skeys:
        W_standardizer = StandardizeInputsAndStatesLink(
            shared_params["W_in_std"])
      else:
        W_standardizer = StandardizeInputsAndStates(W_states)
    else:
      if self.stateful:
        W_learner = StatefulLearner(
            xname=cname, in_dim=W_in_dim, hidden_n=self.stateful_hidden_n,
            pname=W_name, scale_init_val=scale_init_val, is_residual=True,
            out_message_size=message_size)
      W_standardizer = StandardizeInputsAndStates(W_states)

    W_bw_ops = [
        # prepare input data
        RepeatInputs(repeats=in_dim, axis=1),
        RepeatState(W_in_name, repeats=self.out_dim, axis=-1),
        RepeatForBatch(W_name, out_name=W_b_name),
        W_standardizer,
        ConcatenateInputsAndStates(W_states, cname=cname),
    ]
    if self.stateful:
      W_bw_ops += [
          # actual computation
          # We create an alias for the inputs (such as this all_W_in).
          W_learner
      ]
    else:
      # Does not support shared params yet.
      W_bw_ops += [
          ParamUpdate(W_name, in_dim=W_in_dim, xname=cname,
                      scale_init_val=scale_init_val,
                      is_residual=True),
          MessageGenerator(in_dim=W_in_dim, xname=cname,
                           out_message_size=message_size),
      ]
    self.W_bw_arrow = Arrow(W_bw_ops)

    b_name = self.b_name
    b_b_name = b_name + "_b"
    b_in_name = b_name + "_in"
    b_states = [b_b_name, b_in_name]
    b_in_dim = len(b_states) + message_size
    cname = "all_b_in"
    self.b_fw_arrow = Arrow([Add(b_name, [self.out_dim])])

    if shared_params is not None:
      assert self.stateful, "not implemented for stateless."
      skeys = shared_params.keys()
      if "b_net" in skeys:
        if "b_out_std" in skeys:
          b_learner = StatefulLearner(
              xname=cname, shared_network=shared_params["b_net"],
              pname=b_name, shared_out_std=shared_params["b_out_std"],
              is_residual=True, out_message_size=message_size)
        else:
          b_learner = StatefulLearner(
              xname=cname, shared_network=shared_params["b_net"],
              pname=b_name, scale_init_val=scale_init_val, is_residual=True,
              out_message_size=message_size)
      else:
        b_learner = StatefulLearner(
            xname=cname, in_dim=b_in_dim, hidden_n=self.stateful_hidden_n,
            pname=b_name, scale_init_val=scale_init_val, is_residual=True,
            out_message_size=message_size)
      if "b_in_std" in skeys:
        b_standardizer = StandardizeInputsAndStatesLink(
            shared_params["b_in_std"])
      else:
        b_standardizer = StandardizeInputsAndStates(b_states)
    else:
      if self.stateful:
        b_learner = StatefulLearner(
            xname=cname, in_dim=b_in_dim, hidden_n=self.stateful_hidden_n,
            pname=b_name, scale_init_val=scale_init_val, is_residual=True,
            out_message_size=message_size)
      b_standardizer = StandardizeInputsAndStates(b_states)

    b_bw_ops = [
        # prepare input data
        RepeatForBatch(b_name, out_name=b_b_name),
        b_standardizer,
        ConcatenateInputsAndStates(b_states, cname=cname),
    ]
    if self.stateful:
      b_bw_ops += [
          # actual computation
          b_learner
      ]
    else:
      b_bw_ops += [
        # actual computation
        ParamUpdate(b_name, in_dim=b_in_dim, xname=cname,
                    scale_init_val=scale_init_val,
                    is_residual=True),
        MessageGenerator(in_dim=b_in_dim, xname=cname,
                         out_message_size=message_size),
      ]
    self.b_bw_arrow = Arrow(b_bw_ops)

    # these are flattened because they need to be sent to an optimizer.
    self.trainable_weights = self.W_fw_arrow.trainable_weights + \
        self.W_bw_arrow.trainable_weights + \
        self.b_fw_arrow.trainable_weights + self.b_bw_arrow.trainable_weights
    # these keep their structure.
    self.weights = [self.W_fw_arrow.weights, self.W_bw_arrow.weights,
        self.b_fw_arrow.weights, self.b_bw_arrow.weights]

  def get_trainable_weights(self):
    return self.trainable_weights

  def get_all_weights(self):
    return self.weights

  def load_weights(self, weights):
    for arrow, aw in zip([self.W_fw_arrow, self.W_bw_arrow,
                          self.b_fw_arrow, self.b_bw_arrow],
                         weights):
      arrow.load_weights(aw)

  def init(self, inner_batch_size=None):
    W = self.W_fw_arrow.init()[self.W_name]
    b = self.b_fw_arrow.init()[self.b_name]
    params = [W, b]
    if self.stateful and inner_batch_size is not None:
      W_bw_carry = self.W_bw_arrow.init()[self.W_cname]
      W_bw_carry = tf.broadcast_to(
          W_bw_carry,
          (inner_batch_size,) + W.shape + (W_bw_carry.shape[0],))
      b_bw_carry = self.b_bw_arrow.init()[self.b_cname]
      b_bw_carry = tf.broadcast_to(
          b_bw_carry,
          (inner_batch_size,) + b.shape + (b_bw_carry.shape[0],))

      params += [W_bw_carry, b_bw_carry]

    return params

  def forward(self, fw_params, inputs):
    W, b = fw_params[0], fw_params[1]
    # prepare input for graph compatibility
    W_states = {self.W_name: W}
    new_states, result, _ = self.W_fw_arrow(W_states, inputs, False)
    # unpack states
    last_W_in = new_states[self.W_name + "_in"]

    b_states = {self.b_name: b}
    new_states, result, _ = self.b_fw_arrow(b_states, result, False)
    # unpack states
    last_b_in = new_states[self.b_name + "_in"]

    side_outputs = (last_W_in, last_b_in)

    return result, side_outputs

  def backward(self, fw_params, m_in, side_inputs):
    return self._backward(fw_params, m_in, side_inputs, False)

  def backward_init(self, fw_params, m_in, side_inputs):
    return self._backward(fw_params, m_in, side_inputs, True)

  def _backward(self, fw_params, m_in, side_inputs, initialize):
    W, b = fw_params[0], fw_params[1]
    last_W_in, last_b_in = side_inputs

    batch_size = m_in.shape[0]

    ### BIAS
    # prepare input for graph compatibility
    b_states = {self.b_name: b, self.b_name + "_in": last_b_in}
    if self.stateful:
      b_states[self.b_cname] = fw_params[3]

    new_b_states, m_b, side_out_b = self.b_bw_arrow(b_states, m_in, initialize)
    # unpack
    new_b = new_b_states[self.b_name]
    delta_b = side_out_b[self.b_name + "_delta"]
    if initialize:
      b_out_mean = side_out_b[self.b_name + "_out_mean"]
      b_in_mean = side_out_b["inputs_mean"]
      b_in_norm = side_out_b["inputs_norm"]

    ### KERNEL
    # prepare input for graph compatibility
    W_states = {self.W_name: W, self.W_name + "_in": last_W_in}
    if self.stateful:
      W_states[self.W_cname] = fw_params[2]

    new_W_states, m_W, side_out_W = self.W_bw_arrow(W_states, m_b, initialize)
    # unpack
    new_W = new_W_states[self.W_name]
    delta_W = side_out_W[self.W_name + "_delta"]
    if initialize:
      W_out_mean = side_out_W[self.W_name + "_out_mean"]
      W_in_mean = side_out_W["inputs_mean"]
      W_in_norm = side_out_W["inputs_norm"]


    next_params = [new_W, new_b]
    if self.stateful:
      next_W_carry = new_W_states[self.W_cname]
      next_b_carry = new_b_states[self.b_cname]
      next_params += [next_W_carry, next_b_carry]

    if initialize:
      side_outputs = (W_out_mean, W_in_mean, W_in_norm,
                      b_out_mean, b_in_mean, b_in_norm)
    else:
      side_outputs = (delta_W, delta_b)
    return next_params, m_W, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    (W_out_mean, W_in_mean, W_in_norm,
     b_out_mean, b_in_mean, b_in_norm) = stats

    W_stats = {self.W_name + "_out_mean": W_out_mean,
               "inputs_mean": W_in_mean, "inputs_norm": W_in_norm}
    self.W_bw_arrow.update_statistics(W_stats, update_perc)

    b_stats = {self.b_name + "_out_mean": b_out_mean,
               "inputs_mean": b_in_mean, "inputs_norm": b_in_norm}
    self.b_bw_arrow.update_statistics(b_stats, update_perc)


class MPActivation(MPLayer):

  def __init__(self, activation, in_dim=None, message_size=None, stateful=False,
               stateful_hidden_n=15):
    self.activation = activation

    self.stateful = stateful
    self.stateful_hidden_n = stateful_hidden_n
    if stateful:
      self.carry_name = "carry"

    if in_dim is not None and message_size is not None:
      self.setup(in_dim, message_size)

  def setup(self, in_dim, message_size, shared_params=None):
    self.in_dim = in_dim
    self.op_in_name = "act_in"

    op_in_name = self.op_in_name
    states_names = [op_in_name]
    bw_in_dim = len(states_names) + message_size
    cname = "all_act_in"
    self.fw_arrow = Arrow([PairwiseOp(op_in_name, self.activation)])

    bw_ops = [
        # prepare input data
        StandardizeInputsAndStates(states_names),
        ConcatenateInputsAndStates(states_names, cname=cname),
    ]
    if self.stateful:
      bw_ops += [
          # actual computation
          StatefulLearner(
              xname=cname, in_dim=bw_in_dim, hidden_n=self.stateful_hidden_n,
              out_message_size=message_size)
      ]
    else:
      bw_ops += [
        MessageGenerator(in_dim=bw_in_dim, xname=cname,
                         out_message_size=message_size),
      ]
    self.bw_arrow = Arrow(bw_ops)

    self.trainable_weights = self.fw_arrow.trainable_weights + \
        self.bw_arrow.trainable_weights
    self.weights = [self.fw_arrow.weights, self.bw_arrow.weights]

  def init(self, inner_batch_size=None):
    if self.stateful and inner_batch_size is not None:
      bw_carry = self.bw_arrow.init()[self.carry_name]
      bw_carry = tf.broadcast_to(
          bw_carry,
          (inner_batch_size, self.in_dim) + (bw_carry.shape[0],))
      return [bw_carry]
    # return an empty forward list.
    return []

  def get_trainable_weights(self):
    return self.trainable_weights

  def get_all_weights(self):
    return self.weights

  def load_weights(self, weights):
    for arrow, aw in zip([self.fw_arrow, self.bw_arrow], weights):
      arrow.load_weights(aw)

  def forward(self, fw_params, inputs):
    # params is expected to be an empty tuple
    states = {}
    new_states, result, _ = self.fw_arrow(states, inputs, False)
    # unpack states
    last_in = new_states[self.op_in_name]
    return result, last_in

  def backward(self, fw_params, m_in, side_inputs):
    return self._backward(fw_params, m_in, side_inputs, False)

  def backward_init(self, fw_params, m_in, side_inputs):
    return self._backward(fw_params, m_in, side_inputs, True)

  def _backward(self, fw_params, m_in, side_inputs, initialize):
    last_in = side_inputs

    states = {self.op_in_name: last_in}
    if self.stateful:
      states[self.carry_name] = fw_params[0]

    new_states, m_act, side_out = self.bw_arrow(states, m_in, initialize)

    if initialize:
      in_mean = side_out["inputs_mean"]
      in_norm = side_out["inputs_norm"]

    next_params = []
    if self.stateful:
      next_carry = new_states[self.carry_name]
      next_params.append(next_carry)

    if initialize:
      side_outputs = (in_mean, in_norm)
    else:
      side_outputs = ()
    return next_params, m_act, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    (in_mean, in_norm) = stats

    stats = {"inputs_mean": in_mean, "inputs_norm": in_norm}
    self.bw_arrow.update_statistics(stats, update_perc)


class MPSoftmax(MPLayer):

  def __init__(self, in_dim=None, message_size=None, stateful=False,
               stateful_hidden_n=15):
    self.stateful = stateful
    self.stateful_hidden_n = stateful_hidden_n
    if stateful:
      self.carry_name = "carry"

    if in_dim is not None and message_size is not None:
      self.setup(in_dim, message_size)

  def setup(self, in_dim, message_size, shared_params=None):
    self.in_dim = in_dim
    cname = "all_softmax_in"

    op_name = "softmax"
    softmax_op = Softmax(op_name)
    self.states_names = softmax_op.states_names
    bw_in_dim = len(self.states_names) + message_size
    self.fw_arrow = Arrow([softmax_op])

    bw_ops = [
        StandardizeInputsAndStates(self.states_names),
        ConcatenateInputsAndStates(self.states_names, cname=cname),
    ]
    if self.stateful:
      bw_ops += [
          StatefulLearner(
              xname=cname, in_dim=bw_in_dim, hidden_n=self.stateful_hidden_n,
              out_message_size=message_size)
      ]
    else:
      bw_ops += [
        MessageGenerator(in_dim=bw_in_dim, xname=cname,
                         out_message_size=message_size),
      ]
    self.bw_arrow = Arrow(bw_ops)

    self.trainable_weights = self.fw_arrow.trainable_weights + \
        self.bw_arrow.trainable_weights
    self.weights = [self.fw_arrow.weights, self.bw_arrow.weights]

  def init(self, inner_batch_size=None):
    if self.stateful and inner_batch_size is not None:
      bw_carry = self.bw_arrow.init()[self.carry_name]
      bw_carry = tf.broadcast_to(
          bw_carry,
          (inner_batch_size, self.in_dim) + (bw_carry.shape[0],))
      return [bw_carry]
    # return an empty forward list.
    return []

  def get_trainable_weights(self):
    return self.trainable_weights

  def get_all_weights(self):
    return self.weights

  def load_weights(self, weights):
    for arrow, aw in zip([self.fw_arrow, self.bw_arrow], weights):
      arrow.load_weights(aw)

  def forward(self, fw_params, inputs):
    # params is expected to be empty
    states = {}
    new_states, result, _ = self.fw_arrow(states, inputs, False)
    # unpack states
    last_in = tuple(new_states[k] for k in self.states_names)

    return result, last_in

  def backward(self, fw_params, m_in, side_inputs):
    return self._backward(fw_params, m_in, side_inputs, False)

  def backward_init(self, fw_params, m_in, side_inputs):
    return self._backward(fw_params, m_in, side_inputs, True)

  def _backward(self, fw_params, m_in, side_inputs, initialize):
    last_in = side_inputs

    states = {k:v for k, v in zip(self.states_names, last_in)}
    if self.stateful:
      states[self.carry_name] = fw_params[0]

    new_states, m_act, side_out = self.bw_arrow(states, m_in, initialize)

    if initialize:
      in_mean = side_out["inputs_mean"]
      in_norm = side_out["inputs_norm"]

    next_params = []
    if self.stateful:
      next_carry = new_states[self.carry_name]
      next_params.append(next_carry)

    if initialize:
      side_outputs = (in_mean, in_norm)
    else:
      side_outputs = ()
    return next_params, m_act, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    (in_mean, in_norm) = stats

    stats = {"inputs_mean": in_mean, "inputs_norm": in_norm}
    self.bw_arrow.update_statistics(stats, update_perc)


class MPLoss():

  def __init__(self, fw_op, in_dim=None, message_size=None,
               stateful=False, stateful_hidden_n=15):
    self.fw_op = fw_op

    self.stateful = stateful
    self.stateful_hidden_n = stateful_hidden_n
    if stateful:
      self.carry_name = "carry"

    if in_dim is not None and message_size is not None:
      self.setup(in_dim, message_size)

  def setup(self, in_dim, message_size,shared_params=None):
    self.in_dim = in_dim
    cname = "all_loss_in"

    op = self.fw_op
    self.states_names = op.states_names
    bw_in_dim = len(self.states_names) + 1
    self.fw_arrow = Arrow([op])

    bw_ops = [
        StandardizeInputsAndStates(self.states_names),
        ConcatenateInputsAndStates(self.states_names, cname=cname),
    ]
    if self.stateful:
      bw_ops += [
          StatefulLearner(
              xname=cname, in_dim=bw_in_dim, hidden_n=self.stateful_hidden_n,
              out_message_size=message_size)
      ]
    else:
      bw_ops += [
        MessageGenerator(in_dim=bw_in_dim, xname=cname,
                         out_message_size=message_size),
      ]
    self.bw_arrow = Arrow(bw_ops)

    self.trainable_weights = self.fw_arrow.trainable_weights + \
        self.bw_arrow.trainable_weights
    self.weights = [self.fw_arrow.weights, self.bw_arrow.weights]

  def init(self, inner_batch_size=None):
    if self.stateful and inner_batch_size is not None:
      bw_carry = self.bw_arrow.init()[self.carry_name]
      bw_carry = tf.broadcast_to(
          bw_carry,
          (inner_batch_size, self.in_dim) + (bw_carry.shape[0],))
      return [bw_carry]
    # return an empty forward list.
    return []

  def get_trainable_weights(self):
    return self.trainable_weights

  def get_all_weights(self):
    return self.weights

  def load_weights(self, weights):
    for arrow, aw in zip([self.fw_arrow, self.bw_arrow], weights):
      arrow.load_weights(aw)

  def compute_loss(self, inputs, targets, eps=10e-8):
    arrow_in = (inputs, targets)

    # params is expected to be empty
    states = {}
    new_states, result, _ = self.fw_arrow(states, arrow_in, False)
    # unpack states
    last_in = tuple(new_states[k] for k in self.states_names)

    return result, last_in

  def backward(self, params, m_in, side_inputs):
    return self._backward( params, m_in, side_inputs, False)

  def backward_init(self, params, m_in, side_inputs):
    return self._backward(params, m_in, side_inputs, True)

  def _backward(self, params, m_in, side_inputs, initialize):
    last_in = side_inputs

    states = {k:v for k, v in zip(self.states_names, last_in)}
    if self.stateful:
      states[self.carry_name] = params[0]

    new_states, m_act, side_out = self.bw_arrow(states, m_in, initialize)

    if initialize:
      in_mean = side_out["inputs_mean"]
      in_norm = side_out["inputs_norm"]

    next_params = []
    if self.stateful:
      next_carry = new_states[self.carry_name]
      next_params.append(next_carry)

    if initialize:
      side_outputs = (in_mean, in_norm)
    else:
      side_outputs = ()
    return next_params, m_act, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    (in_mean, in_norm) = stats

    stats = {"inputs_mean": in_mean, "inputs_norm": in_norm}
    self.bw_arrow.update_statistics(stats, update_perc)


class MPCrossEntropyLoss(MPLoss):

  def __init__(self, in_dim=None, message_size=None, stateful=False,
               stateful_hidden_n=15):
    op_name = "celoss"
    op = CrossEntropyLoss(op_name)

    super().__init__(
        op, in_dim, message_size, stateful, stateful_hidden_n)


class MPL1Loss(MPLoss):

  def __init__(self, in_dim=None, message_size=None, stateful=False,
               stateful_hidden_n=15):
    op_name = "l1loss"
    op = L1Loss(op_name)

    super().__init__(
        op, in_dim, message_size, stateful, stateful_hidden_n)


class MPL2Loss(MPLoss):

  def __init__(self, in_dim=None, message_size=None, stateful=False,
               stateful_hidden_n=15):
    op_name = "l2loss"
    op = L2Loss(op_name)

    super().__init__(
        op, in_dim, message_size, stateful, stateful_hidden_n)


class MPNetwork():

  def __init__(self, layers, loss_layer):
    self.layers = layers
    self.loss_layer = loss_layer

  def _init_nested(self, inner_batch_size=None):
    fw_result = []
    for l in self.layers + [self.loss_layer]:
      fw_p = l.init(inner_batch_size)
      fw_result.append(fw_p)

    return fw_result

  def setup(self, in_dim, message_size, inner_batch_size=None,
            shared_params=None):
    """Keras-style autmatic setup of all layers inside the network.
    It needs message_size, and the input size in_dim, for the first layer.
    It is likely to differ from Keras actual style, since I didn't read their
    code.

    This code then also runs _init_nested().
    """
    self.shared_params = shared_params
    for l in self.layers + [self.loss_layer]:
      l.setup(in_dim, message_size, shared_params)
      if hasattr(l, "out_dim"):
        in_dim = l.out_dim

    # This is needed to unpack lists of tensors into layered lists.
    self.structure = self._init_nested(inner_batch_size)
    # This is needed to deserialize single vectors into lists of tensors.
    example_pfw = self.init(inner_batch_size)
    self.pfw_sizes = [
        int(tf.reshape(e, [-1]).shape[0]) for e in example_pfw]
    self.pfw_shapes = [e.shape for e in example_pfw]

    # This is needed to deserialized saved checkpoints.
    self.all_weights_structure = [
        l.get_all_weights() for l in self.layers + [self.loss_layer]]
    if self.shared_params is not None:
      self.all_weights_structure += [l.weights for l in self.shared_params.values()]
    all_weights_flat = tf.nest.flatten(self.all_weights_structure)
    self.all_weights_flat_shapes = [t.shape for t in all_weights_flat]
    print("All parameters:", self.all_weights_flat_shapes)

    self.all_weights_flat_sizes = [
        int(tf.reshape(t, [-1]).shape[0]) for t in all_weights_flat]

  def init(self, inner_batch_size=None):
    return tf.nest.flatten(self._init_nested(inner_batch_size))

  def get_trainable_weights(self):
    tr_w = [l.get_trainable_weights() for l in self.layers + [self.loss_layer]]
    if self.shared_params is not None:
      tr_w += [l.trainable_weights for l in self.shared_params.values()]
    return flatten(tr_w)

  def serialize_pfw(self, pfw):
    """Serializes the pfw. It assumes they are already flattened."""
    return tf.concat([tf.reshape(e, [-1]) for e in pfw], 0)

  def deserialize_pfw(self, pfw_f):
    """Deserializes the pfw."""
    pfw_f_split = tf.split(pfw_f, self.pfw_sizes)
    return [tf.reshape(e, s) for e, s in zip(pfw_f_split, self.pfw_shapes)]

  def forward(self, fw_params, inputs):
    return self._forward(fw_params, inputs)

  def _forward(self, fw_params, inputs):
    fw_params = tf.nest.pack_sequence_as(self.structure, fw_params)
    fw_params = fw_params[:-1] # remove loss params (used in bw computations).
    assert len(fw_params) == len(self.layers)
    res = inputs
    side_outputs = []
    for ps, l in zip(fw_params, self.layers):
      res, side_outputs_l = l.forward(ps, res)
      side_outputs.append(side_outputs_l)
    return res, side_outputs

  def forward_init(self, fw_params, inputs):
    fw_params = tf.nest.pack_sequence_as(self.structure, fw_params)
    loss_params = fw_params[-1]
    fw_params = fw_params[:-1] # remove loss params (used in bw computations).
    assert len(fw_params) == len(self.layers)
    res = inputs
    updated_params = []
    side_outputs = []
    for ps, l in zip(fw_params, self.layers):
      res, side_outputs_l, new_ps = l.forward_init(ps, res)
      updated_params.append(new_ps)
      side_outputs.append(side_outputs_l)
    # Add loss params - they don't need a forward_init.
    updated_params.append(loss_params)
    updated_params = tf.nest.flatten(updated_params)
    return res, side_outputs, updated_params

  def compute_loss(self, predictions, targets):
    return self.loss_layer.compute_loss(predictions, targets)

  def compute_deltas_loss(self, deltas):
    l2_losses = 0.
    for d in flatten(deltas):
      l2_losses += tf.reduce_mean(d*d)
    l2_losses /= 2
    return l2_losses

  def backward_init(self, fw_params, loss, side_inputs):
    return self._backward(fw_params, loss, side_inputs, True)

  def backward(self, fw_params, loss, side_inputs):
    return self._backward(fw_params, loss, side_inputs, False)

  def _backward(self, fw_params, loss, side_inputs, initialize):
    bw_name = "backward_init" if initialize else "backward"
    fw_params = tf.nest.pack_sequence_as(self.structure, fw_params)

    # we want [batch_size, ..., output_size, 1]
    m_in = tf.expand_dims(loss, -1)

    # special treatment to the loss
    updated_fw_params = []
    side_outs = []
    updated_fw_params_loss, m_in, side_l = getattr(
        self.loss_layer, bw_name)(fw_params[-1], m_in, side_inputs[-1])
    updated_fw_params.insert(0, updated_fw_params_loss)
    side_outs.insert(0, side_l)

    for fw_ps, l, side_inputs_l in reversed(list(zip(
        fw_params[:-1], self.layers, side_inputs[:-1]))):
      new_fw_ps, m_in, side_l = getattr(l, bw_name)(
          fw_ps, m_in, side_inputs_l)
      updated_fw_params.insert(0, new_fw_ps)
      side_outs.insert(0, side_l)

    updated_fw_params = tf.nest.flatten(updated_fw_params)

    return updated_fw_params, side_outs

  def minibatch_init(self, x, y, inner_batch_size=None, pfw=None):
    """Initializes important parameters that require a minibatch.
    For instance, it initializes weight-norm-like parameters.
    It also standardizes inputs in the backward pass.
    Remember to run "update_statistics" afterwards, since some layers
    require a second step for updating their values.
    """
    if pfw is None:
      pfw = self.init(inner_batch_size)

    # This would initialize weight-norm-like forward params.
    prediction, side_outputs, pfw = self.forward_init(pfw, x)
    act_loss, side_op_loss = self.compute_loss(prediction, y)
    side_outputs.append(side_op_loss)
    # This will initialize some scaling parameters to ensure similar magnitudes
    # in the backward pass.
    pfw, side_outputs = self.backward_init(pfw, act_loss, side_outputs)
    return pfw, side_outputs

  def update_statistics(self, all_stats, update_perc=1.):
    stats_f = [tf.nest.flatten(s) for s in all_stats]
    acc_stats_f = [tf.reduce_mean(s) for s in zip(*stats_f)]
    acc_stats = tf.nest.pack_sequence_as(all_stats[0], acc_stats_f)
    for l, s in zip(self.layers + [self.loss_layer], acc_stats):
      l.update_statistics(s, update_perc)

  @tf.function
  def inner_update(self, pfw, x, y):
    """Utility function to generate an update.
    It is like performing and end-to-end forward-backward pass.
    """
    print("compiling network inner_update.")

    prediction, side_outputs = self.forward(pfw, x)
    loss, side_outputs_l = self.compute_loss(prediction, y)
    side_outputs.append(side_outputs_l)

    pfw, deltas = self.backward(pfw, loss, side_outputs)

    return pfw, deltas

  def save_weights(self, base_fn, step):
    """Saves all the weights of the network."""

    # We need to extract all weights.
    # Notice this is different from the function get_weights, which only returns
    # trainable weights and its purpose is for training steps only.

    network_weights = [l.get_all_weights() for l in self.layers + [self.loss_layer]]
    if self.shared_params is not None:
      network_weights += [l.weights for l in self.shared_params.values()]

    network_weights = tf.nest.flatten(network_weights)
    network_weights = tf.concat(
        [tf.reshape(e, [-1]) for e in network_weights], 0)

    # Then serialize them all in one big array and store it.
    filename = base_fn + "_{:08d}.npy".format(step)
    with gfile.GFile(filename, "wb") as fout:
      np.save(fout, network_weights.numpy())

  def load_weights(self, base_fn):
    """Find the latest checkpoint matching base_fn, and load the weights."""

    matcher = base_fn + "_*.npy"
    filenames = sorted(gfile.glob(matcher), reverse=True)
    assert len(filenames) > 0, "No files matching {}".format(matcher)
    filename = filenames[0]

    # load array
    with gfile.GFile(filename, "rb") as fin:
      serialized_weights = np.load(fin)

    print(serialized_weights.shape, self.all_weights_flat_sizes)
    all_weights_flat_split = tf.split(serialized_weights,
                                      self.all_weights_flat_sizes)
    all_weights_flat = [tf.reshape(t, s) for t, s in zip(
        all_weights_flat_split, self.all_weights_flat_shapes)]

    all_weights = tf.nest.pack_sequence_as(self.all_weights_structure,
                                           all_weights_flat)

    all_layers = self.layers + [self.loss_layer]
    if self.shared_params is not None:
      all_layers += list(self.shared_params.values())
    for l, lw in zip(all_layers, all_weights):
      l.load_weights(lw)

