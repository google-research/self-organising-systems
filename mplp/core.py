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
import copy
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from tensorflow.keras.layers import Dense

class Arrow():

  def __init__(self, ops):
    self.ops = ops

    self.trainable_weights = sum([op.trainable_weights for op in self.ops], [])
    # these keep their structure.
    self.weights = [op.weights for op in self.ops]

  def __call__(self, states, inputs, initialize):
    side_outputs = {}
    # perform ops sequentially.
    new_states = states
    outputs = inputs
    for op in self.ops:
      new_states, outputs, side_op = op(new_states, outputs, initialize)
      side_outputs.update(side_op)

    return new_states, outputs, side_outputs

  def init(self):
    all_states = {}
    for op in self.ops:
      all_states.update(op.init())
    return all_states

  def update_statistics(self, stats, update_perc=1.):
    for op in self.ops:
      op.update_statistics(stats, update_perc)

  def load_weights(self, weights):
    for op, opw in zip(self.ops, weights):
      op.load_weights(opw)


class ArrowOp():

  def __init__(self):
    self.trainable_weights = []
    self.weights = []

  def __call__(self, states, inputs, initialize):
    raise Exception("Not implemented yet!")

  def init(self):
    return {}

  def load_weights(self, weights):
    return

  def update_statistics(self, stats, update_perc=1.):
    return


class GRUBlock():

  def __init__(self, x_dim, carry_n,
               kernel_init=tf.keras.initializers.glorot_uniform(),
               bias_init=tf.keras.initializers.zeros()):
    self.carry_n = carry_n

    self.W_update_x = tf.Variable(kernel_init((x_dim, carry_n)))
    self.W_update_c = tf.Variable(kernel_init((carry_n, carry_n)))
    # update gate should initially favour new information.
    self.b_update = tf.Variable(tf.constant(-1., shape=(carry_n,)))
    self.W_reset_x = tf.Variable(kernel_init((x_dim, carry_n)))
    self.W_reset_c = tf.Variable(kernel_init((carry_n, carry_n)))
    # reset gate should initially favour reset.
    self.b_reset = tf.Variable(tf.constant(-1., shape=(carry_n,)))
    # traditional GRUs have linear matmuls only. However, it's too simple.
    # we therefore enhance them with small multi layer nns
    self.next_x_net = tf.keras.Sequential([
        Dense(80, activation=tf.tanh, input_shape=(x_dim,),
              kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(40, activation=tf.tanh,
              kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(carry_n, activation=None,
              kernel_initializer=kernel_init, use_bias=False),])
    self.next_c_net = tf.keras.Sequential([
        Dense(80, activation=tf.tanh, input_shape=(carry_n,),
              kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(40, activation=tf.tanh,
              kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(carry_n, activation=None,
              kernel_initializer=kernel_init, use_bias=False),])
    self.b_next = tf.Variable(bias_init((carry_n)))

    self.weights = [self.W_update_x, self.W_update_c, self.b_update,
                    self.W_reset_x, self.W_reset_c, self.b_reset,
                    self.b_next,
                    self.next_x_net.weights, self.next_c_net.weights,
                    ]

    self.trainable_weights =  [
        self.W_update_x, self.W_update_c, self.b_update,
        self.W_reset_x, self.W_reset_c, self.b_reset,
        self.b_next
    ] + self.next_x_net.trainable_weights + self.next_c_net.trainable_weights


  def init(self):
    """Return a zero initialized carry state.
    The current implementation returns a 1-dimensional vector. This has to be
    replicated for each synapse for each batch."""
    return tf.zeros([self.carry_n])


  def __call__(self, x, carry):
    update_t = tf.sigmoid(
        x @ self.W_update_x + carry @ self.W_update_c + self.b_update)
    reset_t = tf.sigmoid(
        x @ self.W_reset_x + carry @ self.W_reset_c + self.b_reset)
    new_carry = update_t * carry + (1. - update_t) * tf.tanh(
        self.next_x_net(x) + self.next_c_net(reset_t * carry) + self.b_next)
    return new_carry

  def load_weights(self, weights):
    return self.set_weights(weights)

  def set_weights(self, weights):
    """Loads weights manually. It has the same name of a keras network."""
    [v.assign(w) for v, w in zip(self.weights[:-2], weights[:-2])]
    self.next_x_net.set_weights(weights[-2])
    self.next_c_net.set_weights(weights[-1])


class OutStandardizer():

  def __init__(self, scale_init_val):
    self.out_scale = tf.Variable(scale_init_val)
    self.out_mean = tf.Variable(0., dtype=tf.float32)  # not initialized
    # record that this is not initialized.
    self.is_initialized = False

    self.weights = [self.out_scale, self.out_mean]
    self.trainable_weights =  [self.out_scale]

  def __call__(self, x, initialize):
    assert (initialize or self.is_initialized), "OutStandardizer not initialized"

    side_outputs = {}
    if initialize:
      # We want the output to be zero-centered at the beginning.
      # This is a constant, for now, so it doesn't get trained.
      out_mean = tf.reduce_mean(x)
      side_outputs["out_mean"] = out_mean
      x -= out_mean
    else:
      x -= self.out_mean
    x *= self.out_scale
    return x, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    u_out_mean = self.out_mean * (1. - update_perc) + \
        stats["out_mean"] * update_perc
    self.out_mean.assign(u_out_mean)
    self.is_initialized = True

  def load_weights(self, weights):
    self.out_scale.assign(weights[0])
    self.out_mean.assign(weights[1])

    # We are assuming that we load weights of something already initialized.
    # We could test it, but it would be an overkill.
    self.is_initialized = True


class StatefulLearner(ArrowOp):

  def __init__(self, xname, in_dim=None, hidden_n=None, shared_network=None,
               pname=None, scale_init_val=None, shared_out_std=None,
               is_residual=True, out_message_size=None,):
    """This op can perform either an update of the input params (and/or)
    generate the next message to pass.
    This op does not know the actual params structure, and it assumes
    an input batch. At the end, the model aggregates the results over the batch
    and preserves all the other input dimensions for pname.

    if is_residual is True, the param update is computed as an increment.
    Otherwise, it is recomputed entirely.

    you can call this learner in one of two ways:
    - set in_dim, hidden_n, and don't set shared_network. This creates a new GRU
    - set shared_network to a previously initialized GRU. this shares parameters.

    Likewise for shared_out_std and scale_init_val.

    hidden_n is the number of carry states not including eventual param &
    message outputs.

    If the learner needs to update a param, pname and scale_init_val must be set
    If the learner needs to compute a message, out_message_size must be set.
    """
    super().__init__()
    assert ((pname is None) or (
        scale_init_val is not None or shared_out_std is not None))
    assert ((pname is not None) or (out_message_size is not None))

    self.pname = pname
    self.xname = xname
    self.cname = "carry"
    self.is_residual = is_residual
    self.out_message_size = out_message_size

    if shared_network is None:
      self.has_shared_network = False

      carry_n = (0 if pname is None else 1) + \
          hidden_n + (
          0 if out_message_size is None else out_message_size)

      self.network = GRUBlock(in_dim, carry_n)
      self.weights = [self.network.weights]
      # Nice way to create a shallow copy.
      self.trainable_weights = [] + self.network.trainable_weights
    else:
      self.has_shared_network = True
      self.network = shared_network
      # we do not send weights from here.
      self.weights = []
      self.trainable_weights = []

    if pname is not None:
      if shared_out_std is None:
        self.has_shared_out_std = False
        self.out_standardizer = OutStandardizer(scale_init_val)
        self.weights += self.out_standardizer.weights
        self.trainable_weights += self.out_standardizer.trainable_weights
      else:
        self.has_shared_out_std = True
        self.out_standardizer = shared_out_std

  def __call__(self, states, inputs, initialize):
    """Perform network update rule: it may be either param update and/or
    generating a message.
    Notice how, since TF allows matrix multiplications of higher dimensions:
    card(x) >= 2
    Everything still works. In particular, we can run batches, and then just
    reduce mean across batches without any reshape required!
    """
    x = states[self.xname]
    carry = states[self.cname]
    next_carry = self.network(x, carry)

    new_states = copy.copy(states)
    new_states[self.cname] = next_carry

    side_outputs = {}

    if self.pname is not None:
      # the first channel output is the param update result.
      y = next_carry[..., 0]
      # Average y from the batch
      y = tf.reduce_mean(y, axis=0)

      y, side_out_std = self.out_standardizer(y, initialize)
      if initialize:
        side_outputs[self.pname + "_out_mean"] = side_out_std["out_mean"]

      new_p = y
      if self.is_residual:
        new_p += states[self.pname]

        side_outputs[self.pname + "_delta"] = y

      new_states[self.pname] = new_p

    if self.out_message_size is not None:
      # the last channels are the out_message.
      outputs = next_carry[..., -self.out_message_size:]
      # WARNING! This may require refactoring once we generalize to any graphs.
      # We always want to return a tensor like [bs, in_dims, out_message_size].
      # Therefore, if the rank of y is >3, we reduce_mean it.
      if len(outputs.shape) > 3:
        all_dims_idx = list(range(len(outputs.shape)))
        dims_to_reduce = all_dims_idx[2:-1]
        outputs = tf.reduce_mean(outputs, axis=dims_to_reduce)

    else:
      outputs = inputs

    return new_states, outputs, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    if self.pname is not None:
      in_stats = {"out_mean": stats[self.pname + "_out_mean"]}
      self.out_standardizer.update_statistics(in_stats, update_perc)

  def init(self):
    """Initialize a carry.
    Currently, we return a 1d carry that needs to be Replicated accordingly.
    """
    carry = self.network.init()
    return {self.cname: carry}

  def load_weights(self, weights):
    # this may break if the weights passed are a subset of 3.
    if not self.has_shared_network:
      self.network.set_weights(weights[0])

    if self.pname is not None and not self.has_shared_out_std:
      st_idx = 1 if not self.has_shared_network else 0
      self.out_standardizer.load_weights(weights[st_idx:])


class ParamUpdate(ArrowOp):

  def __init__(self, pname, in_dim, xname, scale_init_val, is_residual):
    """This is the op that performs an update of the input params.
    This op does not know the actual params structure, and it assumes
    an input batch. At the end, the model aggregates the results over the batch
    and preserves all the other input dimensions for pname.

    if is_residual is True, the param update is computed as an increment.
    Otherwise, it is recomputed entirely.
    """
    super().__init__()
    self.pname = pname
    self.xname = xname

    self.is_residual = is_residual

    # update network:
    self.update_rule = tf.keras.Sequential([
        Dense(80, activation=tf.nn.relu, input_shape=(in_dim,)),
        Dense(40, activation=tf.nn.relu),
        Dense(1, activation=tf.nn.tanh),])
    self.out_scale = tf.Variable(scale_init_val)
    self.out_mean = tf.Variable(0., dtype=tf.float32)  # not initialized
    # record that this is not initialized.
    self.is_initialized = False

    self.weights = [self.update_rule.weights, self.out_scale, self.out_mean]
    self.trainable_weights = self.update_rule.trainable_weights + \
        [self.out_scale]

  def __call__(self, states, inputs, initialize):
    """Perform a weight update rule.
    Notice how, since TF allows matrix multiplications of higher dimensions:
    card(x) >= 2
    Everything still works. In particular, we can run batches, and then just
    reduce mean across batches without any reshape required!
    """
    assert initialize or self.is_initialized, "ParamUpdate op not initialized"
    x = states[self.xname]

    y = self.update_rule(x)
    # there's an extra dimension we should get rid of now.
    y = tf.squeeze(y, axis=-1)
    # Average y from the batch
    y = tf.reduce_mean(y, axis=0)

    side_outputs = {}

    if initialize:
      # We want the output to be zero-centered at the beginning.
      # This is a constant, for now, so it doesn't get trained.
      out_mean = tf.reduce_mean(y)
      side_outputs[self.pname + "_out_mean"] = out_mean
      y -= out_mean
    else:
      y -= self.out_mean
    y *= self.out_scale

    new_p = y
    if self.is_residual:
      new_p += states[self.pname]
      side_outputs[self.pname + "_delta"] = y

    new_states = copy.copy(states)
    new_states[self.pname] = new_p

    return new_states, inputs, side_outputs

  def update_statistics(self, stats, update_perc=1.):
    u_out_mean = self.out_mean * (1. - update_perc) + \
        stats[self.pname + "_out_mean"] * update_perc
    self.out_mean.assign(u_out_mean)
    self.is_initialized = True

  def init(self):
     """Initialize a carry for LSTM, if we use stateful meta-learners"""
     # for now, return nothing, since we haven't implemented LSTMs.
     return {}

  def load_weights(self, weights):
    # keras networks require to pass a list of arrays:
    self.update_rule.set_weights(weights[0])

    self.out_scale.assign(weights[1])
    self.out_mean.assign(weights[2])

    # We are assuming that we load weights of something already initialized.
    # We could test it, but it would be an overkill.
    self.is_initialized = True


class MessageGenerator(ArrowOp):

  def __init__(self, in_dim, xname, out_message_size):
    """This is the op that generates a new message to pass to the connected
    nodes.
    This op assumes an input batch, and outputs batch_n params_updates.
    Currently, there is no use case where we do not aggregate the messages
    afterwards to be of [batch_size, in_dim, out_message_size] size.
    Therefore, we currently perform this step here and return such messages.
    """
    super().__init__()

    self.xname = xname

    self.mp_net = tf.keras.Sequential([
        Dense(80, activation=tf.nn.relu, input_shape=(in_dim,)),
        Dense(40, activation=tf.nn.relu),
        Dense(out_message_size, activation=tf.nn.tanh),])

    self.weights = self.mp_net.weights
    self.traianble_weights = self.mp_net.trainable_weights


  def __call__(self, states, inputs, initialize):
    """Perform a MP generating rule.
    Notice how, since TF allows matrix multiplications of higher dimensions:
    card(x) >= 2
    Everything still works. In particular, we can run batches without any
    reshape required!
    """
    x = states[self.xname]

    y = self.mp_net(x)

    # We always want to return a tensor like [bs, in_dims, out_message_size].
    # Therefore, if the rank of y is >3, we reduce_mean it.
    if len(y.shape) > 3:
      all_dims_idx = list(range(len(y.shape)))
      dims_to_reduce = all_dims_idx[2:-1]
      y = tf.reduce_mean(y, axis=dims_to_reduce)

    return states, y, {}

  def init(self):
    """Initialize a carry for LSTM, if we use stateful meta-learners"""
    # for now, return nothing, since we haven't implemented LSTMs.
    return {}

  def load_weights(self, weights):
    # keras networks require to pass a list of arrays:
    self.mp_net.set_weights(weights)


class MatMul(ArrowOp):

  def __init__(self, pname, shape, p_sd):
    super().__init__()
    self.pname = pname
    self.shape = shape
    self.p_sd = p_sd
    self.pin_name = pname + "_in"

  def __call__(self, states, inputs, initialize):
    new_states = copy.copy(states)
    new_states[self.pin_name] = inputs
    outputs = tf.matmul(inputs, states[self.pname])

    return new_states, outputs, {}

  def init(self):
    return {self.pname: tf.random.normal(self.shape, stddev=self.p_sd)}


class Add(ArrowOp):

  def __init__(self, pname, shape):
    super().__init__()
    self.pname = pname
    self.shape = shape
    self.pin_name = pname + "_in"

  def __call__(self, states, inputs, initialize):
    new_states = copy.copy(states)
    new_states[self.pin_name] = inputs
    outputs = inputs + states[self.pname]

    return new_states, outputs, {}

  def init(self):
    return {self.pname: tf.zeros(self.shape)}


class PairwiseOp(ArrowOp):

  def __init__(self, op_in_name, op):
    super().__init__()
    self.op_in_name = op_in_name
    self.op = op

  def __call__(self, states, inputs, initialize):
    new_states = copy.copy(states)
    new_states[self.op_in_name] = inputs
    outputs = self.op(inputs)

    return new_states, outputs, {}


class Softmax(ArrowOp):

  def __init__(self, name):
    super().__init__()

    self.states_names = [
        name + "_" + suff for suff in (
            "inputs", "translated_inputs", "exp_nom", "exp_denom")]

  def __call__(self, states, inputs, initialize):

    translated_inputs = inputs - tf.math.reduce_max(
        inputs, axis=-1, keepdims=True)

    exp_nom = tf.exp(translated_inputs)
    exp_denom = tf.reduce_sum(exp_nom, axis=-1)
    # for convenience, we manually broadcast this.
    exp_denom = tf.repeat(exp_denom[:, tf.newaxis], exp_nom.shape[-1], axis=1)

    outputs = exp_nom / exp_denom

    new_states_t = (inputs, translated_inputs, exp_nom, exp_denom)

    new_states = copy.copy(states)
    for k, v in zip(self.states_names, new_states_t):
      new_states[k] = v

    return new_states, outputs, {}


class CrossEntropyLoss(ArrowOp):

  def __init__(self, name, eps=10e-8):
    super().__init__()
    self.eps = eps

    self.states_names = [name + "_" + suff for suff in (
        "x", "targets", "log_in")]

  def __call__(self, states, inputs, initialize):
    x, targets = inputs

    x = tf.clip_by_value(x, self.eps, 1. - self.eps)
    log_in = tf.math.log(x)
    outputs = -(targets * log_in)

    new_states_t = (x, targets, log_in)

    new_states = copy.copy(states)
    for k, v in zip(self.states_names, new_states_t):
      new_states[k] = v

    return new_states, outputs, {}


class L1Loss(ArrowOp):

  def __init__(self, name):
    super().__init__()

    self.states_names = [name + "_" + suff for suff in ("x", "targets")]

  def __call__(self, states, inputs, initialize):
    x, targets = inputs

    outputs = tf.abs(targets - x)

    new_states_t = (x, targets)


    new_states = copy.copy(states)
    for k, v in zip(self.states_names, new_states_t):
      new_states[k] = v

    return new_states, outputs, {}


class L2Loss(ArrowOp):

  def __init__(self, name):
    super().__init__()

    self.states_names = [name + "_" + suff for suff in ("x", "targets", "diff")]

  def __call__(self, states, inputs, initialize):
    x, targets = inputs

    diff = targets - x
    outputs = (diff**2) / 2

    new_states_t = (x, targets, diff)

    new_states = copy.copy(states)
    for k, v in zip(self.states_names, new_states_t):
      new_states[k] = v

    return new_states, outputs, {}


class RepeatInputs(ArrowOp):

  def __init__(self, repeats, axis):
    super().__init__()
    self.repeats = repeats
    self.axis = axis

  def __call__(self, states, inputs, initialize):
    outputs = tf.repeat(tf.expand_dims(inputs, self.axis),
                        self.repeats, axis=self.axis)

    return states, outputs, {}


class RepeatState(ArrowOp):

  def __init__(self, sname, repeats, axis):
    super().__init__()
    self.sname = sname
    self.repeats = repeats
    self.axis = axis

  def __call__(self, states, inputs, initialize):
    new_states = copy.copy(states)

    new_states[self.sname] = tf.repeat(
        tf.expand_dims(states[self.sname], self.axis),
        self.repeats, axis=self.axis)

    return new_states, inputs, {}


class RepeatForBatch(ArrowOp):

  def __init__(self, sname, out_name):
    super().__init__()
    self.sname = sname
    self.out_name = out_name

  def __call__(self, states, inputs, initialize):
    """The batch size is extracted from inputs. Therefore, beware of
    transformations that flatten inputs."""
    new_states = copy.copy(states)
    new_states[self.out_name] = tf.repeat(
        tf.expand_dims(states[self.sname], 0),
        inputs.shape[0], axis=0)
    return new_states, inputs, {}


class StandardizeInputsAndStates(ArrowOp):

  def __init__(self, s_names):
    super().__init__()
    self.s_names = s_names

    # define the variables we will later update.
    self.norm = [tf.Variable(1., dtype=tf.float32)  for _ in ["inputs"] + s_names]
    self.mean = [tf.Variable(0., dtype=tf.float32)  for _ in ["inputs"] + s_names]
    # record that this is not initialized.
    self.is_initialized = False

    # keep the structure.
    self.weights = [self.norm, self.mean]

  def __call__(self, states, inputs, initialize):
    assert initialize or self.is_initialized, "StandardizeInputsAndStates op not initialized"
    all_inputs = [inputs] + [states[sname] for sname in self.s_names]

    side_outputs = {}

    if initialize:
      mean = self._getMeanZeroer(all_inputs)
      side_outputs["inputs_mean"] = mean
      norm = self._getNormalizer(all_inputs)
      side_outputs["inputs_norm"] = norm
    else:
      mean = self.mean
      norm = self.norm
    all_inputs = self._translateValues(all_inputs, mean)
    all_inputs = self._scaleValues(all_inputs, norm)
    new_states = copy.copy(states)
    for sname, state in zip(self.s_names, all_inputs[1:]):
      new_states[sname] = state
    outputs = all_inputs[0]

    return new_states, outputs, side_outputs

  def _scaleValues(self, inputs, norm):
    assert len(norm) == len(inputs), (
        "Invalid lengths in scaleValues:{} vs {}".format(
            len(norm), len(inputs)))
    return [p * i for p, i in zip(norm, inputs)]

  def _translateValues(self, inputs, mean):
    assert len(mean) == len(inputs), (
        "Invalid lengths in translateValues:{} vs {}".format(
            len(mean), len(inputs)))
    # we use + cause we compute the mean zerorer..
    return [p + i for p, i in zip(mean, inputs)]

  def _getNormalizer(self, inputs, eps=1e-5):
    def _normalizer(x):
      std = tf.math.reduce_std(x)
      return tf.cond(std < eps,
                     lambda: 1.,  # if true
                     lambda: 1./std)        # if false
    return [_normalizer(i) for i in inputs]

  def _getMeanZeroer(self, inputs):
    return [-tf.reduce_mean(i) for i in inputs]

  def update_statistics(self, stats, update_perc=1.):
    new_mean = stats["inputs_mean"]
    for v, m in zip(self.mean, new_mean):
      u_mean = v * (1. - update_perc) + m * update_perc
      v.assign(u_mean)

    new_norm = stats["inputs_norm"]
    for v, no in zip(self.norm, new_norm):
      u_norm = v * (1. - update_perc) + no * update_perc
      v.assign(u_norm)
    self.is_initialized = True

  def load_weights(self, weights):
    [v.assign(w) for v, w in zip(self.norm, weights[0])]
    [v.assign(w) for v, w in zip(self.mean, weights[1])]

    # We are assuming that we load weights of something already initialized.
    # We could test it, but it would be an overkill.
    self.is_initialized = True


class StandardizeInputsAndStatesLink(ArrowOp):

  def __init__(self, shared_standardizer):
    """This class wraps a StandardizeInputsAndStates class. But this does not
    own any variable, ie it does not have weights, trainable_weights."""
    super().__init__()
    self.standardizer = shared_standardizer

  def __call__(self, states, inputs, initialize):
    return self.standardizer(states, inputs, initialize)

  def update_statistics(self, stats, update_perc=1.):
    return self.standardizer.update_statistics(stats, update_perc)


class StandardizeInputsAndStatesOnline(ArrowOp):

  def __init__(self, s_names):
    super().__init__()
    self.s_names = s_names

  def __call__(self, states, inputs, initialize):
    all_inputs = [inputs] + [states[sname] for sname in self.s_names]


    def _normalizer(x):
      std = tf.math.reduce_std(x)
      return tf.cond(std < 1e-5,
                     lambda: 1.,  # if true
                     lambda: 1./std)        # if false

    all_inputs = [i - tf.reduce_mean(i) for i in all_inputs]
    all_inputs = [i * _normalizer(i) for i in all_inputs]

    new_states = copy.copy(states)
    for sname, state in zip(self.s_names, all_inputs[1:]):
      new_states[sname] = state
    outputs = all_inputs[0]

    return new_states, outputs, {}


class ConcatenateInputsAndStates(ArrowOp):

  def __init__(self, s_names, cname):
    super().__init__()
    self.s_names = s_names
    self.cname = cname

  def __call__(self, states, inputs, initialize):
    new_states = copy.copy(states)

    all_inputs = [inputs] + [
        tf.expand_dims(states[sname], -1) for sname in self.s_names]

    new_states[self.cname] = tf.concat(all_inputs, -1)

    return new_states, inputs, {}


class ConcatenateStates(ArrowOp):
  # this op, as opposed to the one above, assumes the dimensionality is already
  # correct!

  def __init__(self, s_names, cname):
    super().__init__()
    self.s_names = s_names
    self.cname = cname

  def __call__(self, states, inputs, initialize):
    new_states = copy.copy(states)

    all_inputs = [states[sname] for sname in self.s_names]

    new_states[self.cname] = tf.concat(all_inputs, -1)

    return new_states, inputs, {}

