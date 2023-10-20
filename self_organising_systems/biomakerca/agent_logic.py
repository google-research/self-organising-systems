"""Module containing the agent logic interface and an example implementation.

Copyright 2023 Google LLC

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
from abc import ABC, abstractmethod

from evojax.util import get_params_format_fn
import jax
from jax import vmap
from jax.nn.initializers import glorot_normal
import jax.numpy as jp
import jax.random as jr

from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.env_logic import AgentProgramType
from self_organising_systems.biomakerca.env_logic import ExclusiveInterface
from self_organising_systems.biomakerca.env_logic import KeyType
from self_organising_systems.biomakerca.env_logic import PerceivedData
from self_organising_systems.biomakerca.env_logic import ReproduceInterface
from self_organising_systems.biomakerca.env_logic import SpawnOpData
from self_organising_systems.biomakerca.env_logic import ParallelInterface
from self_organising_systems.biomakerca.environments import EnvConfig
from self_organising_systems.biomakerca.utils import stringify_class


class AgentLogic(ABC):
  """Interface of all agent logics.
  
  The abstract methods need to be implemented in order to allow for 
  in-environment operations, through the method step_maker.step_env.
  """

  @abstractmethod
  def initialize(self, key: KeyType) -> AgentProgramType:
    """Initialize logic parameters.
    
    Return a one-dimensional array of parameters.
    """
    pass

  @abstractmethod
  def split_params_f(
      self, params: AgentProgramType
      ) -> tuple[AgentProgramType, AgentProgramType, AgentProgramType]:
    """Given all parameters, separate them into par_f params, excl_f params, and repr_f params.
    
    p must be one-dimensional.
    """
    pass

  @abstractmethod
  def par_f(
      self, key: KeyType, perc: PerceivedData, params: AgentProgramType
      ) -> ParallelInterface:
    """Perform a parallel function.
    
    params must be only the parallel params, not all of them.
    
    Return a ParallelInterface.
    """
    pass
  
  @abstractmethod
  def excl_f(self, key: KeyType, perc: PerceivedData, params: AgentProgramType
             ) -> ExclusiveInterface:
    """Perform a exclusive function.
    
    params must be only the exclusive params, not all of them.
    
    Return a ExclusiveInterface.
    """
    pass
  
  @abstractmethod
  def repr_f(self, key: KeyType, perc: PerceivedData, params: AgentProgramType
             ) -> ReproduceInterface:
    """Perform a reproduce function.
    
    params must be only the reproduce params, not all of them.
    
    Return a ReproduceInterface.
    """
    pass

  def __str__(self):
    return stringify_class(self)


def clip_residual(s, ds, clip_val):
  """Correct ds so that the final value of s + ds is within [-clip_val,clip_val].
  """
  new_s = s + ds
  gt = new_s > clip_val
  ltm = new_s < -clip_val
  ds = ds - (new_s - clip_val) * gt - (new_s + clip_val) * ltm
  return ds


class BasicAgentLogic(AgentLogic):
  """Example implementation of an AgentLogic.
  
  This model has two main settings, controlled by minimal_net.
  If minimal_net is True, the logic will be very simple, using only low hundreds
  of parameters. Otherwise, the logic will be potentially very complex, using
  low tens of thousands of parameters.
  
  All parameters are repeated once for each cell specialization, so that it is
  easier to write down explicit rules for each state.
  
  Regardless of the above flag, the model will be initialized with some fine
  tuned parameters, such that agents can grow and reproduce in most environments
  out of the box. When minimal_net is False, we have residual networks that
  take more inputs and are more complex whose goal is to modify the minimal net
  outputs.

  The cells can optionally perceive agent_ids. If so, they will never give
  nutrients to cells with different agent_ids.
  """

  def __init__(self, config: EnvConfig, perceive_ids=True, minimal_net=False):
    self.config = config
    # the types are perceived as one-hot vectors.
    self.n_types = len(config.etd.types.keys())
    self.n_spec = 4  # specializations of agents
    # Whether agent ids are perceivable by the agent.
    # if set to true, agents do not give energy to agents with different ids.
    self.perceive_ids = perceive_ids
    self.minimal_net = minimal_net
    self.state_clip_val = 3.

    ## Parallel op:
    # - dstate

    dsm_params = self.dsm_init(jax.random.PRNGKey(0))
    self.dsm_num_params, self._dsm_format_params_fn = get_params_format_fn(
        dsm_params
    )
    print('BasicAgentLogic.dsm_num_params = {}'.format(self.dsm_num_params))

    # - new_spec_logit: whether to change specialization.
    nsl_params = self.nsl_init(jax.random.PRNGKey(0))
    self.nsl_num_params, self._nsl_format_params_fn = get_params_format_fn(
        nsl_params
    )
    print('BasicAgentLogic.nsl_num_params = {}'.format(self.nsl_num_params))

    # - denergy_neigh: what nutrients to pass to neighbors.
    denm_params = self.denm_init(jax.random.PRNGKey(0))
    self.denm_num_params, self._denm_format_params_fn = get_params_format_fn(
        denm_params
    )
    print('BasicAgentLogic.denm_num_params = {}'.format(self.denm_num_params))

    ## Exclusive op:
    excl_params = self.excl_init(jax.random.PRNGKey(0))
    self.excl_num_params, format_params_fn = get_params_format_fn(excl_params)
    print('BasicAgentLogic.excl_num_params = {}'.format(self.excl_num_params))
    self._excl_format_params_fn = format_params_fn

    ## Reproduce op
    # very, very basic. We only check if the energy is sufficient, and if it is,
    # output to reproduce.
    repr_params = self.repr_init(jax.random.PRNGKey(0))
    self.repr_num_params, format_params_fn = get_params_format_fn(repr_params)
    print('BasicAgentLogic.repr_num_params = {}'.format(self.repr_num_params))
    self._repr_format_params_fn = format_params_fn

    # put all the parameters together now.
    self.num_params = (
        self.dsm_num_params
        + self.nsl_num_params
        + self.denm_num_params
        + self.excl_num_params
        + self.repr_num_params
    )
    print('BasicAgentLogic.num_params = {}'.format(self.num_params))

    # par_f has several subnetworks.
    self.split_par_params_f = lambda params: jp.split(
        params,
        (self.dsm_num_params, self.dsm_num_params + self.nsl_num_params),
        axis=-1,
    )

  def __str__(self):
    return stringify_class(self, include_list=["perceive_ids", "minimal_net"])

  def split_params_f(
      self, params: AgentProgramType
      ) -> tuple[AgentProgramType, AgentProgramType, AgentProgramType]:
    n_par_params = (
        self.dsm_num_params + self.nsl_num_params + self.denm_num_params
    )
    return jp.split(
        params,
        (n_par_params, n_par_params + self.excl_num_params), axis=-1)

  def dsm_init(self, key):
    if self.minimal_net:
      # in this case, the agent cannot modify its internal state.
      return (jp.empty(0),)
 
    # Set initial effect on state to zero.
    # a 2 layer NN.
    # # We look at proportions of neighboring cells and your internal state.
    # so: input:
    # - avg neigh types (n_types)
    # - personal state (env_state_size)
    insize = self.n_types + self.config.env_state_size
    hsize = 32
    # output: internal_state change (agent_state_size)
    outsize = self.config.agent_state_size
    ku, key = jr.split(key)
    dw0 = glorot_normal(batch_axis=0)(ku, (self.n_spec, insize, hsize))
    db0 = jp.zeros((self.n_spec, hsize))
    # output is defaulted to zero.
    dw1 = jp.zeros((self.n_spec, hsize, outsize))
    db1 = jp.zeros((self.n_spec, outsize))

    return (dw0, db0, dw1, db1)

  def dsm_f(self, params, avg_neigh_types, self_state, i):
    if self.minimal_net:
      return None
    dw0, db0, dw1, db1 = params
    x = jp.concatenate([avg_neigh_types, self_state], -1)
    out = jax.nn.relu(x @ dw0[i] + db0[i])
    out = out @ dw1[i] + db1[i]
    # to avoid explosion of states, we can't increase the final state to more 
    # than self.state_clip_val (abs).
    out = clip_residual(
        self_state[..., evm.A_INT_STATE_ST:], out, self.state_clip_val)
    return out

  def nsl_init(self, key):
    # Regardless of whether this is a minimal_net or not, the output is the same
    # output: specialization logits (n_spec)
    nsl_output_size = self.n_spec
    etd = self.config.etd
    spec_idxs = etd.specialization_idxs

    # Moreover, always have a simple linear layer with prewritten initialization
    # that determines how to change specialization.

    ## default initialization:
    # Initialize effect on average types in the neighborhood.
    div = 1.0
    # repeat for all blocks. No difference for now.
    b = jp.zeros([self.n_spec, nsl_output_size])
    w = jp.zeros([self.n_spec, self.n_types, nsl_output_size])
    # Earth discourages UNSPECIALIZED
    w = w.at[:, etd.types.EARTH, spec_idxs.AGENT_UNSPECIALIZED].set(-div)
    # Earth encourages ROOT
    w = w.at[:, etd.types.EARTH, spec_idxs.AGENT_ROOT].set(div)
    # Air discourages UNSPECIALIZED
    w = w.at[:, etd.types.AIR, spec_idxs.AGENT_UNSPECIALIZED].set(-div)
    # Air encourages LEAF
    w = w.at[:, etd.types.AIR, spec_idxs.AGENT_LEAF].set(div)
    # and so on...
    w = w.at[
        :, etd.types.AGENT_UNSPECIALIZED, spec_idxs.AGENT_UNSPECIALIZED
    ].set(div / 8.0)
    w = w.at[:, etd.types.AGENT_ROOT, spec_idxs.AGENT_UNSPECIALIZED].set(
        div / 8.0
    )
    w = w.at[:, etd.types.AGENT_LEAF, spec_idxs.AGENT_UNSPECIALIZED].set(
        div / 8.0
    )
    w = w.at[:, etd.types.AGENT_FLOWER, spec_idxs.AGENT_UNSPECIALIZED].set(
        div / 8.0
    )
    w = w.at[:, etd.types.EARTH, spec_idxs.AGENT_UNSPECIALIZED].set(
        -div / 8.0
    )
    w = w.at[:, etd.types.AIR, spec_idxs.AGENT_UNSPECIALIZED].set(
        -div / 8.0
    )
    # Flowers only grow if they are surrounded by leaves and some air.
    w = w.at[:, etd.types.AGENT_LEAF, spec_idxs.AGENT_FLOWER].set(div / 4.0)
    w = w.at[:, etd.types.AIR, spec_idxs.AGENT_FLOWER].set(div / 2.0)

    # If you are a flower, never change!
    w = w.at[spec_idxs.AGENT_FLOWER, :, spec_idxs.AGENT_FLOWER].set(div)

    if self.minimal_net:
      return w, b
    ## Further initialization
    # a 2 layer NN.
    # We look at proportions of neighboring cells and your internal state.
    # so: input:
    # - avg neigh types (n_types)
    # - personal state (env_state_size)
    insize = self.n_types + self.config.env_state_size
    hsize = 32
    outsize = nsl_output_size
    ku, key = jr.split(key)
    dw0 = glorot_normal(batch_axis=0)(ku, (self.n_spec, insize, hsize))
    db0 = jp.zeros((self.n_spec, hsize))
    # output is defaulted to zero.
    dw1 = jp.zeros((self.n_spec, hsize, outsize))
    db1 = jp.zeros((self.n_spec, outsize))

    return (w, b), (dw0, db0, dw1, db1)

  def nsl_f(self, params, avg_neigh_types, self_state, i):
    if self.minimal_net:
      w, b = params
    else:
      (w, b), (dw0, db0, dw1, db1) = params
    o0 = avg_neigh_types @ w[i] + b[i]
    if self.minimal_net:
      return o0
    o1 = jax.nn.relu(
        jp.concatenate([avg_neigh_types, self_state], -1) @ dw0[i] + db0[i]
    )
    o1 = o1 @ dw1[i] + db1[i]
    return o0 + o1

  def denm_init(self, key):
    """Initialization for denm_f.
    
    Please refer to the docstring on denm_f to understand what it does.
    """
    # - Output denergy for each neighbor. 9*2 (neighbors, energy)
    denm_output_size = 9 * 2
    
    # To enforce growth, keep more than what you need to spawn.
    keep_en = jp.zeros([self.n_spec, 2])
    keep_en = keep_en.at[:].set(
        self.config.dissipation_per_step * 6 + self.config.spawn_cost
        + self.config.specialize_cost * 2)
    # flowers keep a different amount of energy.
    keep_en = keep_en.at[self.config.etd.specialization_idxs.AGENT_FLOWER].set(
        self.config.dissipation_per_step * 7 + self.config.reproduce_cost
        + self.config.specialize_cost * 2)
    div = 1 / 10
    earth_nut_idx = 0
    air_nut_idx = 1
    # repeat for all blocks. No difference for now.
    # manipulate them in a different view, then reshape.
    b = jp.zeros([self.n_spec, 9, 2])
    # don't give earth nut down
    b = b.at[:, jp.array([0, 1, 2, 3, 5]), earth_nut_idx].set(div)
    # don't give air nut up
    b = b.at[:, jp.array([3, 5, 6, 7, 8]), air_nut_idx].set(div)

    b = b.reshape((self.n_spec, denm_output_size))

    if self.minimal_net:
      return (b, keep_en)

    # the extended model chooses individually what logit to give to each
    # neighbor (self included).

    # the operation is repeated for every neighbor. The input is their state
    # and our state, including a specialization one-hot for the target.
    # we don't need it for ourselves since the parameters are different already.
    insize = self.n_spec + 2 * self.config.env_state_size
    hsize = 32
    outsize = 2

    ku, key = jr.split(key)
    dw0 = glorot_normal(batch_axis=0)(ku, (self.n_spec, insize, hsize))
    db0 = jp.zeros((self.n_spec, hsize))
    # output is defaulted to zero.
    dw1 = jp.zeros((self.n_spec, hsize, outsize))
    db1 = jp.zeros((self.n_spec, outsize))

    return (b, keep_en, (dw0, db0, dw1, db1))

  def denm_f(self, params, norm_neigh_state, i, neigh_type, neigh_id, self_en):
    """Compute exactly how much energy is given to each neighbor.

    The op works this way: you keep a certain fixed amount of energy based on
    your specialization.
    Then, the rest is given based on a percentage distribution.
    you can also keep extra to yourself, by outputting a positive logit for your
    position (4).
    non-agent types are filtered out.
    If perceive_ids==True,  agents that are not of your ID are filtered out.
    
    The minimal model gives fixed percentages of energy based on the agent
    specialization.
    """
    if self.minimal_net:
      b, keep_en = params
    else:
      (b, keep_en, (dw0, db0, dw1, db1)) = params
    # only >= 0 values are valid here.
    denergy_neigh_logit = b[i].reshape((9, 2))

    if not self.minimal_net:
      # add output dependent on the inputs.
      norm_self_state = norm_neigh_state[4]
      # defaults to 0 if it is not an agent. It doesn't matter since nonagents
      # are later filtered out.
      neigh_spec = jax.nn.one_hot(
          self.config.etd.get_agent_specialization_idx(neigh_type), 4)

      def compute_logits_f(t_state, t_spec):
        inputs = jp.concatenate([norm_self_state, t_state, t_spec], -1)
        out = jax.nn.relu(inputs @ dw0[i] + db0[i])
        out = out @ dw1[i] + db1[i]
        return out

      inc_den_logit = vmap(compute_logits_f)(norm_neigh_state, neigh_spec)
      denergy_neigh_logit = denergy_neigh_logit + inc_den_logit

    denergy_neigh_logit = jax.nn.relu(denergy_neigh_logit)

    # now remove energy given to non agents.
    is_neigh_agent_fe = self.config.etd.is_agent_fn(neigh_type).astype(
        jp.float32)[:, None]
    denergy_neigh_logit = denergy_neigh_logit * is_neigh_agent_fe

    denergy_neigh_perc = denergy_neigh_logit / denergy_neigh_logit.sum(
        0, keepdims=True
    ).clip(1e-6)

    # get the available energy:
    tot_energy = (self_en - keep_en[i]).clip(0.0)
    # note that the computed amount to 'add' to self is not used anymore.
    denergy_neigh = (denergy_neigh_perc * tot_energy).at[4].set(0.)

    if self.perceive_ids:
      # Filter out energy given to agents that are not yourself.
      neigh_same_id = (neigh_id == neigh_id[4]).astype(jp.float32)
      denergy_neigh *= neigh_same_id[:, None]

    return denergy_neigh

  def excl_init(self, key):
    """Initialization for excl_f.
    
    Read excl_f docstring to understand what this does.
    """
    # default initialization for how much energy is needed to trigger spawn.
    # Note that it is a normalized value (by material_nutrient_cap).
    min_en_for_spawn = (
        self.config.dissipation_per_step * 4 + self.config.spawn_cost
        + self.config.specialize_cost * 2
    ) / self.config.material_nutrient_cap
    # default energy percentage to give to children.
    spawn_en_perc = jp.array([0.5])
    # These two parameters decide the likelihood of performing spawn. It
    # is a sigmoid with a high sensitivity, centered around having at most
    # slightly less than 3 agent neighbors.
    max_neigh_agent_avg = jp.array([3 / 8 - 0.1])
    spawn_prob_sensitivity = jp.array([30.])

    spec_idxs = self.config.etd.specialization_idxs
    # For sp_idx, we want to output 9 logits to sample randomly from.
    # all versions have a bias that initializes these logits in a smart way.
    logits_output_size = 9
    # each block has a different spawn strategy.
    b = jp.zeros([self.n_spec, logits_output_size])
    div = 0.5
    # unspecialized can spawn wherever
    b = b.at[spec_idxs.AGENT_UNSPECIALIZED, jp.array([0, 1, 2, 3, 5, 6, 7, 8])
             ].set(div)
    b = b.at[spec_idxs.AGENT_UNSPECIALIZED, 4].set(-div * 2)  # not self
    # root spawns preferably below
    b = b.at[spec_idxs.AGENT_ROOT, jp.array([3, 5, 6, 7, 8])].set(div)
    b = b.at[spec_idxs.AGENT_ROOT, 4].set(-div * 2)  # not self
    # leaf spawns preferably above
    b = b.at[spec_idxs.AGENT_LEAF, jp.array([0, 1, 2, 3, 5])].set(div)
    b = b.at[spec_idxs.AGENT_LEAF, 4].set(-div * 2)  # not self
    # flowers can spawn wherever.
    b = b.at[spec_idxs.AGENT_FLOWER, jp.array([0, 1, 2, 3, 5, 6, 7, 8])
             ].set(div)
    b = b.at[spec_idxs.AGENT_FLOWER, 4].set(-div * 2)  # not self

    minimal_params = (b, min_en_for_spawn, max_neigh_agent_avg,
                      spawn_prob_sensitivity, spawn_en_perc)
    if self.minimal_net:
      return minimal_params

    # Otherwise, create a deep neural network to modify the above logits and
    # to update states.
    # - inputs:
    # - avg_neigh_types
    # - self state
    # - dx, dy states
    insize = self.n_types + 3 * self.config.env_state_size
    hsize = 32
    # - outputs:
    # - 9 logits for each neighbor
    # - dstate_self
    # - dstate_child (from self)
    outsize = 9 + self.config.agent_state_size * 2

    ku, key = jr.split(key)
    dw0 = glorot_normal(batch_axis=0)(ku, (self.n_spec, insize, hsize))
    db0 = jp.zeros((self.n_spec, hsize))
    # output is defaulted to zero.
    dw1 = jp.zeros((self.n_spec, hsize, outsize))
    db1 = jp.zeros((self.n_spec, outsize))

    return minimal_params, (dw0, db0, dw1, db1)

  def repr_init(self, key):
    """Initialization for repr_f.
    
    Simply create a default requirement of nutrients for triggering the op.
    Note that this value is normalized by the material_nutrient_cap.
    """
    return (self.config.reproduce_cost + (self.config.dissipation_per_step * 4)
            + self.config.specialize_cost * 2) / self.config.material_nutrient_cap

  def initialize(self, key):
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    params = (
        self.dsm_init(k1),
        self.nsl_init(k2),
        self.denm_init(k3),
        self.excl_init(k4),
        self.repr_init(k5),
    )

    return jax.tree_util.tree_reduce(
        lambda a, b: jp.concatenate([a, b], 0),
        jax.tree_util.tree_map(lambda a: a.flatten(), params),
        initializer=jp.empty([0]),
    )

  def normalize_state(self, neigh_state):
    """Normalizes some inputs of neigh_state to be (sort of) scaled wrt the rest."""

    # scale down structural integrity.
    neigh_state = neigh_state.at[:, evm.STR_IDX].set(
        neigh_state[:, evm.STR_IDX] / self.config.struct_integrity_cap
    )
    # scale down energy
    neigh_state = neigh_state.at[:, evm.EN_ST : evm.EN_ST + 2].set(
        neigh_state[:, evm.EN_ST : evm.EN_ST + 2]
        / self.config.material_nutrient_cap
    )
    return neigh_state

  def get_avg_neigh_types(self, neigh_type):
    neigh_idxs = jp.array([0, 1, 2, 3, 5, 6, 7, 8])
    avg_neigh_types = jax.nn.one_hot(neigh_type[neigh_idxs], self.n_types).mean(
        0
    )
    return avg_neigh_types

  def par_f(self, key, perc, params):
    """Implementation of par_f.
    
    This outputs a ParallelInterface. This is made of:
    - denergy_neigh: how much energy to give to each agent neighbor. This is
      computed by denm_f.
    - dstate: how to change self internal states. If minimal_net==True, this is
      always a zero vector. Handled by dsm_f.
    - new_spec_logit: how to change specialization. Handled by nsl_f.
    """
    dsm_params, nsl_params, denm_params = self.split_par_params_f(params)
    dsm_params = self._dsm_format_params_fn(dsm_params)
    nsl_params = self._nsl_format_params_fn(nsl_params)
    denm_params = self._denm_format_params_fn(denm_params)

    neigh_type, neigh_state, neigh_id = perc
    norm_neigh_state = self.normalize_state(neigh_state)
    norm_self_state = norm_neigh_state[4]
    self_type = neigh_type[4]

    # get cell specialization
    spec_idx = self.config.etd.get_agent_specialization_idx(self_type)

    # compute dstate
    avg_neigh_types = self.get_avg_neigh_types(neigh_type)
    if self.minimal_net:
      dstate = jp.zeros([self.config.agent_state_size])
    else:
      dstate = self.dsm_f(
          dsm_params, avg_neigh_types, norm_self_state, spec_idx)

    # compute new_spec_logit
    new_spec_logit = self.nsl_f(
        nsl_params, avg_neigh_types, norm_self_state, spec_idx
    )

    # compute denergy_neigh
    self_en = neigh_state[4, evm.EN_ST : evm.EN_ST + 2]
    denergy_neigh = self.denm_f(
        denm_params, norm_neigh_state, spec_idx, neigh_type, neigh_id, self_en)


    return ParallelInterface(denergy_neigh, dstate, new_spec_logit)

  def excl_f(self, key, perc, params):
    """Implementation of excl_f.
    
    excl_f outputs a ExclusiveInterface. Its values are as follows:
    - sp_m: a mask for whether to spawn an agent or not. This depends on two
      factors: the agent needs to have sufficient energy, and it needs to roll
      a random number that depends on how many agent neighbors there are
      already.
    - sp_idx: the target neighbor cell for spawn. We create that by computing
      9 logits, one for each neighbor, to choose where to spawn. These logits
      are then used to sample from a gumbel-max distribution.
    - en_perc: what percentage of energy to give to the child. Depends on a 
      param.
    - child_state: how to initialize the child's internal states. It is set to
      zero if minimal_net==True, else it gets computed.
    - dstate_self: how to change self internal states. It is set to zero if
      minimal_net==True, else it gets computed.
    """
    decoded_p = self._excl_format_params_fn(params)
    if self.minimal_net:
      (b, min_en_for_spawn, max_neigh_agent_avg, spawn_prob_sensitivity,
       spawn_en_perc) = decoded_p
    else:
      ((b, min_en_for_spawn, max_neigh_agent_avg, spawn_prob_sensitivity,
        spawn_en_perc),
       (dw0, db0, dw1, db1)) = decoded_p

    neigh_type, neigh_state, _ = perc
    norm_neigh_state = self.normalize_state(neigh_state)
    self_type = neigh_type[4]
    norm_self_en = norm_neigh_state[4, evm.EN_ST : evm.EN_ST + 2]
    etd = self.config.etd

    # get cell specialization
    spec_idx = etd.get_agent_specialization_idx(self_type)

    # spawn must not happen if you don't have enough energy
    # (based on dissipation per step and spawn cost).
    # note this is done on a normalized basis.
    sp_m = (
        (norm_self_en - min_en_for_spawn >= 0.0).all(axis=-1).astype(jp.float32)
    )
    # Also, don't spawn if there are too many agent neighbors!
    # This is a probability depending on the average number of neighbors.
    avg_neigh_types = self.get_avg_neigh_types(neigh_type)
    avg_agents = avg_neigh_types[
        etd.types.AGENT_UNSPECIALIZED : etd.types.AGENT_UNSPECIALIZED + 4
    ].sum(-1)
    ku, key = jax.random.split(key)
    rand_prob_sp = (jr.uniform(ku) < jax.nn.sigmoid(
        (max_neigh_agent_avg - avg_agents) * spawn_prob_sensitivity)
                    ).astype(jp.float32)
    sp_m *= rand_prob_sp

    # where to spawn is a random occurrence.
    target_logits = b[spec_idx]

    if self.minimal_net:
      child_state = jp.zeros([self.config.agent_state_size])
      d_state_self = jp.zeros([self.config.agent_state_size])
    else:
      # add input-dependent residuals, both for target_logits and for self and 
      # child states.

      # - inputs:
      # - avg_neigh_types
      # - self state
      # - dx, dy states
      dx = (norm_neigh_state * jp.array(
          [-1.,0.,1., -1.,0.,1., -1.,0.,1.])[:,None]).sum(0)
      dy = (norm_neigh_state * jp.array(
          [1.,1.,1., 0.,0.,0., -1.,-1.,-1.])[:,None]).sum(0)
      inputs = jp.concatenate(
          [avg_neigh_types, norm_neigh_state[4], dx, dy], -1)
      i = spec_idx
      outputs = jax.nn.relu(inputs @ dw0[i] + db0[i])
      outputs = outputs @ dw1[i] + db1[i]
      d_target_logits, d_state_self, d_state_child = jp.split(
          outputs, (9, 9+self.config.agent_state_size))

      target_logits += d_target_logits

      self_internal_state = neigh_state[4, evm.A_INT_STATE_ST:]
      child_state = (self_internal_state + d_state_child).clip(
          -self.state_clip_val, self.state_clip_val)
      # as usual, we actually want to keep states within a certain range
      d_state_self = clip_residual(self_internal_state, d_state_self,
                                   self.state_clip_val)

    # target_logits go through a gumbel max to get an actual index.
    sp_idx = (jr.gumbel(key, target_logits.shape) + target_logits).argmax(
        axis=-1
    )

    # as an extra check, we can just zero out the result if the position of
    # spawn is already taken (there is an agent).
    # Note that this is largely useless, since the interface masks these out
    # anyway, and more than this.
    sp_m *= 1.0 - etd.is_agent_fn(neigh_type[sp_idx])

    # The perc energy to give is a parameter.
    # You could clip it to [0, 1] here, but the interface already takes care of
    # it.
    en_perc = spawn_en_perc * sp_m

    excl_int = ExclusiveInterface(
        sp_m[0], SpawnOpData(sp_idx, en_perc[0], child_state, d_state_self)
    )
    return excl_int

  def repr_f(self, key, perc, params):
    """Implementation of repr_f.
    
    A simple mask that entirely depends on how much energy we have.
    """
    norm_self_en = (perc.neigh_state[4, evm.EN_ST : evm.EN_ST + 2]
                    / self.config.material_nutrient_cap)

    # we could also check whether we are flowers, but it doesn't matter since
    # only flowers can reproduce and it gets masked afterwards.

    # The switch checks whether the output logit is >0.
    mask_logit = (norm_self_en > params).all().astype(jp.float32)
    return ReproduceInterface(mask_logit)
