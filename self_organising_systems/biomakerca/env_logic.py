"""Module containing the logic of the environments.

This module includes the logic for environmental actions such as gravity,
structural integrity, processing of energy, aging, as well as the definition and 
resolution of ExclusiveOps, ParallelOps and ReproduceOps.

This module does not contain methods that are cell-type specific. For instance,
the logic of AIR and EARTH cells is in cells_logic.py.

A note for the difference between ExclusiveOp vs ExclusiveInterface, and its 
equivalent for parallel and reproduce ops.
ExclusiveOps are internal ways of handling data, and agents must not manipulate
them. Instead, agents interface with ExclusiveInterfaces, which will transform
agent-friendly outputs into proper ExclusiveOps.

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
from collections import namedtuple
from functools import partial
from typing import Callable, Iterable

import flax
from jax import vmap
import jax.numpy as jp
import jax.random as jr
import jax.scipy

from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.environments import EnvConfig
from self_organising_systems.biomakerca.environments import EnvTypeDef
from self_organising_systems.biomakerca.environments import Environment
from self_organising_systems.biomakerca.utils import split_2d
from self_organising_systems.biomakerca.utils import vmap2

### Define some useful types for typing.
KeyType = jax.typing.ArrayLike
AgentProgramType = jax.typing.ArrayLike
EnvTypeType = jax.typing.ArrayLike
CellPositionType = jax.typing.ArrayLike


### ExclusiveOp
# These are the operations that act competing within one another.
# When a cell wants to act on a nearby cell, it makes a proposal 
# (a ExclusiveOp) stating where and what they want to happen to the target cell
# and to itself.
# Only one proposal can be accepted per target cell, so if more than one cell
# wants to act on a target, one gets chosen randomly and only that cell's
# ExclusiveOp gets committed. The others are discarded.
if "UpdateOp" not in globals():
  UpdateOp = namedtuple("UpdateOp", "upd_mask upd_type upd_state upd_id")
if "ExclusiveOp" not in globals():
  ExclusiveOp = namedtuple("ExclusiveOp", "target_op actor_op")

# Helper constants and functions for generating UpdateOps and ExclusiveOps
EMPTY_UPD_MASK = jp.zeros([9])
EMPTY_UPD_TYPE = jp.zeros([9], dtype=jp.uint32)

def make_empty_upd_state(config):
  return jp.zeros([9, config.env_state_size])

EMPTY_UPD_ID = jp.zeros([9], dtype=jp.uint32)

def make_empty_exclusive_op_cell(config):
  return ExclusiveOp(
      UpdateOp(EMPTY_UPD_MASK, EMPTY_UPD_TYPE,
               make_empty_upd_state(config), EMPTY_UPD_ID),
      UpdateOp(EMPTY_UPD_MASK, EMPTY_UPD_TYPE,
               make_empty_upd_state(config), EMPTY_UPD_ID))

### ExclusiveInterface
# Agents cannot output raw ExclusiveOps. Instead, they output 
# ExclusiveInterfaces that then get validated and converted into appropriate
# ExclusiveOps.
# Currently, the only exclusive action that agents can perform is Spawn.
# SpawnOpData contains the details on how to spawn new cells.
# The switch in ExclusiveInterface determines whether Spawn should be triggered
# or not.
EMPTY_EXCLUSIVE_INTERFACE_SWITCH = 0.
# Spawn arguments:
#  sp_idx: which neighbor position is the target. [0-8] excluding 4.
#    Possible todo: change it into logits.
#  en_perc: what percentage of energy to give to the new child.
#  child_state: the initial state of the child.
#  d_state_self: how to change the actor's state.
if "SpawnOpData" not in globals():
  SpawnOpData = namedtuple("SpawnOpData",
                           "sp_idx en_perc child_state d_state_self")
if "ExclusiveInterface" not in globals():
  ExclusiveInterface = namedtuple("ExclusiveInterface",
                                   "switch spawn_op_data")

# Helpers for making ExclusiveInterfaces.
def make_empty_agent_state_cell(config):
  return jp.zeros([config.agent_state_size])
def make_empty_spawn_op_data_cell(config):
  return SpawnOpData(0, 0., make_empty_agent_state_cell(config),
                     make_empty_agent_state_cell(config))
def make_empty_exclusive_interface_cell(config):
  return ExclusiveInterface(EMPTY_EXCLUSIVE_INTERFACE_SWITCH,
                             make_empty_spawn_op_data_cell(config))


### ParallelOp
# Agents can peform some operations in parallel: passing nutrients (energy) to
# neighboring agents, modidying their internal states, and changing
# specialization.
# Note that to change specialization, a certain amount of nutrients are required
# or else the change will not happen.
# Agents cannot output ParallelOps directly. See
# ParallelInterface for further information.
if "ParallelOp" not in globals():
  ParallelOp = namedtuple("ParallelOp",
                          "mask denergy_neigh dstate new_type")

# Helpers for making ParallelOps
EMPTY_DENERGY_NEIGH = jp.zeros([9, 2])

def make_empty_dstate(config):
  return jp.zeros([config.agent_state_size])

def make_empty_parallel_op_cell(config):
  return ParallelOp(
      jp.array(0.), EMPTY_DENERGY_NEIGH, make_empty_dstate(config), 
      jp.array(0, dtype=jp.uint32))

### ParallelInterface
# agents cannot output ParallelOps themselves. instead, they output
# ParallelInterfaces that get validated and then processed into 
# ParallelOps.
# Arguments:
#  denergy_neigh: how many nutrients to give to each neighbor (3x3) except for
#    position 4, which is ignored.
#  dstate: how to change self internal state.
#  new_spec_logit: logit of what should be the next specialization.
if "ParallelInterface" not in globals():
  ParallelInterface = namedtuple("ParallelInterface",
                                 "denergy_neigh dstate new_spec_logit")


# ReproduceOp.
# flowers can create ReproduceInterfaces that are converted to ReproduceOps.
# Nutrients from the flower get converted into stored_en and we record the 
# original position and agent id. A new seed is then tentatively created
# following the logic of the environment decided upon.
if "ReproduceOp" not in globals():
  ReproduceOp = namedtuple("ReproduceOp", "mask pos stored_en aid")

# Helpers for making ReproduceOps.
EMPTY_REPRODUCE_OP = ReproduceOp(
    jp.array(0.), jp.zeros([2], dtype=jp.int32), jp.zeros([2]),
    jp.array(0, dtype=jp.uint32))

### ReproduceInterface
# Interface for reproduction of agents.
# If a flower triggers reproduction, all of its energy is converted for the 
# new seed. However, reproduction has a cost and it may fail.
# Arguments:
#   mask_logit: whether or not to perform reproduction. True if > 0.
if "ReproduceInterface" not in globals():
  ReproduceInterface = namedtuple("ReproduceInterface", "mask_logit")


### Code for actions taken by the environment.


# PerceivedData.
# No matter the kind of cell, be it a material or an agent, they can only 
# perceive a limited amount of data. This is the 3x3 neighborhood of the 
# environment. The difference from Environment is that each cell has 9 values
# per grid. That is, neigh_type will be [w,h,9] as opposed to type_grid: [w,h].
# Also note that perceived data is intended to be passed using vmap2 so that
# each cell only perceives their neighbors, that is, their input is of size:
# neigh_type:[9], neigh_state:[9,env_state_size], neigh_id:[9].
# Still, PerceivedData can be manipulated to be either cell- or grid-specific.
if "PerceivedData" not in globals():
  PerceivedData = namedtuple("PerceivedData",
                             "neigh_type, neigh_state, neigh_id")


def perceive_neighbors(env: Environment, etd: EnvTypeDef) -> PerceivedData:
  """Return PerceivedData (gridwise, with leading axes of size [w,h]).
  
  Cells can only perceive their neighbors. Of the neighbors, they can perceive 
  all: type, state and agent_id.
  
  If one wants to explore behaviors where agent_ids are not used by agents, it 
  is the responsibility of the agent logic to not use such information.
  """
  # The convolution patches input has to be 4d and float.
  # Use out_of_bounds padding.
  pad_type_grid = jp.pad(env.type_grid, 1,
                         constant_values=etd.types.OUT_OF_BOUNDS)
  neigh_type = jax.lax.conv_general_dilated_patches(
      pad_type_grid[None, :, :, None].astype(jp.float32),
      (3, 3), (1, 1), "VALID", dimension_numbers=("NHWC", "OIHW", "NHWC")
      )[0].astype(jp.uint32)

  neigh_state = jax.lax.conv_general_dilated_patches(
      env.state_grid[None,:],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  # We want to have [w,h,9,c] so that the indexing is intuitive and consistent
  # for all neigh vectors.
  env_state_size = env.state_grid.shape[-1]
  neigh_state = neigh_state.reshape(
      neigh_state.shape[:2] + (env_state_size, 9)).transpose((0, 1, 3, 2))

  neigh_id = jax.lax.conv_general_dilated_patches(
      env.agent_id_grid[None, :, :, None].astype(jp.float32),
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC")
      )[0].astype(jp.uint32)

  return PerceivedData(neigh_type, neigh_state, neigh_id)


### ExclusiveOp related functions.
# These functions make ExclusiveOps and execute them into the environment.

def vectorize_cell_exclusive_f(
    cell_f: Callable[[KeyType, PerceivedData], ExclusiveOp], w, h):
  """Vectorizes a cell's exclusive_f to work from 0d to 2d."""
  return lambda key, perc: vmap2(cell_f)(split_2d(key, w, h), perc)


def vectorize_agent_cell_f(
    cell_f: Callable[[KeyType, PerceivedData, AgentProgramType],
                     ExclusiveOp|ParallelOp], w, h):
  """Vectorizes an agent cell_f to work from 0d to 2d.
  This works for both ExclusiveOp and ParallelOp.
  Note that the cell_f *requires* to have the proper argument name 'programs',
  so this is an informal interface.
  """
  return lambda key, perc, progs: vmap2(partial(cell_f, programs=progs))(
      split_2d(key, w, h), perc)


def make_material_exclusive_interface(
    cell_type, cell_f: Callable[[KeyType, PerceivedData, EnvConfig], ExclusiveOp],
    config: EnvConfig) -> Callable[[KeyType, PerceivedData], ExclusiveOp]:
  """Constructs an interface for material exclusive functions.
  This interface makes sure that a given cell type only executes its own 
  operations when the cell is of the correct type. Hence, one can write cell_fs
  without having to check that the type is correct.
  Also, the cell_f can depend on configs without us having to pass configs as
  input every time.
  """
  def f(key: KeyType, perc: PerceivedData):
    # extract self type from the perception.
    is_correct_type = cell_type == perc.neigh_type[4]
    excl_op = jax.lax.cond(
        is_correct_type,
        lambda key, perc: cell_f(key, perc, config),  # true
        lambda key, perc: make_empty_exclusive_op_cell(config),  # false
        key,
        perc,
    )
    return excl_op

  return f


def _convert_to_exclusive_op(
    excl_interface: ExclusiveInterface, perc: PerceivedData,
    env_config: EnvConfig
    ) -> ExclusiveOp:
  """Process ExclusiveInterface to convert it to ExclusiveOp.

  This also performs sanity checks and potentially rejects badly formed ops.
  """
  neigh_type, neigh_state, neigh_id = perc
  self_type = neigh_type[4]
  self_state = neigh_state[4]
  self_agent_state = self_state[evm.A_INT_STATE_ST:]
  self_en = self_state[evm.EN_ST : evm.EN_ST + 2]
  self_id = neigh_id[4]
  
  etd = env_config.etd

  switch, spawn_op_data = excl_interface
  sp_idx, en_perc, child_state, d_state_self = spawn_op_data

  sp_m = (switch > 0.5).astype(jp.float32)
  # Spawn sanity checks
  # must have enough energy for spawn.
  has_enough_energy = (self_en >= env_config.spawn_cost).all()
  # agents can only be spawned in some materials.
  is_valid_target = (neigh_type[sp_idx] == etd.agent_spawnable_mats).any()

  sp_m = sp_m * (has_enough_energy & is_valid_target).astype(jp.float32)
  
  # en_perc is within [0,1]
  en_perc = en_perc.clip(0, 1)

  # Spawn has a cost. we need to subtract this from the energy of self.
  spawn_en = (self_en - env_config.spawn_cost).clip(0.0)

  t_upd_mask = EMPTY_UPD_MASK.at[sp_idx].set(sp_m)
  t_upd_type = EMPTY_UPD_TYPE.at[sp_idx].set(
      sp_m.astype(jp.uint32) * etd.types.AGENT_UNSPECIALIZED
  )
  t_upd_state = (
      make_empty_upd_state(env_config)
      .at[sp_idx, evm.EN_ST : evm.EN_ST + 2]
      .set(sp_m * spawn_en * en_perc)
  )
  t_upd_state = t_upd_state.at[sp_idx, evm.A_INT_STATE_ST:].set(
      sp_m * child_state)
  # give the same age
  t_upd_state = t_upd_state.at[sp_idx, evm.AGE_IDX].set(
      sp_m * (self_state[evm.AGE_IDX])
  )
  t_upd_id = EMPTY_UPD_ID.at[sp_idx].set(sp_m.astype(jp.uint32) * self_id)

  # actor
  a_upd_mask = t_upd_mask
  # the type is unchanged regardless of what happens, but we mask this for 
  # consistency.
  a_upd_type = EMPTY_UPD_TYPE.at[sp_idx].set(sp_m.astype(jp.uint32) * self_type)
  a_upd_state = (
      make_empty_upd_state(env_config)
      .at[sp_idx, evm.EN_ST : evm.EN_ST + 2]
      .set(sp_m * spawn_en * (1.0 - en_perc))
  )
  # keep the same structural integrity
  a_upd_state = a_upd_state.at[sp_idx, evm.STR_IDX].set(
      sp_m * (self_state[evm.STR_IDX])
  )
  # update self state
  a_upd_state = a_upd_state.at[sp_idx, evm.A_INT_STATE_ST:].set(
      sp_m * (self_agent_state + d_state_self)
  )
  # keep the same age
  a_upd_state = a_upd_state.at[sp_idx, evm.AGE_IDX].set(
      sp_m * (self_state[evm.AGE_IDX])
  )
  a_upd_id = EMPTY_UPD_ID.at[sp_idx].set(sp_m.astype(jp.uint32) * self_id)

  # wrap this into a exclusive op.
  return ExclusiveOp(
      UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
      UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
  )


def make_agent_exclusive_interface(
    excl_f: Callable[[KeyType, PerceivedData, AgentProgramType],
                     ExclusiveInterface],
    config: EnvConfig) -> Callable[
        [KeyType, PerceivedData, AgentProgramType], ExclusiveOp]:
  """Construct an interface for agent exclusive functions.
  This interface makes sure that a cell's program is only executed for the 
  correct cell type and agent_id. 
  It also converts the output of excl_f (a ExclusiveInterface) into a 
  ExclusiveOp, by making sure that no laws of physics are broken.
  """
  def f(key: KeyType, perc: PerceivedData, programs: AgentProgramType
        ) -> ExclusiveOp:
    is_correct_type = config.etd.is_agent_fn(perc.neigh_type[4])

    curr_agent_id = perc.neigh_id[4]
    program = programs[curr_agent_id]
    excl_op = jax.lax.cond(
        is_correct_type,
        lambda key, perc: _convert_to_exclusive_op(
            excl_f(key, perc, program), perc, config
        ),  # true
        lambda key, perc: make_empty_exclusive_op_cell(config),  # false
        key,
        perc,
    )
    return excl_op

  return f


def tree_map_sum_ops(ops):
  return jax.tree_util.tree_map(
      lambda *t: jp.sum(jp.stack(t, dtype=t[0].dtype), 0), *ops)


def execute_and_aggregate_exclusive_ops(
    key: KeyType, env: Environment, programs: AgentProgramType,
    config: EnvConfig,
    excl_fs: Iterable[tuple[EnvTypeType,
                            Callable[[KeyType, PerceivedData], ExclusiveOp]]],
    agent_excl_f: Callable[[
        KeyType, PerceivedData, AgentProgramType], ExclusiveInterface]
    ) -> ExclusiveOp:
  """Execute all exclusive functions and aggregate them all into a single
  ExclusiveOp for each cell.
  
  This function constructs sanitized interfaces for the input excl_fs and 
  agent_excl_f, making sure that no laws of physics are broken.
  It also then executes these operation for each cell in the grid and then
  aggregates the resulting ExclusiveOp for each cell.
  Aggregation can be done because *only one function*, at most, will be allowed
  to output nonzero values for each cell.
  """
  etd = config.etd
  perc = perceive_neighbors(env, etd)
  w, h = env.type_grid.shape
  v_excl_fs = [vectorize_cell_exclusive_f(
      make_material_exclusive_interface(t, f, config), w, h) for (t, f)
               in excl_fs]
  v_agent_excl_f = vectorize_agent_cell_f(
      make_agent_exclusive_interface(agent_excl_f, config), w, h)

  k1, k2, key = jr.split(key, 3)
  w, h = env.type_grid.shape
  excl_ops = [f(k, perc) for k, f in 
              zip(jr.split(k1, len(v_excl_fs)), v_excl_fs)] + [
                  v_agent_excl_f(k2, perc, programs)
              ]
  excl_op = tree_map_sum_ops(excl_ops)
  return excl_op


def _cell_choose_random_action(key, excl_op_neighs):
  """Return a random index of valid exclusive ops from neighs targeting self.
  """
  t_op = excl_op_neighs.target_op
  t_upd_mask = t_op.upd_mask
  n_asking = t_upd_mask.sum()
  # if nobody is asking, it is ok, since the upd_mask would be set to zero.
  # so we just keep a uniform distribution.
  prob_mask = jax.lax.cond(n_asking >= 1.,
                           lambda: t_upd_mask / n_asking,
                           lambda: jp.full([9], 1./9.))
  target_chosen = jr.choice(key, jp.arange(0, 9, dtype=jp.int32), p=prob_mask)
  return target_chosen


def env_exclusive_decision(
    key: KeyType, env: Environment, excl_op: ExclusiveOp):
  """Choose up to one excl_op for each cell and execute them, updating the env.
  """
  w, h, chn = env.state_grid.shape

  def extract_patches_that_target_cell(x):
    """Extract all ExclusiveOps subarray of neighbors that target each cell.
    
    For each cell, a ExclusiveOp defines 9 different possible targets,
    one for each neighbor. In this function, we want to aggregate all
    ExclusiveOps that target a given cell, and return them, so that one of them
    can be chosen to be executed.
    """
    # input is either (w,h,9) or (w,h,9,c)
    ndim = x.ndim
    x_shape = x.shape
    old_dtype = x.dtype
    if old_dtype == jp.uint32:
      # convert to f32 and then back to uint.
      x = x.astype(jp.float32)
    if ndim == 4:
      x = x.reshape(x_shape[:-2]+(-1,))
    neigh_state = jax.lax.conv_general_dilated_patches(
        x[None,:], (3, 3), (1, 1), "SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
    # make it (w, h, k, 9)
    # where k is either 9 or 9*c
    neigh_state = neigh_state.reshape(
        neigh_state.shape[:-1] + (x.shape[-1], 9))
    # reshape accordingly for the case ndim is 4:
    if ndim == 4:
      neigh_state = neigh_state.reshape(
          neigh_state.shape[:-2] + x_shape[-2:] + (9,)).transpose(
              (0, 1, 2, 4, 3))
    # now it's either (w,h,9,9) or (w,h,9,9,c).
    # The leftmost '9' refers to the position of the cell's neighbors.
    # the rightmost '9' indicates the exclusive op slice of the neighbor, which
    # in turn targets 9 neighbors.
    # We want to get only the slice that targets this cell.
    # the pairing is inverted, since (-1,-1) of the actor reflects to a (1,1)
    # on the target.
    NEIGH_PAIRING = jp.stack([jp.arange(0, 9), jp.arange(8, -1, -1)], 0)
    neigh_state = vmap2(lambda x: x[NEIGH_PAIRING[0], NEIGH_PAIRING[1]])(
        neigh_state)

    if old_dtype == jp.uint32:
      neigh_state = neigh_state.astype(jp.uint32)
    return neigh_state

  excl_op_neighs = jax.tree_util.tree_map(
      extract_patches_that_target_cell, excl_op)


  # choose a random one from them.
  rnd_neigh_idx = vmap2(_cell_choose_random_action)(
      split_2d(key, w, h), excl_op_neighs)
  
  # Do so by zeroing out everything except the chosen update.
  action_mask = jax.nn.one_hot(rnd_neigh_idx, 9)
  target_op = excl_op_neighs.target_op
  # the target update can be just summed after mask.
  t_upd_mask = (target_op.upd_mask * action_mask).sum(-1)
  t_upd_mask_uint32 = t_upd_mask.astype(jp.uint32)
  t_upd_type = (target_op.upd_type * action_mask.astype(jp.uint32)).sum(-1)
  t_upd_state = (target_op.upd_state * action_mask[..., None]).sum(-2)
  t_upd_id = (target_op.upd_id * action_mask.astype(jp.uint32)).sum(-1)

  # The actor update needs to be first masked, then mapped back into its proper
  # position, and then summed.
  actor_op = excl_op_neighs.actor_op
  a_upd_mask_from_t = actor_op.upd_mask * action_mask
  a_upd_type_from_t = actor_op.upd_type * action_mask.astype(jp.uint32)
  a_upd_state_from_t = actor_op.upd_state * action_mask[..., None]
  a_upd_id_from_t = actor_op.upd_id * action_mask.astype(jp.uint32)
  # mapping back
  MAP_TO_ACTOR_SLICING = (
      (0, 0, 0), # inverse indexing was (-1, -1)
      (0, 1, 0), # inverse indexing was (-1, 0)
      (0, 2, 0), # inverse indexing was (-1, 1)
      (1, 0, 0), # inverse indexing was (0, -1)
      (1, 1, 0), # inverse indexing was (0, 0)
      (1, 2, 0), # inverse indexing was (0, 1)
      (2, 0, 0), # inverse indexing was (1, -1)
      (2, 1, 0), # inverse indexing was (1, 0)
      (2, 2, 0), # inverse indexing was (1, 1)
  )
  def map_to_actor(x):
    pad_x = jp.pad(x, ((1, 1), (1, 1), (0, 0)))
    return jp.stack([jax.lax.dynamic_slice(pad_x, s, (w, h, 9))[:,:,n] for n, s
                     in enumerate(MAP_TO_ACTOR_SLICING)], 2)

  def map_to_actor_e(x):
    pad_x = jp.pad(x, ((1, 1), (1, 1), (0, 0), (0, 0)))
    return jp.stack([jax.lax.dynamic_slice(pad_x, s+(0,), (w, h, 9, chn))[:,:,n]
                     for n, s in enumerate(MAP_TO_ACTOR_SLICING)], 2)

  a_upd_mask = map_to_actor(a_upd_mask_from_t).sum(-1)
  a_upd_mask_uint32 = a_upd_mask.astype(jp.uint32)
  a_upd_type = map_to_actor(a_upd_type_from_t).sum(-1)
  a_upd_state = map_to_actor_e(a_upd_state_from_t).sum(-2)
  a_upd_id = map_to_actor(a_upd_id_from_t).sum(-1)

  # Finally, in order, make an actor update and then a target update.
  new_type_grid = (env.type_grid * (1 - a_upd_mask_uint32)
                   + a_upd_type * a_upd_mask_uint32)
  new_type_grid = (new_type_grid * (1 - t_upd_mask_uint32)
                   + t_upd_type * t_upd_mask_uint32)
  new_state_grid = (env.state_grid * (1. - a_upd_mask[..., None])
                    + a_upd_state * a_upd_mask[..., None])
  new_state_grid = (new_state_grid * (1. - t_upd_mask[..., None])
                    + t_upd_state * t_upd_mask[..., None])
  new_agent_id_grid = (env.agent_id_grid * (1 - a_upd_mask_uint32)
                       + a_upd_id * a_upd_mask_uint32)
  new_agent_id_grid = (new_agent_id_grid * (1 - t_upd_mask_uint32)
                       + t_upd_id * t_upd_mask_uint32)

  return Environment(new_type_grid, new_state_grid, new_agent_id_grid)


def env_perform_exclusive_update(
    key: KeyType, env: Environment, programs: AgentProgramType,
    config: EnvConfig,
    excl_fs: Iterable[tuple[EnvTypeType,
                            Callable[[KeyType, PerceivedData], ExclusiveOp]]],
    agent_excl_f: Callable[[
        KeyType, PerceivedData, AgentProgramType], ExclusiveInterface]
    ) -> Environment:
  """Perform exclusive operations in the environment.
  
  This is the function that should be used for high level step_env design.
  
  Arguments:
    key: a jax random number generator.
    env: the input environment to be modified.
    programs: params of agents that govern their agent_excl_f. They should be
      one line for each agent_id allowed in the environment.
    config: EnvConfig describing the physics of the environment.
    excl_fs: list of pairs of (env_type, func) where env_type is the type of 
      material that triggers the exclusive func.
    agent_excl_f: The exclusive function that agents perform. It takes as 
      input a exclusive program and outputs a ExclusiveInterface.
  Returns:
    an updated environment.
  """
  k1, key = jr.split(key)
  excl_op = execute_and_aggregate_exclusive_ops(
      k1, env, programs, config, excl_fs, agent_excl_f)

  key, key1 = jr.split(key)
  env = env_exclusive_decision(key1, env, excl_op)
  return env


### PARALLEL OPERATIONS
# These functions make ParallelOps and execute them into the environment.


def _convert_to_parallel_op(
    par_interface: ParallelInterface, perc: PerceivedData,
    env_config: EnvConfig
    ) -> ParallelOp:
  """Process ParallelInterface to convert it to ParallelOp.
  
  This also performs sanity checks and potentially alters the op to not break 
  laws of physics.
  """
  neigh_type, neigh_state, _ = perc
  self_state = neigh_state[4]
  self_en = self_state[evm.EN_ST : evm.EN_ST + 2]
  etd = env_config.etd

  denergy_neigh, dstate, new_spec_logit = par_interface

  # Make sure that the energy passed is valid.
  # energy is only passed to agents.
  is_neigh_agent_fe = etd.is_agent_fn(neigh_type).astype(jp.float32)[:, None]
  denergy_neigh = denergy_neigh * is_neigh_agent_fe
  # Energy passed is only >= 0.
  # the energy passed to self is ignored.
  denergy_neigh = jp.maximum(denergy_neigh, 0.)

  # scale down to (maximum) the available energy of the cell.
  denergy_asked = denergy_neigh[jp.array([0, 1, 2, 3, 5, 6, 7, 8])].sum(0)
  denergy_self = jp.minimum(denergy_asked, self_en)

  divider = (denergy_asked / denergy_self.clip(1e-6)).clip(min=1.0)

  denergy_neigh = (denergy_neigh / divider).at[4].set(-denergy_self)

  new_type = etd.get_agent_type_from_spec_idx(jp.argmax(new_spec_logit))

  # Specializing has a cost. If you can't afford it, do not change type.
  self_type = neigh_type[4]
  new_en = self_en + denergy_neigh[4]
  is_different_type = (new_type != self_type).astype(jp.uint32)
  cant_afford = (
      ((new_en - env_config.specialize_cost) < 0.0).any().astype(jp.uint32)
  )
  spec_m = is_different_type * (1 - cant_afford)
  new_type = new_type * spec_m + self_type * (1 - spec_m)
  # if we are changing spec, decrease self energy accordingly.
  denergy_neigh = denergy_neigh.at[4].set(
      denergy_neigh[4] - env_config.specialize_cost * spec_m
  )

  return ParallelOp(1.0, denergy_neigh, dstate, new_type)


def make_agent_parallel_interface(
    par_f: Callable[[KeyType, PerceivedData, AgentProgramType],
                     ParallelInterface],
    config: EnvConfig) -> Callable[
        [KeyType, PerceivedData, AgentProgramType], ParallelOp]:
  """Construct an interface for agent parallel functions.
  This interface makes sure that a cell's program is only executed for the 
  correct cell type and agent_id.
  It also converts the output of par_f (a ParallelInterface) into a 
  ParallelOp, by making sure that no laws of physics are broken.
  """
  def f(key, perc, programs):
    is_correct_type = config.etd.is_agent_fn(perc.neigh_type[4])

    curr_agent_id = perc.neigh_id[4]
    program = programs[curr_agent_id]
    par_op = jax.lax.cond(
        is_correct_type,
        lambda key, perc: _convert_to_parallel_op(
            par_f(key, perc, program), perc, config
        ),  # true
        lambda key, perc: make_empty_parallel_op_cell(config),  # false
        key,
        perc,
    )
    return par_op

  return f


def env_perform_parallel_update(
    key: KeyType, env: Environment, programs: AgentProgramType,
    config: EnvConfig,
    par_f: Callable[[
        KeyType, PerceivedData, AgentProgramType], ParallelInterface]
    ) -> Environment:
  """Perform parallel operations in the environment.
  
  This is the function that should be used for high level step_env design.
  
  Arguments:
    key: a jax random number generator.
    env: the input environment to be modified.
    programs: params of agents that govern their par_f. They should be
      one line for each agent_id allowed in the environment.
    config: EnvConfig describing the physics of the environment.
    par_f: The parallel function that agents perform. It takes as 
      input a parallel program and outputs a ParallelInterface.
  Returns:
    an updated environment.
  """
  # First compute the ParallelOp for each cell.
  w, h = env.type_grid.shape
  v_par_f = vectorize_agent_cell_f(make_agent_parallel_interface(
      par_f, config), w, h)

  etd = config.etd
  perc = perceive_neighbors(env, etd)

  k1, key = jr.split(key)
  par_op = v_par_f(k1, perc, programs)

  # Then process them.
  mask, denergy_neigh, dstate, new_type = par_op
  w, h = env.type_grid.shape

  MAP_TO_NEIGH_SLICING = (
      (2, 2, 0), # indexing (-1, -1)
      (2, 1, 0), # indexing (-1, 0)
      (2, 0, 0), # indexing (-1, 1)
      (1, 2, 0), # indexing (0, -1)
      (1, 1, 0), # indexing (0, 0)
      (1, 0, 0), # indexing (0, 1)
      (0, 2, 0), # indexing (1, -1)
      (0, 1, 0), # indexing (1, 0)
      (0, 0, 0), # indexing (1, 1)
  )
  def map_to_neigh(x):
    pad_x = jp.pad(x, ((1, 1), (1, 1), (0, 0), (0, 0)))
    return jp.stack([jax.lax.dynamic_slice(pad_x, s+(0,), (w, h, 9, 2))[:, :, n]
                     for n, s in enumerate(MAP_TO_NEIGH_SLICING)], 2)

  denergy = map_to_neigh(denergy_neigh).sum(2)
  new_en = env.state_grid[:, :, evm.EN_ST : evm.EN_ST+2] + denergy
  # energy state cap is different based on whether the cell is an agent or not.
  is_agent_e = etd.is_agent_fn(env.type_grid).astype(jp.float32)[..., None]
  new_en = (new_en.clip(0., config.material_nutrient_cap) * (1. - is_agent_e) +
            new_en.clip(0., config.nutrient_cap) * is_agent_e)

  # note that this is not the totality of all states. For instance, structural 
  # integrity exists.
  new_int_state = env.state_grid[:, :, evm.A_INT_STATE_ST:] + dstate
  new_en_state = jp.concatenate([new_en, new_int_state], -1)

  mask_i = mask.astype(jp.uint32)
  new_type_grid = env.type_grid * (1 - mask_i) + new_type * mask_i
  new_state_grid = env.state_grid.at[:, :, evm.EN_ST:].set(new_en_state)

  env = Environment(new_type_grid, new_state_grid, env.agent_id_grid)
  return env


### REPRODUCTION OPERATIONS
# Functions for making and processing ReproduceOps.


def vectorize_reproduce_f(repr_f, w, h):
  return  lambda key, perc, pos, progs: vmap2(partial(repr_f, programs=progs))(
      split_2d(key, w, h), perc, pos)


def _convert_to_reproduce_op(
    repr_interface: ReproduceInterface, perc: PerceivedData, pos,
    env_config: EnvConfig
    ) -> ReproduceOp:
  """Code to process the operation validate and convert repr_interface to ReproduceOp.
  
  This also performs sanity checkes and potentially rejects the op to not break
  laws of physics.
  """
  neigh_type, neigh_state, neigh_id = perc
  self_type = neigh_type[4]
  self_state = neigh_state[4]
  self_en = self_state[evm.EN_ST : evm.EN_ST + 2]
  self_id = neigh_id[4]

  mask_logit = repr_interface.mask_logit
  # must want to reproduce
  want_to_repr_m = (mask_logit > 0.0).astype(jp.float32)

  # must be a flower
  is_flower_m = (self_type == env_config.etd.types.AGENT_FLOWER).astype(
      jp.float32)

  # must have enough energy.
  has_enough_en_m = (
      (self_en >= env_config.reproduce_cost).all().astype(jp.float32)
  )

  mask = want_to_repr_m * is_flower_m * has_enough_en_m

  stored_en = (self_en - env_config.reproduce_cost) * mask
  aid = (self_id * mask).astype(jp.uint32)
  return ReproduceOp(mask, pos, stored_en, aid)


def make_agent_reproduce_interface(
    repr_f: Callable[[KeyType, PerceivedData, AgentProgramType],
                     ReproduceInterface],
    config: EnvConfig) -> Callable[
        [KeyType, PerceivedData, CellPositionType, AgentProgramType],
        ReproduceOp]:
  """Construct an interface for agent reproduce functions.
  This interface makes sure that a cell's program is only executed for the 
  correct cell type (flowers) and agent_id.
  It also converts the output of repr_f (a ReproduceInterface) into a 
  ReproduceOp, by making sure that no laws of physics are broken.
  
  Note that the resulting function requires an extra input: the cell position,
  needed to know where to place resulting seeds. This information is not
  available to the agent, and only the ReproduceOp knows about it.
  """
  # Note that we need to pass the extra information of the position of the cell.
  # this is used in the conversion to ReproduceOp, not by the cell.
  def f(key, perc, pos, programs):
    # it has to be a flower!
    is_correct_type = config.etd.types.AGENT_FLOWER == perc.neigh_type[4]

    curr_agent_id = perc.neigh_id[4]
    program = programs[curr_agent_id]
    repr_op = jax.lax.cond(
        is_correct_type,
        lambda key, perc: _convert_to_reproduce_op(
            repr_f(key, perc, program), perc, pos, config
        ),  # true
        lambda key, perc: EMPTY_REPRODUCE_OP,  # false
        key,
        perc,
    )
    return repr_op

  return f


def find_fertile_soil(type_grid, etd):
  """Return, for each column, what is the first fertile row index and whether one exists.
  
  We treat a cell as 'fertile' if they are earth and above them there is air.
  We want to return the first occurrence of such, from top down.
  If no such thing exists in a column, we set the respective column_m[col] to 0.
  
  Returns:
    best_idx_per_column: the first row with a fertile cell, per column.
    column_m: whether each column has at least one fertile cell.
  """
  # create a mask that checks whether 'you are earth and above is air'.
  # the highest has index 0 (it is inverted).
  # the index starts from 1 (not 0) since the first row can never be fine.
  mask = (type_grid[1:] == etd.types.EARTH) & (type_grid[:-1] == etd.types.AIR)
  # only get the highest value, if it exists.
  # note that these indexes now would return the position of the 'high-end' of 
  # the seed. That is, the idx is where the top cell (leaf) would be, and idx+1 
  # would be where the bottom (root) cell would be.
  best_idx_per_column = (mask * jp.arange(mask.shape[0], 0, -1)[:, None]
                         ).argmax(axis=0)
  # We need to make sure that it is a valid position, so:
  column_m = mask[best_idx_per_column, jp.arange(mask.shape[1])]
  return best_idx_per_column, column_m


def _select_random_position_for_seed_within_range(
    key, center, min_dist, max_dist, column_m):
  col_idxs = jp.arange(0, column_m.shape[0])
  valid_range = (
      ((col_idxs >= center - max_dist) & (col_idxs <= center - min_dist))
      | ((col_idxs >= center + min_dist) & (col_idxs <= center + max_dist)))
  mask = (valid_range & column_m).astype(jp.float32)
  n_available = mask.sum()
  chosen_idx = jr.choice(key, col_idxs, p=mask/n_available.clip(1e-6))
  # return a flag to tell whether there was no valid option.
  return chosen_idx, n_available > 0


def env_perform_one_reproduce_op(
    key: KeyType, env: Environment, repr_op: ReproduceOp, config: EnvConfig):
  """Perform one single ReproduceOp.
  
  For a ReproduceOp to be successful, fertile soil in the neighborhood must be
  found.
  If it is, a new seed (two unspecialized cells) are placed in the environment.
  Their age is reset to zero, and they may have a different agent_id than their
  parent, if mutation was set to true.
  """
  mask, pos, stored_en, aid = repr_op
  etd = config.etd

  def true_fn(env):
    best_idx_per_column, column_m = find_fertile_soil(env.type_grid, etd)
    t_column, column_valid = _select_random_position_for_seed_within_range(
        key, pos[1], config.reproduce_min_dist, config.reproduce_max_dist, 
        column_m)

    def true_fn2(env):
      t_row = best_idx_per_column[t_column]
      type_grid = jax.lax.dynamic_update_slice(
          env.type_grid,
          jp.full((2, 1), etd.types.AGENT_UNSPECIALIZED, dtype=jp.uint32),
          (t_row, t_column))
      new_states = jp.zeros([2, 1, config.env_state_size])
      # this by default resets age.
      new_states = jax.lax.dynamic_update_slice(
          new_states, jp.repeat((stored_en/2)[None, None,:], 2, axis=0),
          (t_row, t_column, evm.EN_ST))
      state_grid = jax.lax.dynamic_update_slice(
          env.state_grid, new_states, (t_row, t_column, 0))
      agent_id_grid = jax.lax.dynamic_update_slice(
          env.agent_id_grid, jp.repeat(aid, 2)[:, None], (t_row, t_column))
      return Environment(type_grid, state_grid, agent_id_grid)

    return jax.lax.cond(column_valid, true_fn2, lambda env: env, env)

  return jax.lax.cond(mask, true_fn, lambda env: env, env)


def env_reproduce_operations(key, env, b_repr_op, config):
  """Perform a batch of ReproduceOps.
  
  These are performed sequentially. Note that some ops may be masked and
  therefore be noops.
  """
  def body_f(carry, repr_op):
    env, key = carry
    key, ku = jr.split(key)
    env = env_perform_one_reproduce_op(ku, env, repr_op, config)
    return (env, key), 0

  (env, key), _ = jax.lax.scan(body_f, (env, key), b_repr_op)
  return env


def _select_subset_of_reproduce_ops(key, b_repr_op, neigh_type, config):
  # Only a small subset of possible ReproduceOps are selected at each step.
  # This is config dependent (config.n_reproduce_per_step).
  b_pos = b_repr_op.pos
  mask_flat = b_repr_op.mask.flatten()
  b_pos_flat = b_pos.reshape((-1, 2))

  p_logits = mask_flat
  # Moreover, flowers can only reproduce if in contact with air. The more air, 
  # the more likely they are to be selected.
  # NOTE: a part of this logic can be moved into _convert_to_reproduce_op, but 
  # since it would be redundant, I am putting it only here.
  n_air_neigh = (neigh_type == config.etd.types.AIR).astype(jp.float32
                                                            ).sum(-1).flatten()
  p_logits *= n_air_neigh

  k1, key = jr.split(key)
  selected_pos = jr.choice(
      k1, b_pos_flat, p=p_logits/p_logits.sum().clip(1),
      shape=(config.n_reproduce_per_step,), replace=False)

  return selected_pos


def env_perform_reproduce_update(
    key: KeyType, env: Environment, repr_programs: AgentProgramType,
    config: EnvConfig,
    repr_f: Callable[[KeyType, PerceivedData, AgentProgramType],
                     ReproduceInterface],
    mutate_programs=False,
    programs: (AgentProgramType | None) = None,
    mutate_f: (Callable[[KeyType, AgentProgramType], AgentProgramType] | None
               ) = None):
  """Perform reproduce operations in the environment.
  
  This is the function that should be used for high level step_env design.
  
  Arguments:
    key: a jax random number generator.
    env: the input environment to be modified.
    repr_programs: params of agents that govern their repr_f. They should be
      one line for each agent_id allowed in the environment.
    config: EnvConfig describing the physics of the environment.
    repr_f: The reproduce function that agents perform. It takes as 
      input a reproduce program and outputs a ReproduceInterface.
    mutate_programs: Whether reproduction also mutates programs. If True, this
      function requires to set 'programs' and 'mutate_f', and returns also the 
      updated programs.
    programs: The totality of the parameters for each agent id. This is not just
      reproduce programs, but all of them, including mutation parameters if 
      needed. There should be one line for each agent_id. This arg is used only 
      if mutate_programs is set to True.
    mutate_f: The mutation function to mutate 'programs'. This is intended to be
      the method 'mutate' of a Mutator class.
  Returns:
    an updated environment. if mutate_programs is True, it also returns 
    the updated programs.
  """
  etd = config.etd
  perc = perceive_neighbors(env, etd)
  k1, key = jr.split(key)
  w, h = env.type_grid.shape

  v_repr_f = vectorize_reproduce_f(
      make_agent_reproduce_interface(repr_f, config), w, h)

  b_pos = jp.stack(jp.meshgrid(jp.arange(w), jp.arange(h), indexing="ij"), -1)
  b_repr_op = v_repr_f(k1, perc, b_pos, repr_programs)

  # Only a small subset of possible ReproduceOps are selected at each step.
  # This is config dependent (config.n_reproduce_per_step).
  selected_pos = _select_subset_of_reproduce_ops(
      k1, b_repr_op, perc.neigh_type, config)
  spx, spy = selected_pos[:, 0], selected_pos[:, 1]

  selected_mask = b_repr_op.mask[spx, spy]
  selected_stored_en = b_repr_op.stored_en[spx, spy]
  selected_aid = b_repr_op.aid[spx, spy]

  if mutate_programs:
    # Logic:
    # look into the pool of programs and see if some of them are not used (
    # there are no agents alive with such program).
    # if that is the case, create a new program with mutate_f, then modify the 
    # corresponding 'selected_aid'.
    # If there is no space, set the mask to zero instead.

    # get n agents per id.
    is_agent_flat = etd.is_agent_fn(env.type_grid).flatten().astype(jp.float32)
    env_aid_flat = env.agent_id_grid.flatten()

    n_agents_in_env = jax.ops.segment_sum(
        is_agent_flat, env_aid_flat, num_segments=programs.shape[0])

    sorted_na_idx = jp.argsort(n_agents_in_env).astype(jp.uint32)
    # we only care about the first few indexes
    sorted_na_chosen_idx = sorted_na_idx[:config.n_reproduce_per_step]
    sorted_na_chosen_mask = (n_agents_in_env[sorted_na_chosen_idx] == 0
                             ).astype(jp.float32)

    # assume that the number of selected reproductions is LESS than the total 
    # number of programs.
    to_mutate_programs = programs[selected_aid]
    mutated_programs = vmap(mutate_f)(
        jr.split(key, config.n_reproduce_per_step), to_mutate_programs)

    mutation_mask = (selected_mask * sorted_na_chosen_mask)
    mutation_mask_e = mutation_mask[:, None]
    n_mutation_mask_e = 1. - mutation_mask_e

    # substitute the programs
    programs = programs.at[sorted_na_chosen_idx].set(
        mutation_mask_e * mutated_programs
        + n_mutation_mask_e * programs[sorted_na_chosen_idx])
    # update the aid and the mask
    selected_mask = mutation_mask
    selected_aid = sorted_na_chosen_idx

  # these positions (if mask says yes) are then selected to reproduce.
  # A seed is spawned if possible.
  env = env_reproduce_operations(
      key, env,
      ReproduceOp(selected_mask, selected_pos, selected_stored_en,
                  selected_aid),
      config)
  # The flower is destroyed, regardless of whether the operation succeeds.
  n_selected_mask = 1 - selected_mask
  n_selected_mask_uint = n_selected_mask.astype(jp.uint32)
  env = Environment(
      env.type_grid.at[spx, spy].set(
          n_selected_mask_uint * env.type_grid[spx, spy]), # 0 is VOID
      env.state_grid.at[spx, spy].set(
          n_selected_mask[..., None] * env.state_grid[spx, spy]
          ), # set everything to zero
      env.agent_id_grid.at[spx, spy].set(
          n_selected_mask_uint * env.agent_id_grid[spx, spy]) # default id 0.
  )
  if mutate_programs:
    return env, programs
  return env


def intercept_reproduce_ops(
    key: KeyType, env: Environment,
    repr_programs: AgentProgramType,
    config: EnvConfig,
    repr_f: Callable[[KeyType, PerceivedData, AgentProgramType],
                     ReproduceInterface],
    min_repr_energy_requirement):
  """Intercept reproduce ops to still destroy flowers but cause no reproduction.
  
  Instead, return a counter of 'successful' reproductions happening.
  
  NOTE: This method is largely code repetition (with 
  env_perform_reproduce_update). I chose to do that to avoid making the latter
  too complex. But I might refactor it eventually.
  
  Arguments:
    key: a jax random number generator.
    env: the input environment to be modified.
    repr_programs: params of agents that govern their repr_f. They should be
      one line for each agent_id allowed in the environment.
    config: EnvConfig describing the physics of the environment.
    repr_f: The reproduce function that agents perform. It takes as 
      input a reproduce program and outputs a ReproduceInterface.
    min_repr_energy_requirement: array of two values, one for each nutrient 
      type. This is a user-defined value to determine whether the user believes
      that the seed had enough nutrients to be able to grow. If the seed has
      less energy, we consider that a failed reproduction.
  Returns:
    an updated environment, and a counter determining the number of successful
    reproductions.
  """
  etd = config.etd
  perc = perceive_neighbors(env, etd)
  k1, key = jr.split(key)
  w, h = env.type_grid.shape
  v_repr_f = vectorize_reproduce_f(
      make_agent_reproduce_interface(repr_f, config), w, h)

  b_pos = jp.stack(jp.meshgrid(jp.arange(w), jp.arange(h), indexing="ij"), -1)
  b_repr_op = v_repr_f(k1, perc, b_pos, repr_programs)

  # Only a small subset of possible ReproduceOps are selected at each step.
  # This is config dependent (config.n_reproduce_per_step).
  selected_pos = _select_subset_of_reproduce_ops(
      k1, b_repr_op, perc.neigh_type, config)
  spx, spy = selected_pos[:, 0], selected_pos[:, 1]

  selected_mask = b_repr_op.mask[spx, spy]
  selected_stored_en = b_repr_op.stored_en[spx, spy]

  # stored energy has already the reproduce cost removed from there.
  has_enough_stored_energy = (selected_stored_en >= min_repr_energy_requirement
                              ).all(-1).astype(jp.float32)
  n_successful_repr = (selected_mask*has_enough_stored_energy).sum()

  # The flower is destroyed, regardless of whether the operation succeeds.
  n_selected_mask = 1 - selected_mask
  n_selected_mask_uint = n_selected_mask.astype(jp.uint32)
  env = Environment(
      env.type_grid.at[spx, spy].set(
          n_selected_mask_uint * env.type_grid[spx, spy]), # 0 is VOID
      env.state_grid.at[spx, spy].set(
          n_selected_mask[..., None] * env.state_grid[spx, spy]
          ), # set everything to zero
      env.agent_id_grid.at[spx, spy].set(
          n_selected_mask_uint * env.agent_id_grid[spx, spy]) # default id 0.
  )

  return env, n_successful_repr


### Gravity logic.


def _line_gravity(env, x, h, etd):
  type_grid, state_grid, agent_id_grid = env
  env_state_size = state_grid.shape[-1]
  # self needs to be affected by gravity:
  is_gravity_mat = vmap(lambda ctype: (ctype == etd.gravity_mats).any())(
      type_grid[x])
  # down needs to be intangible:
  is_down_intangible_mat = vmap(
      lambda ctype: (ctype == etd.intangible_mats).any())(type_grid[x+1])
  # structural integrity needs to be 0.
  # or it must be not structural
  is_crumbling = jp.logical_or(
      state_grid[x, :, 0] <= 0.,
      vmap(lambda ctype: (ctype != etd.structural_mats).all())(type_grid[x]))
  swap_mask = (is_gravity_mat & is_down_intangible_mat & is_crumbling
               ).astype(jp.float32)
  # [h] -> [1,h]
  swap_mask_e = swap_mask[None,:]
  swap_mask_uint_e = swap_mask_e.astype(jp.uint32)

  idx_swap_x = jp.repeat(jp.array([x, x+1]), h)
  idx_swap_y = jp.concatenate([jp.arange(0, h, dtype=jp.int32),
                               jp.arange(0, h, dtype=jp.int32)], 0)
  type_slice = jax.lax.dynamic_slice(type_grid, (x, 0), (2, h))
  type_upd = (type_slice * (1 - swap_mask_uint_e) +
              type_slice[::-1] * swap_mask_uint_e).reshape(-1)
  new_type_grid = type_grid.at[idx_swap_x, idx_swap_y].set(type_upd)
  swap_mask_ee = swap_mask_e[..., None]
  state_slice = jax.lax.dynamic_slice(
      state_grid, (x, 0, 0), (2, h, env_state_size))
  state_upd = (state_slice * (1. - swap_mask_ee) +
               state_slice[::-1] * swap_mask_ee).reshape([-1, env_state_size])
  new_state_grid = state_grid.at[idx_swap_x, idx_swap_y].set(state_upd)
  # agent ids
  id_slice = jax.lax.dynamic_slice(agent_id_grid, (x, 0), (2, h))
  id_upd = (id_slice * (1 - swap_mask_uint_e) +
            id_slice[::-1] * swap_mask_uint_e).reshape(-1)
  new_agent_id_grid = agent_id_grid.at[idx_swap_x, idx_swap_y].set(id_upd)

  return Environment(new_type_grid, new_state_grid, new_agent_id_grid), 0


def env_process_gravity(env: Environment, etd: EnvTypeDef) -> Environment:
  """Process gravity in the input env.
  
  Only materials subject to gravity (env.GRAVITY_MATS) can fall.
  They fall if there is nothing below them and either they are not structural
  (like EARTH), or they are structural and their structural integrity is 0.
  
  Create a new env by applying gravity on every line, from bottom to top.
  Nit: right now, you can't fall off, so we start from the second to bottom.
  """
  w, h = env.type_grid.shape
  env, _ = jax.lax.scan(
      partial(_line_gravity, h=h, etd=etd),
      env,
      jp.arange(w-2, -1, -1))
  return env


### Structural integrity logic.


def process_structural_integrity(env: Environment, config: EnvConfig):
  """Process one step of structural integrity.
  
  Immovable materials generate structural integrity. Every structural cell
  extracts the maximum structural integrity among its neighbors and updates
  itself to it minus a cell-type-specific decay.
  
  This step does NOT perform gravity! Call 'env_process_gravity' for that.
  
  It is desirable to run more than one step of stuctural integrity per
  iteration, since we want disconnected groups to lose structural integrity
  quickly. Therefore, it is recommended to use 
  'process_structural_integrity_n_times' instead.
  """
  type_grid, state_grid, agent_id_grid = env
  etd = config.etd
  is_immovable = type_grid == etd.types.IMMOVABLE
  propagates_structure = (type_grid[..., None] == etd.propagate_structure_mats
                          ).any(-1)

  # if cell is immovable, set the structural integrity to the cap.
  # otherwise, if it is structural, propagate the integrity of the maximum
  # neighbor.
  max_neigh_struct_int = flax.linen.max_pool(
      state_grid[:, :, evm.STR_IDX: evm.STR_IDX+1], window_shape=(3, 3),
      padding="SAME")[..., 0]
  
  # Every material has a certain structural decay. Subtract that.
  propagated_int = (max_neigh_struct_int -
                    etd.structure_decay_mats[type_grid]).clip(0)

  # this way it defaults to 0 for non structural materials.
  struct_upd = (is_immovable.astype(jp.float32) * config.struct_integrity_cap +
                (propagates_structure & (jp.logical_not(is_immovable))
                 ).astype(jp.float32) * (propagated_int))
  return Environment(
      type_grid, state_grid.at[:, :, evm.STR_IDX].set(struct_upd),
      agent_id_grid)


def process_structural_integrity_n_times(
    env: Environment, config: EnvConfig, n):
  """Process multiple steps of structural integrity.
  
  Immovable materials generate structural integrity. Every structural cell
  extracts the maximum structural integrity among its neighbors and updates
  itself to it minus a cell-type-specific decay.
  
  This step does NOT perform gravity! Call 'env_process_gravity' for that.
  
  It is desirable to run more than one step of stuctural integrity per
  iteration, since we want disconnected groups to lose structural integrity
  quickly.
  """
  env, _ = jax.lax.scan(
      (lambda env, x: (process_structural_integrity(env, config), 0.)), env,
      None, n)
  return env

# agents interact with the environment.
# They can be energy based, but I will make that an optional configuration.


# Processing of energy (nutrients)
# I use energy and nutrients synonimously in this code base. It's because 
# sometimes it is clearer to talk about them in one way, sometimes in another.
# I might eventually refactor that to remove 'energy' everywhere.


def process_energy(env: Environment, config: EnvConfig) -> Environment:
  """Process one step of energy transfer and dissipation.
  
  This function works in different steps:
  1) Nutrients diffuse. Immovable and Sun materials generate nutrients for
    earth and air respectively. Alongisde Earth and Air cells, respectively, 
    diffuse these nutrients.
  2) Nutrient extraction. Roots and Leaves can extract nutrients from Earth and
    Air neighbors.
  3) Energy dissipation. Agents, at every step, dissipate energy (nutrients).
    This depends on both a constant value and on their age.
  4) Kill energy-less agents. If an agent doesn't have either of the required
    nutrients, it gets killed and converted to either Earth, Air, or Void, 
    depending on what kind of nutrients are left.
  """
  # How it works: The top gets padded with 'air' that contains maximum air nutrient.
  # IMMOVABLE (for now) is treated as earth as it had maximum earth nutrient.
  # then, we perform diffusion of the nutrients (air to air, earth to earth).
  # Finally, energy is absorbed by the roots/leaves (with a cap).
  etd = config.etd

  ### Nutrient diffusion.

  # earth nutrients:
  is_earth_grid_f = (env.type_grid == etd.types.EARTH).astype(jp.float32)
  is_immovable_grid_f = (env.type_grid == etd.types.IMMOVABLE).astype(
      jp.float32)
  earth_nutrient = (
      env.state_grid[:,:, evm.EN_ST+evm.EARTH_NUTRIENT_RPOS] * is_earth_grid_f +
      config.material_nutrient_cap[evm.EARTH_NUTRIENT_RPOS
                                   ] * is_immovable_grid_f)
  # diffuse this nutrient to all earth (+ immovable, which remains unchanged).
  EARTH_DIFFUSION_RATE = 0.1
  neigh_earth_nutrient = jax.lax.conv_general_dilated_patches(
      earth_nutrient[None,:,:, None],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  neigh_earth_mask = jax.lax.conv_general_dilated_patches(
      (is_earth_grid_f+is_immovable_grid_f)[None,:,:, None],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  d_earth_n = ((neigh_earth_nutrient - earth_nutrient[:,:, None]
                ) * neigh_earth_mask * EARTH_DIFFUSION_RATE).sum(-1)

  # discard immovable nutrients.
  new_earth_nutrient = (earth_nutrient + d_earth_n) * is_earth_grid_f

  # air nutrients:
  is_air_grid_f = (env.type_grid == etd.types.AIR).astype(jp.float32)
  is_sun_grid_f = (env.type_grid == etd.types.SUN).astype(jp.float32)
  air_nutrient = (
      env.state_grid[:,:, evm.EN_ST+evm.AIR_NUTRIENT_RPOS] * is_air_grid_f +
      config.material_nutrient_cap[evm.AIR_NUTRIENT_RPOS] * is_sun_grid_f)
  # diffuse this nutrient to all air (+ sun, which remains unchanged.)
  AIR_DIFFUSION_RATE = 0.1

  # compute neighbors
  neigh_air_nutrient = jax.lax.conv_general_dilated_patches(
      air_nutrient[None,:,:, None],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  neigh_air_mask = jax.lax.conv_general_dilated_patches(
      (is_air_grid_f+is_sun_grid_f)[None,:,:, None],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  d_air_n = ((neigh_air_nutrient - air_nutrient[:,:, None]
              ) * neigh_air_mask * AIR_DIFFUSION_RATE).sum(-1)

  # discard sun nutrients.
  new_air_nutrient = (air_nutrient + d_air_n) * is_air_grid_f

  ### Nutrient extraction.
  
  is_root_grid = (env.type_grid == etd.types.AGENT_ROOT)
  is_leaf_grid = (env.type_grid == etd.types.AGENT_LEAF)

  # this says how much each agent is asking to each neighbor.
  asking_nutrients = jp.stack(
      [is_root_grid, is_leaf_grid], -1) * config.absorbtion_amounts
  # this is how much is available.
  available_nutrients = jp.stack([new_earth_nutrient, new_air_nutrient], -1)
  # compute the total amount of asking nutrients per cell and nutrient kind.
  neigh_asking_nutrients = jax.lax.conv_general_dilated_patches(
      asking_nutrients[None,:],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  # we want to have [w,h,9,c] so that the indexing is intuitive and consistent
  # for both neigh vectors.
  neigh_asking_nutrients = neigh_asking_nutrients.reshape(
      neigh_asking_nutrients.shape[:2] + (2, 9)).transpose((0, 1, 3, 2))
  tot_asking_nutrients = neigh_asking_nutrients.sum(2)
  # then output the percentage you can give.
  perc_to_give = (available_nutrients / tot_asking_nutrients.clip(1e-6)
                  ).clip(0, 1)
  # also update your nutrients accordingly.
  new_mat_nutrients = (available_nutrients - tot_asking_nutrients).clip(0.)
  # and give this energy to the agents, scaled by this percentage.
  absorb_perc = jax.lax.conv_general_dilated_patches(
      perc_to_give[None,:],
      (3, 3), (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))[0]
  # we want to have [w,h,9,c] so that the indexing is intuitive and consistent
  # for both neigh vectors.
  absorb_perc = absorb_perc.reshape(
      absorb_perc.shape[:2] + (2, 9)).transpose((0, 1, 3, 2))
  absorb_perc = absorb_perc.sum(2)
  absorbed_energy = absorb_perc * asking_nutrients

  ### Energy dissipation at every step.
  
  # dissipate energy by living.
  is_agent_grid = etd.is_agent_fn(env.type_grid)
  is_agent_grid_e_f = is_agent_grid.astype(jp.float32)[..., None]
  ag_spec_idx = etd.get_agent_specialization_idx(env.type_grid)
  # [w,h,3] @ [3,2] -> [w,h,2]
  agent_dissipation_rate = etd.dissipation_rate_per_spec[ag_spec_idx]
  dissipated_energy = (config.dissipation_per_step * agent_dissipation_rate *
                       is_agent_grid_e_f)

  new_agent_energy = (
      is_agent_grid_e_f * (
          env.state_grid[:,:, evm.EN_ST:evm.EN_ST+2] + absorbed_energy -
          dissipated_energy).clip(0, config.nutrient_cap))
  # new_energy includes earth, air, and agent energies.
  new_energy = new_mat_nutrients + new_agent_energy

  ### AGING: if the cell is older than half max lifetime, they leak energy.
  # energy is leaked in a linearly increasing fashion.

  age = env.state_grid[:,:, evm.AGE_IDX]
  half_lftm = config.max_lifetime/2
  reached_half_age = age >= half_lftm
  keep_perc = ((1. - reached_half_age) +
               reached_half_age * (1. - (age - half_lftm)/half_lftm).clip(0))
  new_energy *= keep_perc[..., None]

  # if an agent has 0 or negative energy, kill it.
  # replace the agent with the material that it has left. Otherwise, void.
  is_agent_energy_depleted = (new_energy == 0.)&(is_agent_grid[..., None])
  kill_agent_int = jp.any(is_agent_energy_depleted, axis=-1).astype(jp.uint32)
  kill_agent_e_f = kill_agent_int[:,:, None].astype(jp.float32)

  is_agent_energy_depleted_uint32 = is_agent_energy_depleted.astype(jp.uint32)
  replacement_type_grid = (
      (1 - is_agent_energy_depleted_uint32[:,:, 0]) * etd.types.EARTH +
      (1 - is_agent_energy_depleted_uint32[:,:, 1]) * etd.types.AIR)
  new_type_grid = env.type_grid * (1 - kill_agent_int) + kill_agent_int * (
      replacement_type_grid)
  # potentially reset states but the energy is preserved, since if an agent dies
  # it gets converted.
  new_state_grid = (env.state_grid * (1. - kill_agent_e_f)
                    ).at[:,:, evm.EN_ST:evm.EN_ST+2].set(new_energy)

  new_agent_id_grid = (
      env.agent_id_grid * (1 - kill_agent_int) +
      kill_agent_int * (jp.zeros_like(env.type_grid, dtype=jp.uint32)))
  return Environment(new_type_grid, new_state_grid, new_agent_id_grid)


### Processing age of agents.


def env_increase_age(env: Environment, etd: EnvTypeDef) -> Environment:
  """Increase the age of all agents by 1."""
  is_agent_m = etd.is_agent_fn(env.type_grid).astype(jp.float32)
  new_state_grid = env.state_grid.at[:,:, evm.AGE_IDX].set(
      env.state_grid[:,:, evm.AGE_IDX] + is_agent_m)
  return evm.update_env_state_grid(env, new_state_grid)
