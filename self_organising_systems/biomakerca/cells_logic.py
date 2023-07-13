"""Module containing the logic of materials.

To implement logic for new materials, or different logics, simply implement a
cell-level function with the following signature:
[KeyType, PerceivedData, EnvConfig] -> [ExclusiveOp]

Note that ExclusiveOp is not validated externally, so it does what you write!

Currently, contains logic for AIR and EARTH materials.

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

import jax
import jax.numpy as jp
import jax.random as jr

from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.env_logic import ExclusiveOp
from self_organising_systems.biomakerca.env_logic import EMPTY_UPD_ID
from self_organising_systems.biomakerca.env_logic import EMPTY_UPD_MASK
from self_organising_systems.biomakerca.env_logic import EMPTY_UPD_TYPE
from self_organising_systems.biomakerca.env_logic import KeyType
from self_organising_systems.biomakerca.env_logic import make_empty_exclusive_op_cell
from self_organising_systems.biomakerca.env_logic import make_empty_upd_state
from self_organising_systems.biomakerca.env_logic import PerceivedData
from self_organising_systems.biomakerca.env_logic import UpdateOp
from self_organising_systems.biomakerca.environments import ENV_TYPES
from self_organising_systems.biomakerca.environments import EnvConfig


### AIR


def air_cell_op(key: KeyType, perc: PerceivedData, config: EnvConfig
                ) -> ExclusiveOp:
  """Create the exclusive function of AIR cells.
  
  AIR simply spreads through neighboring VOID cells.
  """
  # choose a random neighbor.
  k1, key = jr.split(key)
  neigh_idx = jr.choice(k1, jp.array([0, 1, 2, 3, 5, 6, 7, 8]))

  def action_fn(neigh_idx):
    t_upd_mask = EMPTY_UPD_MASK.at[neigh_idx].set(1.0)
    a_upd_mask = EMPTY_UPD_MASK
    t_upd_type = EMPTY_UPD_TYPE.at[neigh_idx].set(ENV_TYPES.AIR)
    a_upd_type = EMPTY_UPD_TYPE
    t_upd_state = make_empty_upd_state(config)
    a_upd_state = make_empty_upd_state(config)
    t_upd_id = EMPTY_UPD_ID
    a_upd_id = EMPTY_UPD_ID

    return ExclusiveOp(
        UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
        UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
    )

  # needs to be in bounds and only spreads through void.
  result = jax.lax.cond(
      perc.neigh_type[neigh_idx] == ENV_TYPES.VOID,
      action_fn,  # true
      lambda _: make_empty_exclusive_op_cell(config),  # false
      neigh_idx,
  )
  return result


# EARTH


def earth_cell_op(key: KeyType, perc: PerceivedData, config: EnvConfig
                  ) -> ExclusiveOp:
  """Create the exclusive function of EARTH cells.
  
  EARTH behaves similarly to the typical falling-sand algorithms.
  If it can fall below, gravity already takes care of that, so nothing happens
  here. If it cannot fall below, but it can fall sideways, we move the cell to
  the side. NOTE: it moves to the side, not down-side! This is because then
  gravity will push it down.
  """
  neigh_type, neigh_state, neigh_id = perc

  # first check if you can fall.
  # for now, you can't fall out of bounds.
  can_fall = jp.logical_and(
      neigh_type[7] != ENV_TYPES.OUT_OF_BOUNDS,
      (neigh_type[7] == evm.INTANGIBLE_MATS).any(),
  )

  # if you can fall, do nothing. Gravity will take care of it.

  def execute_move_f(side_idx):
    t_upd_mask = EMPTY_UPD_MASK.at[side_idx].set(1.0)
    a_upd_mask = EMPTY_UPD_MASK.at[side_idx].set(1.0)

    # switch the types, states and ids
    t_upd_type = EMPTY_UPD_TYPE.at[side_idx].set(ENV_TYPES.EARTH)
    a_upd_type = EMPTY_UPD_TYPE.at[side_idx].set(neigh_type[side_idx])
    t_upd_state = make_empty_upd_state(config).at[side_idx].set(neigh_state[4])
    a_upd_state = (
        make_empty_upd_state(config).at[side_idx].set(neigh_state[side_idx])
    )
    t_upd_id = EMPTY_UPD_ID.at[side_idx].set(neigh_id[4])
    a_upd_id = EMPTY_UPD_ID.at[side_idx].set(neigh_id[side_idx])

    return ExclusiveOp(
        UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
        UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
    )

  # Else, check if you can fall on a random side.
  # both the side and below need to be free.
  def side_walk_f(key):
    rnd_idx = jr.choice(key, jp.array([3, 5]))
    down_rnd_idx = rnd_idx + 3

    can_fall = (
        (neigh_type[rnd_idx] != ENV_TYPES.OUT_OF_BOUNDS)
        & (neigh_type[down_rnd_idx] != ENV_TYPES.OUT_OF_BOUNDS)
        & ((neigh_type[rnd_idx] == evm.INTANGIBLE_MATS).any())
        & ((neigh_type[down_rnd_idx] == evm.INTANGIBLE_MATS).any())
    )

    return jax.lax.cond(
        can_fall,
        execute_move_f,
        lambda _: make_empty_exclusive_op_cell(config),
        rnd_idx,
    )

  k1, key = jr.split(key)
  result = jax.lax.cond(
      can_fall,
      lambda k: make_empty_exclusive_op_cell(config),  # true
      side_walk_f,  # false
      k1,
  )

  return result
