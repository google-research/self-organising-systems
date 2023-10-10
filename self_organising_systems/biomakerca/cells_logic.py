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

import jax.numpy as jp
import jax.random as jr

from self_organising_systems.biomakerca.env_logic import ExclusiveOp
from self_organising_systems.biomakerca.env_logic import EMPTY_UPD_ID
from self_organising_systems.biomakerca.env_logic import EMPTY_UPD_MASK
from self_organising_systems.biomakerca.env_logic import EMPTY_UPD_TYPE
from self_organising_systems.biomakerca.env_logic import KeyType
from self_organising_systems.biomakerca.env_logic import make_empty_upd_state
from self_organising_systems.biomakerca.env_logic import PerceivedData
from self_organising_systems.biomakerca.env_logic import UpdateOp
from self_organising_systems.biomakerca.environments import EnvConfig
from self_organising_systems.biomakerca.utils import conditional_update


### AIR


def air_cell_op(key: KeyType, perc: PerceivedData, config: EnvConfig
                ) -> ExclusiveOp:
  """Create the exclusive function of AIR cells.
  
  AIR simply spreads through neighboring VOID cells.
  """
  neigh_type = perc[0]
  etd = config.etd
  
  # look for a random neighbor and see if it is VOID
  rnd_idx = jr.choice(key, jp.array([0, 1, 2, 3, 5, 6, 7, 8]))
  is_void_f = (neigh_type[rnd_idx] == etd.types.VOID).astype(jp.float32)
  is_void_i = is_void_f.astype(jp.int32)

  # if the target is VOID, we create a new AIR cell there.
  t_upd_mask = EMPTY_UPD_MASK.at[rnd_idx].set(is_void_f)
  # note that we dont update the actor, so we don't need to fill anything here.
  a_upd_mask = EMPTY_UPD_MASK
  t_upd_type = EMPTY_UPD_TYPE.at[rnd_idx].set(etd.types.AIR * is_void_i)
  # likewise here, if we update a, it is because fire is becoming void.
  a_upd_type = EMPTY_UPD_TYPE
  t_upd_state = make_empty_upd_state(config)
  a_upd_state = make_empty_upd_state(config)
  t_upd_id = EMPTY_UPD_ID
  a_upd_id = EMPTY_UPD_ID

  return ExclusiveOp(
      UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
      UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
  )


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
  etd = config.etd

  # Create the output (and modify it based on conditions)
  t_upd_mask = EMPTY_UPD_MASK
  a_upd_mask = EMPTY_UPD_MASK
  t_upd_type = EMPTY_UPD_TYPE
  a_upd_type = EMPTY_UPD_TYPE
  t_upd_state = make_empty_upd_state(config)
  a_upd_state = make_empty_upd_state(config)
  t_upd_id = EMPTY_UPD_ID
  a_upd_id = EMPTY_UPD_ID

  # First, check if you can fall.
  # for now, you can't fall out of bounds.
  can_fall_i = jp.logical_and(
      neigh_type[7] != etd.types.OUT_OF_BOUNDS,
      (neigh_type[7] == etd.intangible_mats).any(),
  )
  # if you can fall, do nothing. Gravity will take care of it.
  done_i = can_fall_i

  # Else, check if you can fall on a random side.
  # both the side and below need to be free.
  # if yes, do that.
  key, ku = jr.split(key)
  side_idx = jr.choice(ku, jp.array([3, 5]))
  down_side_idx = side_idx + 3

  can_fall_to_side_i = (1 - done_i) * (
      (neigh_type[side_idx] != etd.types.OUT_OF_BOUNDS)
      & (neigh_type[down_side_idx] != etd.types.OUT_OF_BOUNDS)
      & ((neigh_type[side_idx] == etd.intangible_mats).any())
      & ((neigh_type[down_side_idx] == etd.intangible_mats).any())
  )

  # update the outputs if true.
  t_upd_mask = conditional_update(t_upd_mask, side_idx, 1.0, can_fall_to_side_i)
  a_upd_mask = conditional_update(a_upd_mask, side_idx, 1.0, can_fall_to_side_i)
  # switch the types, states and ids
  t_upd_type = conditional_update(
      t_upd_type, side_idx, neigh_type[4], can_fall_to_side_i)
  a_upd_type = conditional_update(
      a_upd_type, side_idx, neigh_type[side_idx], can_fall_to_side_i)
  t_upd_state = conditional_update(
      t_upd_state, side_idx, neigh_state[4], can_fall_to_side_i)
  a_upd_state = conditional_update(
      a_upd_state, side_idx, neigh_state[side_idx], can_fall_to_side_i)
  t_upd_id = conditional_update(
      t_upd_id, side_idx, neigh_id[4], can_fall_to_side_i.astype(jp.uint32))
  a_upd_id = conditional_update(
      a_upd_id, side_idx, neigh_id[side_idx],
      can_fall_to_side_i.astype(jp.uint32))

  return ExclusiveOp(
      UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
      UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
  )
