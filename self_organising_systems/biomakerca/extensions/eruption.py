"""Code to reproduce the Eruption extension, introducing LAVA and FIRE.

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
import math
import jax
import tqdm
import jax.numpy as jp
import jax.random as jr
import mediapy as media
from jax import jit
from functools import partial

from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca import cells_logic
from self_organising_systems.biomakerca import env_logic
from self_organising_systems.biomakerca.display_utils import zoom, add_text_to_img
from self_organising_systems.biomakerca.step_maker import step_env

### New EnvTypeDef

ERUPTION_MATERIALS = evm.DEFAULT_MATERIALS + ["LAVA", "FIRE"]
ERUPTION_TYPE_COLOR_DICT = dict(
    evm.DEFAULT_TYPE_COLOR_DICT,
    **{"LAVA": jp.array([0.298, 0., 0.075]),  # RGB: 76,0,19
       "FIRE": jp.array([0.812, 0.063, 0.125])})  # RGB: 207,16,32
ERUPTION_STRUCTURE_DECAY_MATS_DICT = dict(
    evm.DEFAULT_STRUCTURE_DECAY_MATS_DICT,
    **{"LAVA": 1, "FIRE": -1})


class EruptionTypeDef(evm.DefaultTypeDef):
  """EnvTypeDef for Eruption.
  
  LAVA and FIRE are gravity, aging mats.
  LAVA also propagate structural integrity.
  """

  def __init__(self):
    super().__init__(
        ERUPTION_MATERIALS, evm.DEFAULT_AGENT_TYPES,
        ERUPTION_STRUCTURE_DECAY_MATS_DICT,
        evm.DEFAULT_DISSIPATION_RATE_PER_SPEC_DICT,
        ERUPTION_TYPE_COLOR_DICT)
    types = self.types

    # Now add material specific properties.
    # LAVA and FIRE are gravity_mats
    self.gravity_mats = jp.concatenate([
        self.gravity_mats,
        jp.array([types.LAVA, types.FIRE], dtype=jp.int32)], 0)
    self.structural_mats = self.agent_types
    # LAVA propagates structural integrity (but is not structural).
    self.propagate_structure_mats = jp.concatenate([
        self.propagate_structure_mats,
        jp.array([types.LAVA], dtype=jp.int32)], 0)
    # LAVA and FIRE age.
    self.aging_mats = jp.concatenate([
        self.aging_mats,
        jp.array([types.LAVA, types.FIRE], dtype=jp.int32)], 0)


### New ExclusiveOps


def is_burnable_fn(t, etd):
  """Return True if t is any of (LEAF,FLOWER,UNSPECIALIZED) agent types."""
  burnable_types = jp.array([
      etd.types.AGENT_LEAF, etd.types.AGENT_FLOWER,
      etd.types.AGENT_UNSPECIALIZED])
  return (t == burnable_types).any(axis=-1)


def conditional_update(arr, idx, val, cond):
  """Updates an entire row or a scalar value, based on the cond."""
  return arr.at[idx].set((1 - cond) * arr[idx] + cond * val)


def lava_cell_op(key, perc, config):
  """Create the exclusive function of LAVA cells.

  LAVA expires after a minimum age.
  LAVA has a falling-sand property. But, if it cannot fall, it then may
  burn nearby burnable cells (making them FIRE).
  """
  neigh_type, neigh_state, neigh_id = perc
  etd = config.etd

  # Create the output (and modify it based on conditions)
  t_upd_mask = env_logic.EMPTY_UPD_MASK
  a_upd_mask = env_logic.EMPTY_UPD_MASK
  t_upd_type = env_logic.EMPTY_UPD_TYPE
  a_upd_type = env_logic.EMPTY_UPD_TYPE
  t_upd_state = env_logic.make_empty_upd_state(config)
  a_upd_state = env_logic.make_empty_upd_state(config)
  t_upd_id = env_logic.EMPTY_UPD_ID
  a_upd_id = env_logic.EMPTY_UPD_ID


  ## lava expires after at least some time, (it ages) and then with
  # some chance.
  # If it expires, nothing else happens.
  key, ku = jr.split(key)
  min_age = 200
  extinction_prob = 0.05
  lava_extinguishes_i = jp.logical_and(
      neigh_state[4,evm.AGE_IDX] >= min_age,
      (jr.uniform(ku) < extinction_prob)).astype(jp.int32)

  # update the output accordingly: it becomes void (which is 0).
  t_upd_mask = t_upd_mask.at[4].set(lava_extinguishes_i)
  a_upd_mask = a_upd_mask.at[4].set(lava_extinguishes_i)

  done_i = lava_extinguishes_i
  # Then, check if you can fall.
  # for now, you can't fall out of bounds.
  # if you can fall, do nothing. Gravity will take care of it.
  can_fall_i = (1 - done_i) * jp.logical_and(
      neigh_type[7] != etd.types.OUT_OF_BOUNDS,
      (neigh_type[7] == etd.intangible_mats).any(),
  )

  done_i = (done_i + can_fall_i).clip(max=1)

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


  done_i = (done_i + can_fall_to_side_i).clip(max=1)
  # Else, check if you can ignite a fire.
  # for now, we take it easy. We check a random neighbor and if it is burnable,
  # we create a FIRE.
  k1, key = jr.split(key)
  rnd_idx = jr.choice(k1, jp.array([0,1,2,3, 5,6,7,8]))
  is_burnable_i = (1 - done_i) * is_burnable_fn(neigh_type[rnd_idx], etd
                                                ).astype(jp.int32)

  t_upd_mask = conditional_update(t_upd_mask, rnd_idx, 1.0, is_burnable_i)
  t_upd_type = conditional_update(t_upd_type, rnd_idx, etd.types.FIRE,
                                  is_burnable_i)

  return env_logic.ExclusiveOp(
      env_logic.UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
      env_logic.UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
  )



def fire_cell_op(key, perc, config):
  """Create the exclusive function of FIRE cells.

  FIRE may extinguish itself after a minimum age.
  Further, if it randomly catches a burnable type, it spreads there.
  """
  neigh_type, neigh_state, neigh_id = perc
  etd = config.etd

  # check whether it extinguishes
  key, ku = jr.split(key)
  # fire expires after at least some time, (it ages) and then with some chance.
  min_age = 10
  extinction_prob = 0.2
  fire_extinguishes_f = jp.logical_and(
      neigh_state[4, evm.AGE_IDX] >= min_age,
      (jr.uniform(ku) < extinction_prob)).astype(jp.float32)

  # look for a random neighbor and see if it is burnable
  rnd_idx = jr.choice(key, jp.array([0,1,2,3, 5,6,7,8]))
  is_burnable_f = is_burnable_fn(neigh_type[rnd_idx], etd).astype(jp.float32)
  is_burnable_i = is_burnable_f.astype(jp.int32)

  # if the target is burnable, we put that as target and convert it to FIRE.
  # regardless, if we extinguish, we update self.
  # so, if we extinguish but not spread, we target ourselves.

  t_idx = rnd_idx * is_burnable_i + 4 * (1 - is_burnable_i)
  # for redundancy, we put extinguish in bot target and actor updateops, in case
  # that we only extinguish.

  t_upd_mask = env_logic.EMPTY_UPD_MASK.at[t_idx].set(
      (is_burnable_f + fire_extinguishes_f).clip(max=1.))
  a_upd_mask = env_logic.EMPTY_UPD_MASK.at[t_idx].set(fire_extinguishes_f)
  # the target update type is 0 (VOID) if it is not burnable, but also if fire
  # extinguishes.
  t_upd_type = env_logic.EMPTY_UPD_TYPE.at[t_idx].set(
      etd.types.FIRE * is_burnable_i)
  # likewise here, if we update a, it is because fire is becoming void.
  a_upd_type = env_logic.EMPTY_UPD_TYPE
  t_upd_state = env_logic.make_empty_upd_state(config)
  a_upd_state = env_logic.make_empty_upd_state(config)
  t_upd_id = env_logic.EMPTY_UPD_ID
  a_upd_id = env_logic.EMPTY_UPD_ID

  return env_logic.ExclusiveOp(
      env_logic.UpdateOp(t_upd_mask, t_upd_type, t_upd_state, t_upd_id),
      env_logic.UpdateOp(a_upd_mask, a_upd_type, a_upd_state, a_upd_id),
  )

# Create the exclusive fs
def make_eruption_excl_fs(etd):
  return (
      (etd.types.AIR, cells_logic.air_cell_op),
      (etd.types.EARTH, cells_logic.earth_cell_op),
      (etd.types.LAVA, lava_cell_op),
      (etd.types.FIRE, fire_cell_op),
      )

### Eruption config
def get_eruption_config():
  """Return a new Eruption config.
  
  This config is quite trivial. It actually is taken from 'persistence'.
  """
  return evm.get_env_and_config(
      "persistence", width_type="petri", h=10, etd=EruptionTypeDef()).config

### Eruption environment

def create_eruption_env(h, config):
  """Create the Eruption environment.
  
  Note that w is always the same, regardless of h.
  """
  w = 360
  env = evm.create_default_environment(config, h, w)
  env = evm.place_seed(env, w//2, config)
  return env


@partial(jit, static_argnames=["etd"])
def add_eruption_lava_f(key, env, etd):
  """Lava operation specific to the Eruption env from create_eruption_env.
  
  Most parameters are hard coded, for now. If you need them to be variable,
  feel free to make a pull request.
  """
  # parameters, fixed for now.
  h, w = env[0].shape
  side_lengths = w * 2 // 5
  # left side
  wave_perc = 0.005
  wave_max = 1.
  wave_min = 0.25
  wave_max_stop = side_lengths // 10
  wave_lava_perc = jp.concatenate([
      jp.full([wave_max_stop], wave_max),
      jp.linspace(wave_max, wave_min, side_lengths - wave_max_stop)], 0)

  # do this only rarely:
  key, k1 = jr.split(key)
  wave_lava_perc = wave_lava_perc * (jr.uniform(k1) < wave_perc)
  key, k1 = jr.split(key)
  env = update_slice_with_lava(k1, env, 1, 0, wave_lava_perc, etd)

  # right side
  freq_min = 0.001
  freq_max = 0.007

  freq_lava_perc = jp.linspace(freq_min, freq_max, side_lengths)
  key, k1 = jr.split(key)
  env = update_slice_with_lava(k1, env, 1, -side_lengths, freq_lava_perc, etd)

  return env


@partial(jit, static_argnames=["etd"])
def update_slice_with_lava(key, env, r, c, lava_perc, etd):
  """Update the env with a dynamic slice of lava at (r,c) position.
  
  The lava is sampled with lava_perc (same size).
  """
  type_grid, state_grid, agent_id_grid = env
  if lava_perc.ndim == 1:
    lava_perc = lava_perc[None, :]
  lava_mask = (jr.uniform(key, lava_perc.shape) < lava_perc).astype(jp.uint32)

  type_grid = jax.lax.dynamic_update_slice(
      type_grid,
      lava_mask * etd.types.LAVA +
      jax.lax.dynamic_slice(type_grid, (r, c), lava_perc.shape) * (1-lava_mask),
      (r, c))
  empty_states = jp.zeros(
      [lava_perc.shape[0], lava_perc.shape[1], state_grid.shape[-1]])
  lava_mask_e = lava_mask[..., None]
  state_grid = jax.lax.dynamic_update_slice(
      state_grid,
      lava_mask_e * empty_states +
      (1-lava_mask_e) * jax.lax.dynamic_slice(state_grid, (r, c, 0),
                                              empty_states.shape),
      (r, c, 0))
  empty_agent_ids = jp.zeros(lava_perc.shape, dtype=jp.uint32)
  agent_id_grid = jax.lax.dynamic_update_slice(
      agent_id_grid,
      lava_mask * empty_agent_ids +
      (1-lava_mask) * jax.lax.dynamic_slice(agent_id_grid, (r, c),
                                            lava_perc.shape),
      (r, c))
  return evm.Environment(type_grid, state_grid, agent_id_grid)


### Testing functions

def run_eruption_env(
    key, config, programs, env, agent_logic, mutator, n_steps, zoom_sz=12,
    steps_per_frame=2, when_to_double_speed=[100, 500, 1000, 2000, 5000]):
  """Create a video running Eruption."""

  fps = 20
  excl_fs = make_eruption_excl_fs(config.etd)
  def make_frame(env):
    return zoom(evm.grab_image_from_env(env, config),zoom_sz)

  frame = make_frame(env)

  out_file = "video.mp4"
  with media.VideoWriter(out_file, shape=frame.shape[:2], fps=fps, crf=18
                        ) as video:
    for i in tqdm.trange(n_steps):
      if i in when_to_double_speed:
        steps_per_frame *= 2

      key, k1 = jr.split(key)
      env = add_eruption_lava_f(k1, env, config.etd)

      key, ku = jr.split(key)
      env, programs = step_env(
          ku, env, config, agent_logic, programs, do_reproduction=True,
          mutate_programs=True, mutator=mutator, excl_fs=excl_fs)

      if i % steps_per_frame == 0:
        video.add_image(make_frame(env))

  media.show_video(media.read_video(out_file))
  return programs, env


def test_freq_lava(key, st_env, config, init_program, agent_logic, mutator,
                   zoom_sz=8):
  """Create a video while testing init_program with a frequent lava environment.
  
  It does not return a metric of success. If that is needed, I recommend
  creating a new function that doesn't generate videos.
  """

  N_MAX_PROGRAMS = 128

  etd = config.etd
  excl_fs = make_eruption_excl_fs(etd)

  # on what step to double speed.
  when_to_double_speed = [100, 500, 1000]

  fps = 20
  env = st_env
  programs = jp.repeat(init_program[None, :], N_MAX_PROGRAMS, axis=0)

  ones = jp.ones([st_env[0].shape[1]])
  init_freq_lava_perc = 0.00025
  increment = 0.00025
  increase_every = 250
  max_perc = 1.

  step = 0
  steps_per_frame = 2
  freq_lava_perc = init_freq_lava_perc

  def make_frame(env, lava_perc):
    return add_text_to_img(
        zoom(evm.grab_image_from_env(env, config), zoom_sz),
        "LAVA FREQUENCY: {:.3f}%".format(lava_perc*100), 
        origin=(5, 35),
        fontScale=1.)

  frame = make_frame(env, freq_lava_perc)

  out_file = "video.mp4"
  with media.VideoWriter(out_file, shape=frame.shape[:2], fps=fps, crf=18
                         ) as video:
    video.add_image(frame)
    while freq_lava_perc < max_perc or math.isclose(freq_lava_perc, max_perc):
      for j in range(increase_every):
        step += 1
        if step in when_to_double_speed:
          steps_per_frame *= 2

        key, k1 = jr.split(key)
        env = update_slice_with_lava(k1, env, 1, 0, freq_lava_perc * ones,
                                     config.etd)

        key, ku = jr.split(key)
        env, programs = step_env(
            ku, env, config, agent_logic, programs, do_reproduction=True,
            mutate_programs=True, mutator=mutator, excl_fs=excl_fs)

        if step % steps_per_frame == 0:
          video.add_image(make_frame(env, freq_lava_perc))

      # check if they are alive
      any_alive = config.etd.is_agent_fn(env.type_grid).any()
      if any_alive:
        print("Survived with frequency: {:.3f}%".format(freq_lava_perc*100))
        freq_lava_perc += increment
      else:
        print("Not survived with frequency: {:.3f}%".format(freq_lava_perc*100))
        # make sure we visualize that.
        if step % steps_per_frame != 0:
          video.add_image(make_frame(env, freq_lava_perc))
        break
    # visualize for longer the final part.
    if freq_lava_perc > max_perc:
      freq_lava_perc = max_perc
    for j in range(increase_every):
      step += 1
      if step in when_to_double_speed:
        steps_per_frame *= 2

      key, k1 = jr.split(key)
      env = update_slice_with_lava(k1, env, 1, 0, freq_lava_perc * ones, etd)
      key, ku = jr.split(key)
      env, programs = step_env(
          ku, env, config, agent_logic, programs, do_reproduction=True,
          mutate_programs=True, mutator=mutator, excl_fs=excl_fs)

      if step % steps_per_frame == 0:
        video.add_image(make_frame(env, freq_lava_perc))

  media.show_video(media.read_video(out_file))


def test_wave_lava(
    key, st_env, config, init_program, agent_logic, mutator, zoom_sz=8):
  """Create a video while testing init_program with a wave lava environment.

  It does not return a metric of success. If that is needed, I recommend
  creating a new function that doesn't generate videos.
  """
  N_MAX_PROGRAMS = 128

  etd = config.etd
  excl_fs = make_eruption_excl_fs(etd)

  # on what step to double speed.
  when_to_double_speed = [100, 500, 1000]

  fps = 20
  env = st_env
  programs = jp.repeat(init_program[None, :], N_MAX_PROGRAMS, axis=0)

  ones = jp.ones([st_env[0].shape[1]])
  init_wave_lava_perc = 0.25
  increment = 0.05
  increase_every = 250
  max_perc = 1.

  step = 0
  steps_per_frame = 2
  wave_lava_perc = init_wave_lava_perc

  def make_frame(env, lava_perc):
    return add_text_to_img(
        zoom(evm.grab_image_from_env(env, config), zoom_sz),
        "LAVA FREQUENCY: {:.0f}%".format(lava_perc*100),
        origin=(5, 35),
        fontScale=1.)

  frame = make_frame(env, wave_lava_perc)

  out_file = "video.mp4"
  with media.VideoWriter(out_file, shape=frame.shape[:2], fps=fps, crf=18
                        ) as video:
    video.add_image(frame)
    # first round
    for j in range(increase_every):
      step += 1
      if step in when_to_double_speed:
        steps_per_frame *= 2

      key, ku = jr.split(key)
      env, programs = step_env(
          ku, env, config, agent_logic, programs, do_reproduction=True,
          mutate_programs=True, mutator=mutator, excl_fs=excl_fs)

      if step % steps_per_frame == 0:
        video.add_image(make_frame(env, wave_lava_perc))

    while wave_lava_perc < max_perc or math.isclose(wave_lava_perc, max_perc):
      key, k1 = jr.split(key)
      env = update_slice_with_lava(k1, env, 1, 0, wave_lava_perc * ones, etd)
      for j in range(increase_every):
        step += 1
        if step in when_to_double_speed:
          steps_per_frame *= 2

        key, ku = jr.split(key)
        env, programs = step_env(
            ku, env, config, agent_logic, programs, do_reproduction=True,
            mutate_programs=True, mutator=mutator, excl_fs=excl_fs)

        if step % steps_per_frame == 0:
          video.add_image(make_frame(env, wave_lava_perc))

      # check if they are alive
      any_alive = config.etd.is_agent_fn(env.type_grid).any()
      if any_alive:
        print("Survived with frequency: {:.0f}%".format(wave_lava_perc*100))
        wave_lava_perc += increment
      else:
        print("Not survived with frequency: {:.0f}%".format(wave_lava_perc*100))
        # make sure we visualize that.
        if step % steps_per_frame != 0:
          video.add_image(make_frame(env, wave_lava_perc))
        break
    # visualize for longer the final part.
    if wave_lava_perc > max_perc:
      wave_lava_perc = max_perc
    for j in range(increase_every):
      step += 1
      if step in when_to_double_speed:
        steps_per_frame *= 2

      key, ku = jr.split(key)
      env, programs = step_env(
          ku, env, config, agent_logic, programs, do_reproduction=True,
          mutate_programs=True, mutator=mutator, excl_fs=excl_fs)

      if step % steps_per_frame == 0:
        video.add_image(make_frame(env, wave_lava_perc))
          # check if they are alive
    if any_alive:
      # check if it actually survived.
      any_alive = config.etd.is_agent_fn(env.type_grid).any()
      if any_alive:
        print("Confirmed survived with frequency: {:.0f}%".format(wave_lava_perc*100))
        wave_lava_perc += increment
      else:
        print("Actually not survived with frequency: {:.0f}%".format(wave_lava_perc*100))
        # make sure we visualize that.
        if step % steps_per_frame != 0:
          video.add_image(make_frame(env, wave_lava_perc))
  media.show_video(media.read_video(out_file))
