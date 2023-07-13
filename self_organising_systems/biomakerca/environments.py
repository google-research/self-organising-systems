"""Information about the environment and the configurations of environments.

This class contains definitions for Environment and EnvConfig, as well as 
several default environments and ways to make them, and how to visualize them.

Due to the extreme number of constants present in this module, I recommend
importing environments as evm (environment module)

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

from jax import jit
from jax import numpy as jp

from self_organising_systems.biomakerca.utils import dotdict
from self_organising_systems.biomakerca.utils import vmap2

### Environment
# This is the state of any given environment. It is made of 3 grids:
# - type_grid: (uint32) what material type is in a given position.
# - state_grid: (f32) internal state of the cell. includes structural integrity,
#   age, nutrients and agent internal states.
# - agent_id_grid: (uint32) unique program identifiers for agents. This doesn't 
#   make sense for non-agents.
if "Environment" not in globals():
  Environment = namedtuple("Environment", "type_grid state_grid agent_id_grid")

# Helper functions for generating environments.
def make_empty_grid_type(config):
  return jp.zeros([config.w, config.h], dtype=jp.uint32)

def make_empty_grid_state(config):
  return jp.zeros([config.w, config.h, config.env_state_size])

def make_empty_grid_agent_id(config):
  return jp.zeros([config.w, config.h], dtype=jp.uint32)

# Helper functions for manipulating environments.
def update_env_type_grid(env: Environment, type_grid):
  return Environment(type_grid, env.state_grid, env.agent_id_grid)

def update_env_state_grid(env: Environment, state_grid):
  return Environment(env.type_grid, state_grid, env.agent_id_grid)

def update_env_agent_id_grid(env: Environment, agent_id_grid):
  return Environment(env.type_grid, env.state_grid, agent_id_grid)

### HARDCODED ENVIRONMENTAL CONSTANTS
# Some constants are, so far, so essential to the framework that they are 
# immutable. In the future, we may move some of them into EnvConfig.
# States indexes:
# 0: Structural integrity value
# 1: age value
# 2-3: nutrients for earth and air
# 4-(4+AGENT_STATE_SIZE): internal stateful data for agents
STR_IDX = 0
AGE_IDX = 1
EN_ST = 2
A_INT_STATE_ST = 4

# Don't be confused with the env type id below. The positions here are currently
# inverted.
EARTH_NUTRIENT_RPOS = 0
AIR_NUTRIENT_RPOS = 1

# Environment types.
# These are fixed for all configs. Allowing for different types for different 
# configs is a possible room for improvement. Note, however, that almost 
# everything depends on these env types, and many things are defined by this 
# fixed order. For instance, colors and agent specialization indexing.
# A refactor of ENV_TYPES should make sure that nothing gets broken.
# Note that they are not unsigned ints, since python doesn't have such a thing,
# but they should be treated as such.
ENV_TYPES = dotdict({
    # Void: an 'empty' space. This is intangible and can be filled by anything.
    #   In particular, Air spreads through void. Flowers, when they reproduce,
    #   turn into void.
    "VOID": 0,
    # Air: Intangible material; propagates air nutrients and Leaf agents can
    #   extract nutrients from them.
    "AIR": 1,
    # Earth: Propagates earth nutrients and Root agents can extract nutrients 
    #   from them. Subject to gravity and structural propagation.
    "EARTH": 2,
    # Immovable: A hard type that cannot be passed through and does not suffer
    #   from gravity. Moreover, it _generates_ earth nutrients and structural
    #   integrity.
    "IMMOVABLE": 3,
    # Sun: Sunrays; generates air nutrients and it is intangible.
    "SUN": 4,
    # Out of Bounds: This material only appears when being at the edge of an
    #   environment and observing neighbours out of bounds.
    "OUT_OF_BOUNDS": 5,
    ## Agents: any of the below are considered 'agents'. All agents can spawn
    #   new agents and can transfer nutrients among each other.
    # Unspecialized: the starting point of any organism, and the result of a
    #   spawn operation. They *tend* to consume fewer nutrients.
    "AGENT_UNSPECIALIZED": 6,
    # Root: Capable of absorbing earth nutrients.
    "AGENT_ROOT": 7,
    # Leaf: Capable of absorbing air nutrients.
    "AGENT_LEAF": 8,
    # Flower: Capable of performing a reproduce operation. They *tend* to
    #   consume more nutrients.
    "AGENT_FLOWER": 9,
})

## Material groups.
# these groups are used for environmental and cellular logics.

# Agents: The only materials that can have programs to be executed, and a
# respective agent_id.
AGENT_TYPES = jp.array([
    ENV_TYPES.AGENT_UNSPECIALIZED, ENV_TYPES.AGENT_ROOT,
    ENV_TYPES.AGENT_LEAF, ENV_TYPES.AGENT_FLOWER], dtype=jp.int32)
# Intangible: gravity will allow for elements to fall and swap places with
# intangible materials.
INTANGIBLE_MATS = jp.array([ENV_TYPES.VOID, ENV_TYPES.AIR, ENV_TYPES.SUN], 
                           dtype=jp.int32)
# Gravity: these materials are subject to gravity and can fall.
GRAVITY_MATS = jp.concatenate([
    jp.array([ENV_TYPES.EARTH], dtype=jp.int32), AGENT_TYPES], 0)
# Structural materials: a structure with a structural integrity > 0 will not 
# fall to gravity. Agents therefore can branch out.
# Earth is not structural. This means that even if its structural integrity were
# to be > 0, it would still crumble if it can.
# Trivia: I used to have a 'wood' material that was generated by agents but was 
# not an agent. This would have been structural, and would have a slow
# structural decay, allowing for bigger structures to be created.
STRUCTURAL_MATS = AGENT_TYPES
# Structural integrity propagation: even if some materials may not be,
# structural, they would still want to propagated structural integrity.
# Earth, for instance, propagates it while still remaining a crumbling material.
PROPAGATE_STRUCTURE_MATS = jp.concatenate([jp.array([
    ENV_TYPES.EARTH, ENV_TYPES.IMMOVABLE], dtype=jp.int32), AGENT_TYPES], 0)
# Agent spawnable: material that can be substituted by agent cells. In practice,
#   this is where agents can perform 'spawn' operations (and reproduce ops too).
#   There is extra leeway to allow for agent movement with this feature, but
#   for now, agents cannot move.
AGENT_SPAWNABLE_MATS = jp.array([
    ENV_TYPES.VOID, ENV_TYPES.AIR, ENV_TYPES.EARTH], dtype=jp.int32)

def is_agent_fn(env_type):
  """Return true if the cell is an agent. Works for any input dimensionality."""
  return (env_type[..., None] == AGENT_TYPES).any(axis=-1)

# Index order for agent specializations. Useful for code clarity.
SPECIALIZATION_IDXS = dotdict({
    "UNSPECIALIZED": 0,
    "ROOT": 1,
    "LEAF": 2,
    "FLOWER": 3,
})

def get_agent_specialization_idx(env_type):
  """Return the index of the agent specialization.
  Works for any dimensionality.
  If the input is not an agent, return 0.
  """
  return (env_type - 6).clip(0).astype(jp.uint32)

def get_agent_type_from_spec_idx(spec_idx):
  """Return the agent_type (uint32) from the specialization index.
  Works for any dimensionality.
  It assumes that the input is a valid specialization index.
  """
  return (6 + spec_idx).astype(jp.uint32)


### EnvConfig
# These are the configurations that make environments have different laws of
# 'physics'.
# Below, I added some 'default' values, so that one can create some configs and 
# environments without being overwhelmed with parameters to set.
# However, keep in mind that *several parameters are correlated!* 
# (To what? to an agent's survivalship.) Hence, modifying some parameters
# usually implies that you have to modify others too.


DEFAULT_STRUCT_INTEGRITY_CAP = 200
# indexed by the type, it tells how much structure decays.
# values should not matter for non structurally propagating cells.
DEFAULT_STRUCTURE_DECAY_MATS = jp.array([
    DEFAULT_STRUCT_INTEGRITY_CAP, # void
    DEFAULT_STRUCT_INTEGRITY_CAP, # air
    1, # earth
    0, # immovable
    DEFAULT_STRUCT_INTEGRITY_CAP, # sun
    DEFAULT_STRUCT_INTEGRITY_CAP, # out of bounds
    5,  # agent unspecialized
    5,  # agent root
    5,  # agent leaf
    5,  # agent flower
    ], dtype=jp.int32)

# how much each earth and air gives as material.
DEFAULT_ABSORBTION_AMOUNTS = jp.array([0.25, 0.25])
# How many nutrients are dissipated by agents per step.
DEFAULT_DISSIPATION_PER_STEP = jp.array([0.05, 0.05])
# A modifier of the dissipation based on the agent specialization.
# The y axis HERE is ordered like in SPECIALIZATION_IDXS.
# Note, however, the .T (transpose) at the end. So the result is an array of
# size (n_specializations, 2)
DEFAULT_DISSIPATION_RATE_PER_SPEC = jp.array([
    [0.5, 1.0, 1.0, 1.2], # cell specialization for EARTH nutrient
    [0.5, 1.0, 1.0, 1.2] #  for AIR nutrient
    ]).T
# cost of exclusive ops.
DEFAULT_SPAWN_COST = jp.array([0.1, 0.1])
DEFAULT_REPRODUCE_COST = jp.array([0.5, 0.5])
# cost of switching agents specializations.
DEFAULT_SPECIALIZE_COST = jp.array([0.02, 0.02])

# maximum value of nutrients for agents.
DEFAULT_NUTRIENT_CAP = jp.array([10., 10.])
# maximum value of nutrients for other materials.
DEFAULT_MATERIAL_NUTRIENT_CAP = jp.array([10., 10.])


class EnvConfig:
  """Configuration of an environment that changes the laws of physics.
  
  Attributes:
    agent_state_size: size of the internal states of agents.
    env_state_size: the total size of each cell. It defaults to 
      4 + agent_state_size
    struct_integrity_cap: the maximum value of structural integrity.
      This is what gets propagated by IMMUTABLE materials. Structural integrity
      decays the further away it goes from IMMUTABLE mats. If 0, gravity mats 
      fall.
    structure_decay_mats: how much each material decays in structural integrity
      while propagating the integrity to neighbors.
    absorbtion_amounts: how many nutrients earth and air mats give to agents at
      every step. If there are not enough nutrients, only up to that amount is 
      given, distributed to however many ask for them.
    dissipation_per_step: how many nutrients are dissipated by agents per step.
    dissipation_rate_per_spec: A modifier of the dissipation based on the agent 
      specialization.
    spawn_cost: the cost of performing a Spawn exclusive operation.
    reproduce_cost: the cost of performing a Reproduce operation.
    specialize_cost: the cost of changing an agent's specialization.
    reproduce_min_dist: the minimum distance from a reproducing flower where a
      seed can be placed.
    reproduce_max_dist: the maximum distance from a reproducing flower where a
      seed can be placed.
    n_reproduce_per_step: how many reproduce ops can be selected per step to be
      executed. In effect, this means that flowers may not execute reproduce ops
      as soon as they desire, but they may wait, depending on how many other 
      flowers are asking the same in the environment. See that competition over
      a scarse external resource (bees, for instance).
    nutrient_cap: maximum value of nutrients for agents.
    material_nutrient_cap: maximum value of nutrients for other materials.
    max_lifetime: the maximum lifetime of an organism. Agents age at every step.
      after reaching half max_lifetime, they will start to lose a linearly 
      increasing number of materials until they would lose 100% of them at 
      max_lifetime. You can essentially disable this feature by setting an 
      enormous value for this attribute.
    """
  
  def __init__(self,
               agent_state_size=2,
               struct_integrity_cap=DEFAULT_STRUCT_INTEGRITY_CAP,
               structure_decay_mats=DEFAULT_STRUCTURE_DECAY_MATS, 
               absorbtion_amounts=DEFAULT_ABSORBTION_AMOUNTS, 
               dissipation_per_step=DEFAULT_DISSIPATION_PER_STEP,
               dissipation_rate_per_spec=DEFAULT_DISSIPATION_RATE_PER_SPEC,
               spawn_cost=DEFAULT_SPAWN_COST,
               reproduce_cost=DEFAULT_REPRODUCE_COST,
               specialize_cost=DEFAULT_SPECIALIZE_COST,
               reproduce_min_dist=5,
               reproduce_max_dist=15,
               n_reproduce_per_step=2,
               nutrient_cap=DEFAULT_NUTRIENT_CAP,
               material_nutrient_cap=DEFAULT_MATERIAL_NUTRIENT_CAP,
               max_lifetime=int(1e6)):
    self.agent_state_size = agent_state_size
    self.env_state_size = 4 + self.agent_state_size
    self.struct_integrity_cap = struct_integrity_cap
    self.structure_decay_mats = structure_decay_mats
    self.absorbtion_amounts = absorbtion_amounts
    self.dissipation_per_step = dissipation_per_step
    self.dissipation_rate_per_spec = dissipation_rate_per_spec
    self.spawn_cost = spawn_cost
    self.reproduce_cost = reproduce_cost
    self.specialize_cost = specialize_cost
    self.reproduce_min_dist = reproduce_min_dist
    self.reproduce_max_dist = reproduce_max_dist
    self.n_reproduce_per_step = n_reproduce_per_step
    self.nutrient_cap = nutrient_cap
    self.material_nutrient_cap = material_nutrient_cap
    self.max_lifetime = max_lifetime



### Helpers for making environments.

# namedtuple useful to define pairs of envs and configs.
if 'EnvAndConfig' not in globals():
  EnvAndConfig = namedtuple('EnvAndConfig', 'env config')



def add_agent_to_env(env, x, y, init_nutrients, aid,
                     init_spec=ENV_TYPES.AGENT_UNSPECIALIZED):
  """Add an agent to an environment in a specific (x,y) position."""
  return Environment(
      env.type_grid.at[x, y].set(init_spec),
      env.state_grid.at[x, y, EN_ST:EN_ST+2].set(init_nutrients),
      env.agent_id_grid.at[x, y].set(aid))

def set_nutrients_to_materials(env, earth_nut_val=None, air_nut_val=None):
  """Set the nutrient values of all EARTH and AIR cells to the respective vals.
  """
  assert earth_nut_val is not None or air_nut_val is not None, (
      "At least one nutrient value must be not None.")
  state_grid = env.state_grid
  if earth_nut_val is not None:
    is_earth = (env.type_grid == ENV_TYPES.EARTH).astype(jp.float32)
    state_grid = state_grid.at[:, :, EN_ST+EARTH_NUTRIENT_RPOS].set(
        is_earth * earth_nut_val +
        (1. - is_earth) * state_grid[:, :, EN_ST+EARTH_NUTRIENT_RPOS])
  if air_nut_val is not None:
    is_air = (env.type_grid == ENV_TYPES.AIR).astype(jp.float32)
    state_grid = state_grid.at[:, :, EN_ST+AIR_NUTRIENT_RPOS].set(
        is_air * air_nut_val +
        (1. - is_air) * state_grid[:, :, EN_ST+AIR_NUTRIENT_RPOS])
  return update_env_state_grid(env, state_grid)

def create_enviroment_filled_with_type(config, h, w, env_type):
  type_grid = jp.full([h, w], env_type, dtype=jp.uint32)
  state_grid = jp.zeros([h, w, config.env_state_size])
  agent_id_grid = jp.zeros([h, w], dtype=jp.uint32)
  return Environment(type_grid, state_grid, agent_id_grid)


def create_default_environment(config, h, w, with_earth=True,
                               init_nutrient_perc=0.1):
  """Create a simple default environment.
  It is filled with air, with immovable in the bottom and sun on top.
  If with_earth is True, it also contains earth covering the bottom half of the 
  environment.
  init_nutrient_perc defines the initialized nutrient values for earth and air,
  as a percentage of config.material_nutrient_cap
  """
  env = create_enviroment_filled_with_type(config, h, w, ENV_TYPES.AIR)
  type_grid = env.type_grid.at[-1,:].set(ENV_TYPES.IMMOVABLE)
  type_grid = type_grid.at[0,:].set(ENV_TYPES.SUN)
  if with_earth:
    type_grid = type_grid.at[-h//2:-1, :].set(ENV_TYPES.EARTH)
  env = update_env_type_grid(env, type_grid)

  env = set_nutrients_to_materials(
      env,
      init_nutrient_perc * config.material_nutrient_cap[EARTH_NUTRIENT_RPOS],
      init_nutrient_perc * config.material_nutrient_cap[AIR_NUTRIENT_RPOS])
  return env


def get_env_and_config(ec_id: str, width_type="wide", h=72) -> EnvAndConfig:
  """Return a prepared EnvAndConfig from a limited selection.
  
  The Environment and config get dynamically generated.
  
  The height of the env is predetermined. The width can be chose.
  Valid width_type:
  - 'wide': the default; makes the width 4 times larger than the height. Useful
    for exploring evolution on long timelines.
  - 'landscape': crates a 16:9 screen ratio. Useful for making visually
    pleasing environments.
  - 'square': creates a 1:1 screen ratio. Useful for making good looking small
    environments.
  - 'petri': creates a 1:2 screen ratio. Useful for petri dish-like experiments.
  - any integer: explicitly selecting the size.
  
  
  Valid ec_ids:
  - 'persistence': long lifetime, spawn and reproduce are expensive,
    but dissipation is very low.
  - 'pestilence': short lifetime. Spawn, reproduce and specialize
    are expensive, but dissipation is very low.
  - 'collaboration': agents don't age. dissipation is high, reproduce and
    specialize are costly. There is a higher structural integrity cap than usual
    to allow for longer structures.
  - 'sideways': long lifetime, spawn and reproduce are expensive,
    but dissipation is very low. It has nutrients only at the extremes: SUN is 
    only on the top left, IMMOVABLE is only on the bottom right. The structural
    integrity cap is increased to account for that. Due to this setup, it is 
    recommended to set 'landscape' width for this.
  """
  
  def infer_width(h, width_type):
    if isinstance(width_type, int):
      return width_type
    if width_type == "wide":
      return 4 * h
    if width_type == "landscape":
      return int(1.778 * h)
    if width_type == "square":
      return h
    if width_type == "petri":
      return h // 2
    raise ValueError("invalid width_type", width_type)
  
  if ec_id == "persistence":
    w = infer_width(h, width_type)
    config = EnvConfig(
        material_nutrient_cap=jp.array([10., 10.]),
        nutrient_cap=jp.array([10., 10.]),
        dissipation_per_step=jp.array([0.01, 0.01]),
        absorbtion_amounts=jp.array([0.25, 0.25]),
        spawn_cost=jp.array([0.75, 0.75]),
        reproduce_cost=jp.array([1., 1.]),
        specialize_cost=jp.array([0.02, 0.02]),
        reproduce_min_dist=15, reproduce_max_dist=35,
        max_lifetime=10000,
        struct_integrity_cap=200,
        )
    env = create_default_environment(config, h, w, init_nutrient_perc=0.2)

    t_aid = 0
    half_h = h // 2
    half_w = w // 2

    agent_init_nutrients = (config.dissipation_per_step * 4 +
                            config.specialize_cost)
    # add a seed at the center.
    env = add_agent_to_env(env, -half_h, half_w, agent_init_nutrients, t_aid)
    env = add_agent_to_env(env, -half_h-1, half_w, agent_init_nutrients, t_aid)

    return EnvAndConfig(env, config)
  
  if ec_id == "pestilence":
    w = infer_width(h, width_type)
    config = EnvConfig(
        material_nutrient_cap=jp.array([10., 10.]),
        nutrient_cap=jp.array([10., 10.]),
        dissipation_per_step=jp.array([0.01, 0.01]),
        absorbtion_amounts=jp.array([0.25, 0.25]),
        spawn_cost=jp.array([0.75, 0.75]),
        reproduce_cost=jp.array([1., 1.]),
        specialize_cost=jp.array([0.05, 0.05]),
        reproduce_min_dist=15, reproduce_max_dist=35,
        max_lifetime=300,
        struct_integrity_cap=200,
        )
    env = create_default_environment(config, h, w, init_nutrient_perc=0.2)

    t_aid = 0
    half_h = h // 2
    half_w = w // 2

    agent_init_nutrients = (config.dissipation_per_step * 4 +
                            config.specialize_cost)
    # add a seed at the center.
    env = add_agent_to_env(env, -half_h, half_w, agent_init_nutrients, t_aid)
    env = add_agent_to_env(env, -half_h-1, half_w, agent_init_nutrients, t_aid)

    return EnvAndConfig(env, config)
  
  if ec_id == "collaboration":
    w = infer_width(h, width_type)
    config = EnvConfig(
        material_nutrient_cap=jp.array([10., 10.]),
        nutrient_cap=jp.array([10., 10.]),
        dissipation_per_step=jp.array([0.05, 0.05]),
        absorbtion_amounts=jp.array([0.25, 0.25]),
        spawn_cost=jp.array([0.25, 0.25]),
        reproduce_cost=jp.array([1., 1.]),
        specialize_cost=jp.array([0.05, 0.05]),
        reproduce_min_dist=15, reproduce_max_dist=35,
        max_lifetime=int(1e8),  # essentially, they don't age.
        struct_integrity_cap=300,
        )
    env = create_default_environment(config, h, w, init_nutrient_perc=0.2)

    t_aid = 0
    half_h = h // 2
    half_w = w // 2

    agent_init_nutrients = (config.dissipation_per_step * 4 +
                            config.specialize_cost)
    # add a seed at the center.
    env = add_agent_to_env(env, -half_h, half_w, agent_init_nutrients, t_aid)
    env = add_agent_to_env(env, -half_h-1, half_w, agent_init_nutrients, t_aid)

    return EnvAndConfig(env, config)
  
  if ec_id == "sideways":
    # This environment is recommended to be used with 'landscape' width.
    # this is because the nutrient generators are only present on opposite 
    # east-west directions.
    w = infer_width(h, width_type)
    config = EnvConfig(
        material_nutrient_cap=jp.array([10., 10.]),
        nutrient_cap=jp.array([10., 10.]),
        dissipation_per_step=jp.array([0.01, 0.01]),
        absorbtion_amounts=jp.array([0.25, 0.25]),
        spawn_cost=jp.array([0.75, 0.75]),
        reproduce_cost=jp.array([1., 1.]),
        specialize_cost=jp.array([0.02, 0.02]),
        reproduce_min_dist=15, reproduce_max_dist=35,
        max_lifetime=10000,
        struct_integrity_cap=400,
        )
    env = create_default_environment(config, h, w, init_nutrient_perc=0.2)

    t_aid = 0
    half_h = h // 2
    half_w = w // 2

    agent_init_nutrients = (config.dissipation_per_step * 4 +
                            config.specialize_cost)
    # add a seed at the center.
    env = add_agent_to_env(env, -half_h, half_w, agent_init_nutrients, t_aid)
    env = add_agent_to_env(env, -half_h-1, half_w, agent_init_nutrients, t_aid)
    
    # now the kicker: remove nutrients from top and bottom.
    env = update_env_type_grid(
        env, env.type_grid.at[0,:].set(ENV_TYPES.AIR).at[-1, :].set(ENV_TYPES.EARTH))
    # place generators on the sides.
    for wi in range(10):
      env = update_env_type_grid(
          env, env.type_grid.at[wi,:20-wi*2].set(ENV_TYPES.SUN).at[
              -1-wi, -20+wi*2:].set(ENV_TYPES.IMMOVABLE))
  
    # fill the nutrients appropriately.
    env = set_nutrients_to_materials(
        env,
        earth_nut_val=0.2*config.material_nutrient_cap[EARTH_NUTRIENT_RPOS],
        air_nut_val=0.2*config.material_nutrient_cap[AIR_NUTRIENT_RPOS])

    return EnvAndConfig(env, config)
  

def slice_environment_from_center(env, new_w):
  """Cuts a vertical slice of the environment centered at the original center,
  but with new_w as the final second dimension size.
  
  Very useful for petri dish-like experiments to evolve single agents before
  deploying them in a larger environment.
  """
  w = env.type_grid.shape[1]
  new_w_st = w//2 - new_w//2
  new_w_end = w//2 + new_w//2
  return Environment(
      env.type_grid[:, new_w_st:new_w_end],
      env.state_grid[:, new_w_st:new_w_end],
      env.agent_id_grid[:, new_w_st:new_w_end])

### Visualization of environments
ENV_COLOR_MAP = jp.array([
    [1., 1., 1.], # 0: Void
    [0.84, 1., 1.], # 1: Air
    [0.769, 0.643, 0.518], # 2: Earth
    [0., 0., 0.], # 3: Immovable
    [1., 1., 0.5], # 4: Sun
    [1., 0., 0.], # 5: Out of bounds, error color (should not use this)
    [0.65, 0.68, 0.65], # 6: Agent Unspecialized
    [0.52, 0.39, 0.14], # 7: Agent Root
    [0.16, 0.49, 0.10], # 8: Agent Leaf
    [1., 0.42, 0.71], # 9: Agent Flower
    ])


@partial(jit, static_argnames=["config"])
def grab_image_from_env(env, config):
  """Create a visualization of the environment.
  
  Resulting values are floats ranging from [0,1].
  """

  def map_cell(cell_type, state):
    env_c = ENV_COLOR_MAP[cell_type]
    # in particular, EARTH and AIR colors degrade by how little nutrients they have.
    is_earth_f = (cell_type == ENV_TYPES.EARTH).astype(jp.float32)
    is_air_f = (cell_type == ENV_TYPES.AIR).astype(jp.float32)
    env_c = env_c * (1. - is_earth_f) + env_c * is_earth_f * (0.3 +(
        state[EN_ST+EARTH_NUTRIENT_RPOS]/config.material_nutrient_cap[EARTH_NUTRIENT_RPOS])*0.7)
    env_c = env_c * (1. - is_air_f) + env_c * is_air_f * (0.3 +(
        state[EN_ST+AIR_NUTRIENT_RPOS]/config.material_nutrient_cap[AIR_NUTRIENT_RPOS])*0.7)
    return env_c
  return vmap2(map_cell)(env.type_grid, env.state_grid)
