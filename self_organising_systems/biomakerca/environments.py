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
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial

import jax
from jax import jit
from jax import numpy as jp

from self_organising_systems.biomakerca.utils import dotdict
from self_organising_systems.biomakerca.utils import vmap2
from self_organising_systems.biomakerca.utils import stringify_class

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

### Environment types.

def make_env_types(materials, agent_types):
  """Utility function to make env_type enum dictionary.

  Args:
    materials: List of strings of material types.
    agent_types: List of strings of agent types.
  Returns:
    a dotdict to represent the enum of all env types. The enum has increasing 
    values, starting from 0 for materials and then continuing with agent_types.
  """
  n_mats = len(materials)
  return dotdict({k: v for k, v in zip(materials, range(n_mats))},
                 **{k: v for k, v in zip(
                     agent_types, range(n_mats, n_mats + len(agent_types)))})

def make_specialization_idxs(agent_types):
  """Utility function to make specialization_idxs enum dictionary.

  Args:
    agent_types: List of strings of agent types.
  Returns:
    a dotdict to represent the order of agent_types.
  """
  return dotdict({k: v for k, v in zip(agent_types, range(len(agent_types)))})


class EnvTypeDef(ABC):
  """Base class for defining environment types and their behaviors.
  
  Attributes:
    types: Enum for all possible cell types. Note that they are not unsigned 
      ints, since python doesn't have such a thing, but they should be treated 
      as such. See DEFAULT_MATERIALS and DEFAULT_AGENT_TYPES for an example of 
      the minimal cell types needed and what they do.
    materials_list: List of strings containing the names of materials. This 
      excludes agent types, which are stored in a different way.
    agent_types: The only materials that can have programs to be executed. A jp
      array.
    intangible_mats: Intangible materials. Gravity will allow for elements to 
      fall and swap places with intangible materials. A jp array.
    gravity_mats: Gravity materials. These materials are subject to gravity and 
      can fall. A jp array.
    structural_mats: Structural materials. A structure with a structural 
      integrity > 0 will not fall to gravity. Agents therefore can branch out.
      Usually, Earth is not structural. This means that even if its structural 
      integrity were to be > 0, it would still crumble if it can.
      Trivia: I used to have a 'wood' material that was generated by agents but 
      was not an agent. This would have been structural, and would have a slow
      structural decay, allowing for bigger structures to be created.
    propagate_structure_mats: Materials that propagate structural integrity.
      Even if some materials may not be structural, they would still want to 
      propagate structural integrity. Usually Earth, for instance, propagates it 
      while still remaining a crumbling material.
    agent_spawnable_mats: Materials that can be substituted by agent cells.
      In practice, this is where agents can perform 'spawn' operations (and 
      reproduce ops too). There is extra leeway to allow for agent movement with
      this feature, but for now, agents cannot move.
    aging_mats: Materials that age at every step. For instance, all agent cells
      are expected to age.
    specialization_idxs: Index order for agent specializations. Useful for code 
      clarity.
    structure_decay_mats: Indexed by the type enum, it tells how much structure
      decays. Values should not matter for non structurally propagating cells.
      A jp array.
    dissipation_rate_per_spec: A modifier of the dissipation based on the agent
      specialization. A jp array of size (n_specializations, 2).
    type_color_map: A map of type to color. Useful for visualizations.
      A jp array of size (n_types, 3).

  __create_types__ is highly recommended to be called as a first step during 
  initialization. This way, types, specialization_idxs, agent_types and
  materials_list will be initialized in a consistent way.

  __post_init__ should be called at the end of the subclass __init__ to make 
  sure that all attributes are initialized.
  """

  def __post_init__(self):
    self._has_required_attributes()

  def _has_required_attributes(self):
    req_attrs = [
        "types", "materials_list", "agent_types", "intangible_mats", 
        "gravity_mats", "structural_mats", "propagate_structure_mats", 
        "agent_spawnable_mats", "specialization_idxs", "structure_decay_mats",
        "aging_mats", "dissipation_rate_per_spec", "type_color_map"]
    for attr in req_attrs:
      if not hasattr(self, attr):
        raise AttributeError(f"Missing attribute: '{attr}'")

  def __create_types__(self, materials, agent_types):
    # save materials as a list. Used for get_agent_specialization_idx and
    # get_agent_type_from_spec_idx.
    self.materials_list = materials
    # create types
    self.types = make_env_types(materials, agent_types)
    # create specialization_idxs
    self.specialization_idxs = make_specialization_idxs(agent_types)
    # convert agent_types into a jp array
    self.agent_types = jp.array(
        [self.types[t] for t in agent_types], dtype=jp.int32)

  def get_agent_specialization_idx(self, env_type):
    """Return the index of the agent specialization.
    
    This function must be compatible with self.specialization_idxs and types. 
    This means that if specialization_idxs.AGENT_LEAF == 2, then 
    self.get_agent_specialization_idx(type.AGENT_LEAF) == 2
    Works for any dimensionality.
    If the input is not an agent, return 0.
    """
    return jp.array(env_type - len(self.materials_list)).clip(0).astype(jp.uint32)

  def get_agent_type_from_spec_idx(self, spec_idx):
    """Return the agent_type (uint32) from the specialization index.

    This function must be compatible with specialization_idxs and types.
    Essentially,
    self.get_agent_type_from_spec_idx(specialization_idxs.AGENT_LEAF) ==
    type.AGENT_LEAF, and so on.

    Works for any dimensionality.
    It assumes that the input is a valid specialization index.
    """
    return jp.array(len(self.materials_list) + spec_idx).astype(jp.uint32)

  def is_agent_fn(self, env_type):
    """Return true if the cell is an agent. Works for any input dimensionality.
    """
    return (env_type[..., None] == self.agent_types).any(axis=-1)
  
  def __str__(self):
    return stringify_class(self, exclude_list=["type_color_map"])


DEFAULT_MATERIALS = [
    # Void: an 'empty' space. This is intangible and can be filled by anything.
    #   In particular, Air spreads through void. Flowers, when they reproduce,
    #   turn into void.
    "VOID",
    # Air: Intangible material; propagates air nutrients and Leaf agents can
    #   extract nutrients from them.
    "AIR",
    # Earth: Propagates earth nutrients and Root agents can extract nutrients 
    #   from them. Subject to gravity and structural propagation.
    "EARTH",
    # Immovable: A hard type that cannot be passed through and does not suffer
    #   from gravity. Moreover, it _generates_ earth nutrients and structural
    #   integrity.
    "IMMOVABLE",
    # Sun: Sunrays; generates air nutrients. Can't be interacted with.
    "SUN",
    # Out of Bounds: This material only appears when being at the edge of an
    #   environment and observing neighbours out of bounds.
    "OUT_OF_BOUNDS",
]

DEFAULT_AGENT_TYPES = [
    # Unspecialized: the starting point of any organism, and the result of a
    #   spawn operation. They *tend* to consume fewer nutrients.
    "AGENT_UNSPECIALIZED",
    # Root: Capable of absorbing earth nutrients.
    "AGENT_ROOT",
    # Leaf: Capable of absorbing air nutrients.
    "AGENT_LEAF",
    # Flower: Capable of performing a reproduce operation. They *tend* to
    #   consume more nutrients.
    "AGENT_FLOWER",
]

# indexed by the type, it tells how much structure decays.
# values should not matter for non structurally propagating cells.
DEFAULT_STRUCTURE_DECAY_MATS_DICT = {
    "VOID": -1,
    "AIR": -1,
    "EARTH": 1,
    "IMMOVABLE": 0,
    "SUN": -1,
    "OUT_OF_BOUNDS": -1,
    "AGENT_UNSPECIALIZED": 5,
    "AGENT_ROOT": 5,
    "AGENT_LEAF": 5,
    "AGENT_FLOWER": 5,
}

# A modifier of the dissipation based on the agent specialization.
# the first element is for the earth nutrients, the second element is for the 
# air nutrients.
DEFAULT_DISSIPATION_RATE_PER_SPEC_DICT = {
    "AGENT_UNSPECIALIZED": jp.array([0.5, 0.5]),
    "AGENT_ROOT": jp.array([1.0, 1.0]),
    "AGENT_LEAF": jp.array([1.0, 1.0]),
    "AGENT_FLOWER": jp.array([1.2, 1.2]),
}

# Colors for visualising the default types.
DEFAULT_TYPE_COLOR_DICT = {
    "VOID": jp.array([1., 1., 1.]),  # RGB: 255,255,255
    "AIR": jp.array([0.84, 1., 1.]),  # RGB: 217,255,255
    "EARTH": jp.array([0.769, 0.643, 0.518]),  # RGB: 198,164,132
    "IMMOVABLE": jp.array([0., 0., 0.]),  # RGB: 0,0,0
    "SUN": jp.array([1., 1., 0.5]),  # RGB: 255,255,127
    "OUT_OF_BOUNDS": jp.array([1., 0., 0.]),  # error color (should not be seen)
    "AGENT_UNSPECIALIZED": jp.array([0.65, 0.68, 0.65]),  # RGB: 166,173,166
    "AGENT_ROOT": jp.array([0.52, 0.39, 0.14]),  # RGB: 133,99,36
    "AGENT_LEAF": jp.array([0.16, 0.49, 0.10]),  # RGB: 41,125,26
    "AGENT_FLOWER": jp.array([1., 0.42, 0.71]),  # RGB: 255,107,181
}

def convert_string_dict_to_type_array(d, types):
  """Converts a string dict to a type indexed array.
  Useful for creating type_color_map from a type_color_dict and 
  structure_decay_mats from structure_decay_dict.
  """
  idxs, vals = zip(*[(types[k], v) for k, v in d.items()])
  idxs = jp.array(idxs)
  vals = jp.array(vals)
  v_type = jp.array(vals[0]).dtype
  v0_shape = jp.array(vals[0]).shape
  res_shape = [len(types)]
  if v0_shape:
    res_shape.append(v0_shape[0])
  return jp.zeros(res_shape, dtype=v_type).at[idxs].set(vals)


class DefaultTypeDef(EnvTypeDef):
  """Default implementation of EnvTypeDef.
  
  This etd, with its default values, is used in the original Biomaker CA paper.
  
  If you are subclassing this ETD, remember to call super() during init, and
  override the relevant properties of mats, such as gravity_mats.
  """

  def __init__(
      self, materials=DEFAULT_MATERIALS,
      agent_types=DEFAULT_AGENT_TYPES,
      structure_decay_mats_dict=DEFAULT_STRUCTURE_DECAY_MATS_DICT,
      dissipation_rate_per_spec_dict=DEFAULT_DISSIPATION_RATE_PER_SPEC_DICT,
      type_color_dict=DEFAULT_TYPE_COLOR_DICT):
    """Initialization of DefaultTypeDef.
      
    Args:
      materials: List of strings of material types.
      agent_types: List of strings of agent types.
      structure_decay_mats_dict: dictionary of agent type strings and structural
        decay values.
      dissipation_rate_per_spec_dict: dictionary of agent type strings and 
        modifiers of the dissipation based on the agent specialization.
      type_color_dict: dictionary of agent type strings and rgb colors for 
        visualising them.
    """
    # initialize types, specialization_idxs, agent_types, materials_list
    super().__create_types__(materials, agent_types)
    types = self.types
    # setup material specific properties. If you are subclassing this, consider
    # changing these values manually.
    self.intangible_mats = jp.array([types.VOID, types.AIR], dtype=jp.int32)
    self.gravity_mats = jp.concatenate([
        jp.array([types.EARTH], dtype=jp.int32), self.agent_types], 0)
    self.structural_mats = self.agent_types
    self.propagate_structure_mats = jp.concatenate([jp.array([
        types.EARTH, types.IMMOVABLE], dtype=jp.int32), self.agent_types], 0)
    self.agent_spawnable_mats = jp.array([
        types.VOID, types.AIR, types.EARTH], dtype=jp.int32)
    self.structure_decay_mats = convert_string_dict_to_type_array(
        structure_decay_mats_dict, types)
    self.aging_mats = self.agent_types

    self.dissipation_rate_per_spec = convert_string_dict_to_type_array(
        dissipation_rate_per_spec_dict, self.specialization_idxs)

    self.type_color_map = convert_string_dict_to_type_array(
        type_color_dict, types)

    # Class abstraction checks for attributes.
    super().__post_init__()

### EnvConfig
# These are the configurations that make environments have different laws of
# 'physics'.
# Below, I added some 'default' values, so that one can create some configs and 
# environments without being overwhelmed with parameters to set.
# However, keep in mind that *several parameters are correlated!* 
# (To what? to an agent's survivalship.) Hence, modifying some parameters
# usually implies that you have to modify others too.


DEFAULT_STRUCT_INTEGRITY_CAP = 200

# how much each earth and air gives as material.
DEFAULT_ABSORBTION_AMOUNTS = jp.array([0.25, 0.25])
# How many nutrients are dissipated by agents per step.
DEFAULT_DISSIPATION_PER_STEP = jp.array([0.05, 0.05])

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
    etd: the EnvTypeDef defining all cell types and some of their properties.
    env_state_size: the total size of each cell. It is: 4 + agent_state_size
    struct_integrity_cap: the maximum value of structural integrity.
      This is what gets propagated by IMMUTABLE materials. Structural integrity
      decays the further away it goes from IMMUTABLE mats. If 0, gravity mats 
      fall.
    absorbtion_amounts: how many nutrients earth and air mats give to agents at
      every step. If there are not enough nutrients, only up to that amount is 
      given, distributed to however many ask for them.
    dissipation_per_step: how many nutrients are dissipated by agents per step.
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
               etd: EnvTypeDef = DefaultTypeDef(),
               struct_integrity_cap=DEFAULT_STRUCT_INTEGRITY_CAP,
               absorbtion_amounts=DEFAULT_ABSORBTION_AMOUNTS, 
               dissipation_per_step=DEFAULT_DISSIPATION_PER_STEP,
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
    self.etd = etd
    self.env_state_size = 4 + self.agent_state_size
    self.struct_integrity_cap = struct_integrity_cap
    self.absorbtion_amounts = absorbtion_amounts
    self.dissipation_per_step = dissipation_per_step
    self.spawn_cost = spawn_cost
    self.reproduce_cost = reproduce_cost
    self.specialize_cost = specialize_cost
    self.reproduce_min_dist = reproduce_min_dist
    self.reproduce_max_dist = reproduce_max_dist
    self.n_reproduce_per_step = n_reproduce_per_step
    self.nutrient_cap = nutrient_cap
    self.material_nutrient_cap = material_nutrient_cap
    self.max_lifetime = max_lifetime

  def __str__(self):
    return stringify_class(self)


### Helpers for making environments.

# namedtuple useful to define pairs of envs and configs.
if 'EnvAndConfig' not in globals():
  EnvAndConfig = namedtuple('EnvAndConfig', 'env config')



def add_agent_to_env(env, x, y, init_nutrients, aid, init_spec):
  """Add an agent to an environment in a specific (x,y) position."""
  return Environment(
      env.type_grid.at[x, y].set(init_spec),
      env.state_grid.at[x, y, EN_ST:EN_ST+2].set(init_nutrients),
      env.agent_id_grid.at[x, y].set(aid))


def place_seed(env: Environment, col, config: EnvConfig,
               row_optional=None, aid=0,
               custom_agent_init_nutrient=None):
  """Place a seed in an environment.
  
  Arguments:
    env: the environment to modify.
    col: the target column where to place the vertical seed.
    config: EnvConfig necessary for populating the env.
    row_optional: The optional specified row (positioned where the 'leaf' would
      usually be). If not specified, it will be inferred with extra computation.
    aid: the agent id of the new seed.
    custom_agent_init_nutrient: the value of nutrients for *each* cell. If not
      specified, a default value that tends to work well will be used. The
      default value is intended to be used when manually placing seeds. Use a
      custom value for when reproduction happens.
  """
  type_grid = env.type_grid
  etd = config.etd
  if row_optional is None:
    # find the row yourself. This code is somewhat redundant with
    # env_logic.find_fertile_soil.
    # create a mask that checks whether 'you are earth and above is air'.
    # the highest has index 0 (it is inverted).
    # the index starts from 1 (not 0) since the first row can never be fine.
    mask = ((type_grid[1:, col] == etd.types.EARTH) &
            (type_grid[:-1, col] == etd.types.AIR))
    # only get the highest value, if it exists.
    # note that these indexes now would return the position of the 'high-end' of 
    # the seed. That is, the idx is where the top cell (leaf) would be, and idx+1 
    # would be where the bottom (root) cell would be.
    row = (mask * jp.arange(mask.shape[0]+1, 1, -1)).argmax(axis=0)
    # note that we don't check if there is a valid row at all here.
    # for that, use the more expensive env_logic.find_fertile_soil or write
    # your own variant.
  else:
    row = row_optional
  if custom_agent_init_nutrient is None:
    agent_init_nutrient = (config.dissipation_per_step * 4 +
                           config.specialize_cost)
  else:
    agent_init_nutrient = custom_agent_init_nutrient

  # this needs to use dynamic_update_slice because it needs to be jittable.
  type_grid = jax.lax.dynamic_update_slice(
      type_grid,
      jp.full((2, 1), etd.types.AGENT_UNSPECIALIZED, dtype=jp.uint32),
      (row, col))
  new_states = jax.lax.dynamic_update_slice(
      jp.zeros([2, 1, config.env_state_size]),
      jp.repeat((agent_init_nutrient)[None, None,:], 2, axis=0),
      (0, 0, EN_ST))
  state_grid = jax.lax.dynamic_update_slice(
      env.state_grid, new_states, (row, col, 0))
  agent_id_grid = jax.lax.dynamic_update_slice(
      env.agent_id_grid, jp.repeat(aid, 2)[:, None].astype(jp.uint32),
      (row, col))
  return Environment(type_grid, state_grid, agent_id_grid)


def set_nutrients_to_materials(env, etd, earth_nut_val=None, air_nut_val=None):
  """Set the nutrient values of all EARTH and AIR cells to the respective vals.
  """
  assert earth_nut_val is not None or air_nut_val is not None, (
      "At least one nutrient value must be not None.")
  state_grid = env.state_grid
  if earth_nut_val is not None:
    is_earth = (env.type_grid == etd.types.EARTH).astype(jp.float32)
    state_grid = state_grid.at[:, :, EN_ST+EARTH_NUTRIENT_RPOS].set(
        is_earth * earth_nut_val +
        (1. - is_earth) * state_grid[:, :, EN_ST+EARTH_NUTRIENT_RPOS])
  if air_nut_val is not None:
    is_air = (env.type_grid == etd.types.AIR).astype(jp.float32)
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
                               init_nutrient_perc=0.2):
  """Create a simple default environment.
  It is filled with air, with immovable in the bottom and sun on top.
  If with_earth is True, it also contains earth covering the bottom half of the 
  environment.
  init_nutrient_perc defines the initialized nutrient values for earth and air,
  as a percentage of config.material_nutrient_cap
  """
  etd = config.etd
  env = create_enviroment_filled_with_type(config, h, w, etd.types.AIR)
  type_grid = env.type_grid.at[-1,:].set(etd.types.IMMOVABLE)
  type_grid = type_grid.at[0,:].set(etd.types.SUN)
  if with_earth:
    type_grid = type_grid.at[-h//2:-1, :].set(etd.types.EARTH)
  env = update_env_type_grid(env, type_grid)

  env = set_nutrients_to_materials(
      env, etd,
      init_nutrient_perc * config.material_nutrient_cap[EARTH_NUTRIENT_RPOS],
      init_nutrient_perc * config.material_nutrient_cap[AIR_NUTRIENT_RPOS])
  return env


def get_env_and_config(
    ec_id: str, width_type="wide", h=72, etd: EnvTypeDef = DefaultTypeDef()
    ) -> EnvAndConfig:
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
        etd=etd,
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
    env = create_default_environment(config, h, w)
    # add a seed at the center.
    env = place_seed(env, w // 2, config)

    return EnvAndConfig(env, config)
  
  if ec_id == "pestilence":
    w = infer_width(h, width_type)
    config = EnvConfig(
        etd=etd,
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
    env = create_default_environment(config, h, w)
    # add a seed at the center.
    env = place_seed(env, w // 2, config)

    return EnvAndConfig(env, config)
  
  if ec_id == "collaboration":
    w = infer_width(h, width_type)
    config = EnvConfig(
        etd=etd,
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
    env = create_default_environment(config, h, w)
    # add a seed at the center.
    env = place_seed(env, w // 2, config)

    return EnvAndConfig(env, config)
  
  if ec_id == "sideways":
    # This environment is recommended to be used with 'landscape' width.
    # this is because the nutrient generators are only present on opposite 
    # east-west directions.
    w = infer_width(h, width_type)
    config = EnvConfig(
        etd=etd,
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
    env = create_default_environment(config, h, w)
    # add a seed at the center.
    env = place_seed(env, w // 2, config)
    
    # now the kicker: remove nutrients from top and bottom.
    env = update_env_type_grid(
        env, env.type_grid.at[0,:].set(etd.types.AIR).at[-1, :].set(etd.types.EARTH))
    # place generators on the sides.
    for wi in range(10):
      env = update_env_type_grid(
          env, env.type_grid.at[wi,:20-wi*2].set(etd.types.SUN).at[
              -1-wi, -20+wi*2:].set(etd.types.IMMOVABLE))
  
    # fill the nutrients appropriately.
    env = set_nutrients_to_materials(
        env, etd,
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

@partial(jit, static_argnames=["config"])
def grab_image_from_env(env, config):
  """Create a visualization of the environment.
  
  Resulting values are floats ranging from [0,1].
  """
  etd = config.etd
  def map_cell(cell_type, state):
    env_c = etd.type_color_map[cell_type]
    # EARTH and AIR colors degrade by how little nutrients they have.
    is_earth_f = (cell_type == etd.types.EARTH).astype(jp.float32)
    is_air_f = (cell_type == etd.types.AIR).astype(jp.float32)
    env_c = env_c * (1. - is_earth_f) + env_c * is_earth_f * (0.3 +(
        state[EN_ST+EARTH_NUTRIENT_RPOS]/
        config.material_nutrient_cap[EARTH_NUTRIENT_RPOS])*0.7)
    env_c = env_c * (1. - is_air_f) + env_c * is_air_f * (0.3 +(
        state[EN_ST+AIR_NUTRIENT_RPOS]/
        config.material_nutrient_cap[AIR_NUTRIENT_RPOS])*0.7)
    return env_c
  return vmap2(map_cell)(env.type_grid, env.state_grid)
