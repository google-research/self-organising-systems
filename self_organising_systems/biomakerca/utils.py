"""
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
from jax import vmap
import jax.random as jr

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


vmap2 = lambda f: vmap(vmap(f))


def split_2d(key, w, h):
  return vmap(lambda k: jr.split(k, h))(jr.split(key, w))


def conditional_update(arr, idx, val, cond):
  """Update arr[idx] to val if cond is True."""
  return arr.at[idx].set((1 - cond) * arr[idx] + cond * val)

