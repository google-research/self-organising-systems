"""Module to use a version of IPython.display that uses display_id.
To be used inside a colab.

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

from IPython import get_ipython
from IPython.display import clear_output
from IPython.display import display as og_display
import ipywidgets as widgets

_output_handles = {}
_cell_to_invocation_id = {}

# lifted from IPython... guess this is sufficient to avoid collisions)
# and that collisions are sufficiently non-destructive as this is just UI/UX
def _new_id():
  """Generate a new random text id with urandom"""
  return b2a_hex(os.urandom(16)).decode('ascii')

def get_cell_id():
  return get_ipython().parent_header.get('metadata', {}).get('colab', {}).get('cell_id')

def get_invocation_id():
  return get_ipython().parent_header.get("msg_id")

class _DisplayAdapter:
  def __init__(self, out):
    self.out = out

  def update_display(self, obj):
    with self.out:
      clear_output(wait=True)
      og_display(obj)

def display(obj, **kwargs):

  if (get_cell_id() not in _output_handles) or (_cell_to_invocation_id.get(get_cell_id(), None) != get_invocation_id()):
    _output_handles[get_cell_id()] = {} 
    _cell_to_invocation_id[get_cell_id()] = get_invocation_id()
  
  outputs = _output_handles[get_cell_id()]

  if "display_id" in kwargs and kwargs["display_id"] is not None:
    if kwargs["display_id"] is True:
      kwargs["display_id"] = _new_id()   

    if kwargs["display_id"] not in outputs:
      outputs[kwargs["display_id"]] = widgets.Output()
      og_display(outputs[kwargs["display_id"]])

    out = outputs[kwargs["display_id"]]

    with out:
      clear_output(wait=True)
      og_display(obj, **kwargs)
      
    return _DisplayAdapter(out)
  else:
    og_display(obj, **kwargs)
    return None
