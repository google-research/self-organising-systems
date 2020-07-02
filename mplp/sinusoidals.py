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

import numpy as np
import tensorflow.compat.v2 as tf

class SinusoidalsDS():

  def __init__(self):
    None

  def _create_task(self):
    A = np.random.uniform(low=.1, high=.5)
    ph = np.random.uniform(low=0., high=np.pi)
    return A, ph

  def _create_instance(self, A, ph, inner_batch_size, num_steps):
    x = np.random.uniform(
        low=-5., high=5., size=(num_steps, inner_batch_size, 1)).astype(
            np.float32)
    y = A * np.sin(x + ph)
    return x, y


  def _generator(self, inner_batch_size, num_steps):
    while True:
      A, ph = self._create_task()
      xt, yt = self._create_instance(A, ph, inner_batch_size, num_steps)
      xe, ye = self._create_instance(A, ph, inner_batch_size, num_steps)

      yield xt, yt, xe, ye

  def create_ds(self, outer_batch_size, inner_batch_size, num_steps):
    return tf.data.Dataset.from_generator(
            lambda:self._generator(inner_batch_size, num_steps),
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
        ).batch(outer_batch_size)
