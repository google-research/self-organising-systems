""" Useful functions for running in colab.
"""

import PIL.Image, PIL.ImageDraw
from IPython.display import Image, HTML, clear_output
from self_organising_systems.shared.util import imencode

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))
