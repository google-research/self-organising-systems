"""
Converts exported .npy models to a single json.
"""
import numpy as np
from self_organising_systems.shared.util import im2url, tile2d

def export_models_to_js(models, fixed_filter_n=4):
  '''Exoprt numpy models in a form that ca.js can read.'''
  model_names = list(models.keys())
  models_js = {'model_names':model_names, 'layers': []}
  params = models.values()
  quant_scale_zero = [(2.0, 0.0), (4.0, 127.0 / 255.0)]
  for i, layer in enumerate(zip(*params)):
    shape = layer[0].shape
    layer = np.array(layer)  # shape: [n, h, w]
    if i == 0:
      # Replaced with np equiv. for time being so this works internally.
      # layer[:,:-1] = rearrange(layer[:,:-1], 'n (h c) w -> n (c h) w', c=fixed_filter_n)
      s = layer[:, :-1].shape
      layer[:, :-1] = (layer[:, :-1]
                       .reshape(s[0], -1, fixed_filter_n, s[2])
                       .transpose(0, 2, 1, 3)
                       .reshape(s))
    #layer = rearrange(layer, 'n h (w c) -> h (n w) c', c=4)
    # N.B. this 4 is not the fixed filter number, but a webgl implementation detail.
    # Pad when number of channels is not a multiple of 4.
    s = layer.shape
    layer = np.pad(layer, ((0,0), (0,0), (0, (4 - s[2]) % 4)), mode='constant')
    layer = layer.reshape(s[0], s[1], -1, 4)
    n, ht, wt = layer.shape[:3]
    w = 1
    while w<n and w*wt < (n+w-1)//w*ht:
      w += 1
    layer = tile2d(layer, w)
    layout = (w, (n+w-1)//w)

    scale = 2.0*np.abs(layer).max()
    layer = np.round(layer/scale*255.0+127.0)
    layer = np.uint8(layer.clip(0, 255))

    url = im2url(layer, 'png')
    layer_js = {'scale': scale,
                'data': url,
                'shape':shape,
                'quant_scale_zero': quant_scale_zero[i],
                'layout': layout}
    models_js['layers'].append(layer_js)
  return models_js


