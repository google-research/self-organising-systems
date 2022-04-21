from self_organising_systems.texture_ca.config import cfg
from self_organising_systems.shared.util import imread
import tensorflow as tf
import numpy as np


style_layers = ['block%d_conv1'%i for i in range(1, 6)]
content_layer = 'block4_conv2'


class StyleModel:
  def __init__(self, input_texture_path):
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    layers = style_layers + [content_layer]
    layers = {name:vgg.get_layer(name).output for name in layers}
    self.model = tf.keras.Model([vgg.input], layers)
    self.style_img = imread(input_texture_path, cfg.texture_ca.vgg_input_img_size)
    self.target_style, _ = self.calc_style_content(self.style_img[None,...])

  def run_model(self, img):
    img = img[..., ::-1]*255.0 - np.float32([103.939, 116.779, 123.68])
    layers = self.model(img)
    style = [layers[name] for name in style_layers]
    return style, layers[content_layer]

  def calc_style_content(self, img):
    style_layers, content = self.run_model(img)
    style = [self.gram_style(a) for a in style_layers]
    return style, content

  @tf.function
  def __call__(self, x):
    gs, content = self.calc_style_content(x)
    sl = tf.reduce_mean(self.style_loss(gs, self.target_style))
    return sl

  @tf.function
  def style_loss(self, a, b):
    return tf.add_n([tf.reduce_mean(tf.square(x-y), [-2, -1]) for x, y in zip(a, b)])

  def gram_style(self, a):
    n, h, w, ch = tf.unstack(tf.shape(a))
    a = tf.sqrt(a+1.0)-1.0
    gram = tf.einsum('bhwc, bhwd -> bcd', a, a)
    return gram / tf.cast(h*w, tf.float32)

class Inception:
  def __init__(self, layer, ch):
    with tf.io.gfile.GFile(cfg.texture_ca.inception_pb, 'rb') as f:
      self.graph_def = tf.compat.v1.GraphDef.FromString(f.read())
    self.layer = layer
    self.ch = ch
    avgpool0_idx = [n.name for n in self.graph_def.node].index('avgpool0')
    del self.graph_def.node[avgpool0_idx:]
    # use pre_relu layers for Concat nodes
    node = {n.name:n for n in self.graph_def.node}[layer]
    self.outputs = [layer+':0']
    if 'Concat' in node.op:
      self.outputs = [inp+'_pre_relu:0' for inp in node.input[1:]]
  
  @tf.function
  def __call__(self, x):
    overflow_loss = tf.reduce_mean(tf.square(tf.clip_by_value(x, 0.0, 1.0)-x))
    imgs = x*255.0-117.0
    outputs = tf.import_graph_def(self.graph_def, {'input':imgs}, self.outputs)
    a = tf.concat(outputs, -1)
    return -tf.reduce_mean(a[...,self.ch]) + overflow_loss*cfg.texture_ca.overflow_loss_coef
