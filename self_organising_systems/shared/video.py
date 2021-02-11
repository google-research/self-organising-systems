from self_organising_systems.shared.config import cfg
import tensorflow as tf
import numpy as np
import subprocess
import tempfile
import fcntl
import os
import sys

class VideoWriter:
  def __init__(self, out_path, fps=30, pp_fn=None, **kw):
    self.shape = None
    self.fps = fps
    self.out_path = out_path
    self.tempfilename = os.path.join(tempfile.mkdtemp(), 'selforg_out.mp4')

  def create_writer(self):
    command = [cfg.ffmpeg_path,
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '%ix%i' % self.shape, # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', str(self.fps), # frames per second
        '-i', '-', # The input comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-f', 'mp4',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        self.tempfilename]
    self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  def add(self, img):
    img = np.asarray(img)
    h, w = img.shape[:2]
    if self.shape is None:
      self.shape = (w, h) # (!) opposite order to numpy/imshow implemenation
      self.create_writer()
    if self.shape != (w, h):
      raise Exception("incorrectly sized frame")
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.pipe.stdin.write(img.tobytes())

  def close(self):
    # wait for ffmpeg to finish
    stdout, stderr = self.pipe.communicate()
    tf.io.gfile.copy(self.tempfilename, self.out_path, overwrite=True)
    tf.io.gfile.remove(self.tempfilename)

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
