import ml_collections

cfg = ml_collections.ConfigDict()
cfg.logdir = "/tmp/ca" # base log dir
cfg.experiment_name = "0" # directory under log_dir to place output.
cfg.ffmpeg_path = "ffmpeg"

try:
  __IPYTHON__
  cfg.is_ipython = True
except NameError:
  cfg.is_ipython = False
