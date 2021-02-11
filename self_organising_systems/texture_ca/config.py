from self_organising_systems.shared.config import cfg
import ml_collections

cfg.texture_ca = ml_collections.ConfigDict()
cfg.texture_ca.channel_n = 12
cfg.texture_ca.hidden_n = 96
cfg.texture_ca.fire_rate = 0.5
cfg.texture_ca.batch_size = 4
cfg.texture_ca.lr = 2e-3
cfg.texture_ca.pool_size = 1024
cfg.texture_ca.fixed_seed = 123 # 0 to disable
cfg.texture_ca.lr = 2e-3
cfg.texture_ca.lr_decay = 2000
cfg.texture_ca.rollout_len_min = 32
cfg.texture_ca.rollout_len_max = 64
cfg.texture_ca.train_steps = 2000
cfg.texture_ca.gradnorm = True
cfg.texture_ca.q = 2.0
cfg.texture_ca.bias = True
cfg.texture_ca.learned_filters = 0
cfg.texture_ca.laplacian = True
cfg.texture_ca.gradient = True
cfg.texture_ca.identity = True

# texture synth / style transfer
cfg.texture_ca.ancestor_npy = ''
cfg.texture_ca.img_size = 128
cfg.texture_ca.vgg_input_img_size = 128
cfg.texture_ca.texture_dir = 'textures'
cfg.texture_ca.ancestor_dir = 'models'
cfg.texture_ca.objective = "style:mondrian.jpg" #{style:mondrian.jpg, inception:mixed4b_pool_reduce_pre_relu:30}
cfg.texture_ca.inception_pb = 'gs://modelzoo/vision/other_models/InceptionV1.pb'
cfg.texture_ca.hidden_viz_group = False # Group the hidden states into RGB when vizualizing
cfg.texture_ca.viz_rollout_len = 1000
cfg.texture_ca.overflow_loss_coef = 1e4 # auxiliary loss to keep generated values in [0,1]
