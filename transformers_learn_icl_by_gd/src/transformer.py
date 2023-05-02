"""Fleixble Transformer model.

The multi-head-attention class is to some degreee copied from
https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py.

This code makes heavy use of Haiku but enables ablation studies on architecture
choices such as including linear projections, skip connections, normalization
layer. The aim is to interpolate between stripped down linear and the classic 
transformer architecture.
"""

import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from src.attn import (MLP,
                      MultiHeadAttention,
                      TokenVocab,
                      create_pos_encoding,
                      LNorm,
                      layer_norm)


@dataclasses.dataclass
class Transformer(hk.Module):
  """A flexible Transformer implementation.
  """

  def __init__(
      self,
      num_heads: int = 2,
      widening_factor: int = 4,
      num_layers: int = 3,
      key_size: int = 5,
      embedding_size: int = 64,
      output_size: int = 1,
      in_context_length: int = 17,
      in_context_length_test: int = 17,
      test_points: int = 1,
      dropout_rate: float = 0,
      only_attention: bool = True,
      use_layer_norm: bool = True,
      use_pe: bool = True,
      pe_size: int = 6,
      concat_pe: bool = False,
      output_mapping: bool = False,
      input_mapping: bool = False,
      use_bias_p: bool = True,
      zero_embeddings: bool = False,
      deq: bool = True,
      init_scale: float = 0.02,
      use_softmax: bool = False,
      use_non_lin_mix: bool = False,
      first_layer_sm: bool = False,
      y_update: bool = False,
      input_mlp: bool = False,
      input_mlp_out_dim: int = 0,
      gd_mlp_config: bool = False,
      sum_norm: bool = False,
      dampening: float = 1.0,
      clip: float = 0.0,
      ana_copy: bool = False,
      flip: bool = False,
      vocab_size: int = 0,
      vocab_token_dim: int = 0,
      vocab_init: int = 0.01,
      return_logits: bool = False,
      include_query: bool = False,
      name: Optional[str] = None,
  ):


    """Initialises the module.

    Args:
      num_heads: Number of heads in the self-attention module.
      widening_factor: Blow up in the hidden layer of MLP.
      num_layers: Number of transformer layers, usually one due DEQ behaviour.
      key_size: Key and querie size.
      embedding_size: Embedding size.
      output_size: Output size.
      in_context_length: Sequence length.
      test_points: Number of test points.
      dropout_rate: Optional dropout layer with rate dropout_rate if not None.
      only_attention: Only the attention layer without the MLP.
      use_layer_norm: Use layer norm or not.
      use_pe: Use positional encoding. 
      pe_size: Positional encoding size.
      concat_pe: Concat pe.
      output_mapping: Use output mapping.
      input_mapping: Use input mapping.
      lin_proj_after_att: Linear projection after attention layer.
      use_bias_p: Use bias parameter in the linear operations in the network.
      zero_embeddings: Use zero embeddings.
      full_state_update: Update context tokens or only querry.
      deq: Use recurrent transformer.
      y_update: Update only output states e.g. as in gradient descent.
      input_mlp: Use MLP instead of linear embedding.
      input_mlp_out_dim: Output dim of input MLP.
      gd_mlp_config: Gradient descent special MLP config.
      sum_norm: Use sum normalization from Schlag et. al 2012
      dampening: Dampen forward dynamics
      clip: Clip the activations to some value
      ana_copy: Return full prediction stack instead of last entry.
      include_query: Include query vector in computation.
      name : Optional name for this module.
    """

    super().__init__(name=name)
    self.num_heads = num_heads
    self.widening_factor = widening_factor
    self.num_layers = num_layers
    self.key_size = key_size
    self.dropout_rate = dropout_rate
    self.only_attention = only_attention
    self.use_layer_norm = use_layer_norm
    self.use_pe = use_pe
    self.pe_size = pe_size
    self.concat_pe = concat_pe
    self.output_mapping = output_mapping
    self.input_mapping = input_mapping
    self.use_bias_p = use_bias_p
    self.embedding_size = embedding_size
    self.output_size = output_size
    self.in_context_length = in_context_length
    self.in_context_length_test = in_context_length_test
    self.zero_embeddings = zero_embeddings
    self.init_scale = init_scale
    self.use_softmax = use_softmax
    self.use_non_lin_mix = use_non_lin_mix
    self.first_layer_sm = first_layer_sm
    self.deq = deq
    self.y_update = y_update
    self.input_mlp = input_mlp
    self.input_mlp_out_dim = input_mlp_out_dim
    self.gd_mlp_config = gd_mlp_config
    self.sum_norm = sum_norm
    self.dampening = dampening
    self.clip = clip
    self.ana_copy = ana_copy
    self.vocab_size = vocab_size
    self.vocab_token_dim = vocab_token_dim
    self.vocab_init = vocab_init
    self.return_logits = return_logits
    self.include_query = include_query

    if pe_size > 0:
      self.pos_encoding = create_pos_encoding(in_context_length, pe_size, flip)
      self.pos_encoding_test = create_pos_encoding(in_context_length_test,
                                                   pe_size, flip)
    else:
      self.pos_encoding = None

  def trans_block(self, h, nl):
    # First the attention block.

    if self.deq:
      h_norm = self.lnorm1(h) if self.use_layer_norm else h
      if not self.include_query:
        key = h_norm[:, :-1, :]
        value = h_norm[:, :-1, :]
      else:
        key = h_norm
        value = h_norm

      h_attn, att_map =self.attn_block(h_norm,key,value)
    else:
      if nl == 0:
        h_norm = h
      else:
        h_norm = layer_norm(h, name="norm_"+str(nl)) if self.use_layer_norm else h

      sm = self.use_softmax or (self.first_layer_sm and nl == 0)
      mix = self.use_non_lin_mix and nl == 0
      attn_block = MultiHeadAttention(num_heads=self.num_heads,
                                      key_size=self.key_size,
                                      model_size=self.model_size,
                                      w_init=self.w_init,
                                      use_softmax=sm,
                                      use_non_lin_mix=mix,
                                      use_bias_p=self.use_bias_p,
                                      sum_normalization=self.sum_norm,
                                      name="layer_"+str(nl))
      if not self.include_query:
        key = h_norm[:, :-1, :]
        value = h_norm[:, :-1, :]
      else:
        key = h_norm
        value = h_norm

      h_attn, att_map = attn_block(h_norm, key, value)
    h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)

    if self.y_update:
      h = h.at[:, :, -1].set(h[:, :, -1] + self.dampening*h_attn[:, :, -1])
    else:
      h = h + self.dampening*h_attn

    if self.clip > 0:
      h = jnp.clip(h, -self.clip, self.clip)

    if not self.only_attention:
      if self.deq:
        h_inter_norm = self.lnorm2(h) if self.use_layer_norm else h
        h_dense = self.dense_block(h_inter_norm)
      else:
        h_inter_norm = layer_norm(h) if self.use_layer_norm else h
        dense_block = MLP(w_init=self.w_init,
                          widening_factor=self.widening_factor,
                          use_bias_p=self.use_bias_p)
        h_dense = dense_block(h_inter_norm)

      h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
      h = h + self.dampening*h_dense

      if self.clip > 0:
        h = jnp.clip(h, -self.clip, self.clip)
    return h, att_map

  def __call__(
      self,
      x: jnp.ndarray,
      is_training: bool,
      predict_test: bool
  ) -> jnp.ndarray:

    """Computes the transformer forward pass.

    Args:
      x: Inputs.
      is_training: Whether we're training or not.
      predict_test: Test or train prediction.
    Returns:
      Array of shape [B, T, H].
    """

    if self.vocab_size > 0 and self.vocab_token_dim > 0:
      self.w_init_vocab = hk.initializers.VarianceScaling(self.vocab_init)
      vocab = TokenVocab(w_init=self.w_init_vocab,
                         e_size=self.vocab_token_dim,
                         vocab_size=self.vocab_size)
      x = vocab(x)

    self.w_init = hk.initializers.VarianceScaling(self.init_scale)
    self.dropout_rate = self.dropout_rate if is_training else 0.

    if self.input_mapping:
      embeddings = hk.Linear(self.embedding_size,
                             with_bias=self.use_bias_p,
                             w_init=self.w_init, name="emb")(x)
    else:
      embeddings = x

    if self.input_mlp:
      input_mlp = MLP(w_init=self.w_init, widening_factor=self.widening_factor,
                      second_layer=False, use_bias_p=True,
                      outputdim=self.input_mlp_out_dim, name="input_mlp")

      embeddings = embeddings + input_mlp(embeddings)

    # add / concat positional encodings to each member in batch
    if self.use_pe:
      if self.concat_pe:
        if predict_test:
          pos_encoding_test = self.pos_encoding_test[None, ...]
          pos_encoding_test = jnp.repeat(pos_encoding_test, embeddings.shape[0],
                                         axis=0)
          pos_encoding_test = pos_encoding_test*0 if self.zero_embeddings else pos_encoding_test
          h = jnp.concatenate([embeddings, pos_encoding_test], axis=2)
        else:
          pos_encoding = self.pos_encoding[None, ...]
          pos_encoding = jnp.repeat(pos_encoding, embeddings.shape[0], axis=0)
          pos_encoding = pos_encoding*0 if self.zero_embeddings else pos_encoding
          h = jnp.concatenate([embeddings, pos_encoding], axis=2)
      else:
        if predict_test:
          h = self.pos_encoding_test + embeddings
        else:
          h = self.pos_encoding + embeddings
    else:
      h = embeddings

    if len(h.shape) == 2:
      _, model_size = h.shape
    elif len(h.shape) == 3:
      _, _, model_size = h.shape
    self.model_size = model_size
    if self.deq:
      self.attn_block = MultiHeadAttention(num_heads=self.num_heads,
                                           key_size=self.key_size,
                                           model_size=model_size,
                                           w_init=self.w_init,
                                           use_softmax=self.use_softmax,
                                           use_non_lin_mix=self.use_non_lin_mix,
                                           use_bias_p=self.use_bias_p,
                                           sum_normalization=self.sum_norm
                                           )
      if not self.only_attention:
        self.dense_block = MLP(w_init=self.w_init,
                               widening_factor=self.widening_factor,
                               use_bias_p=self.use_bias_p)

      if self.use_layer_norm:
        self.lnorm1 = LNorm()
        self.lnorm2 = LNorm()

    st = h[:, -1, -1]*(-1.0) if not self.ana_copy else (h if self.include_query else h[:, :-1, :])
    stack_h = [] if not self.input_mlp else [st]
    stack_att = []
    for nl in range(self.num_layers):
      h, att_map = self.trans_block(h, nl)
      # intermediate readout of test prediction
      st = h[:, -1, -1]*(-1.0) if not self.ana_copy else (h if self.include_query else h[:, :-1, :])
      stack_h.append(st)
      stack_att.append(att_map)
    out = hk.Linear(self.output_size)(h) if self.output_mapping else h

    if self.return_logits:
      out = vocab(out, logits=True)
    return(out, stack_h, stack_att)
