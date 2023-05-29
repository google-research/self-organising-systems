"""Flexible Multi Head Attention module.

Attention: The first class is to high % copied from
https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py.

On top of the classic self-attention module, I want a stand alone self-attention
layer without the additional linear projection. This only makes sense for single
head attention module.
(Multi-Head) Attention module (plus some extra functionalities) for use in 
Transformer architectures.

"""
import dataclasses
import math
from typing import Optional
import warnings

import haiku as hk

import jax
import jax.numpy as jnp
import numpy as np


class TokenVocab(hk.Module):
  """Learnable Vocabulary with certain "token" size. These will be chosen 
  bases on a sequence of integers given to the system and function as input 
  to the Transformer.
  """

  def __init__(
      self,
      w_init: hk.initializers.Initializer,
      e_size: Optional[int] = 128,
      vocab_size: Optional[int] = 60000,
      name: Optional[str] = None,
  ):
    """Initialises the module.

    Args:
      w_init: Initialiser for weights in vocabulary.
      e_size: Dimension of each entry in the vocab.
      vocab_size: Size of vocabulary.
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.w_init = w_init
    self.e_size = e_size
    self.vocab_size = vocab_size

  def __call__(self, x, logits=False):
    vocab = hk.get_parameter("vocab", [self.vocab_size, 1, self.e_size], init=self.w_init)
    if logits:
      return jnp.einsum("...l,Vl->...V", x, jnp.squeeze(vocab))
    else:
      return jnp.take_along_axis(vocab, jnp.expand_dims(x, axis=-1), axis=0)


@dataclasses.dataclass
class MultiHeadAttention(hk.Module):
  """Multi-headed attention (MHA) module.

  This module is intended for attending over sequences of vectors.
  Rough sketch:
  - Compute keys (K), queries (Q), and values (V) as projections of inputs.
  - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
  - Output is another projection of WV^T.
  For more detail, see the original Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762.
  Glossary of shapes:
  - T: Sequence length.
  - D: Vector (embedding) size.
  - H: Number of attention heads.
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init_scale: Optional[float] = None,
      *,
      w_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      use_bias_p: Optional[bool] = False,
      use_softmax: Optional[bool] = False,
      use_non_lin_mix: Optional[bool] = False,
      sum_normalization: Optional[bool] = False,
      name: Optional[str] = None,
  ):
    """Initialises the module.

    Args:
      num_heads: Number of independent attention heads (H).
      key_size: The size of keys (K) and queries used for attention.
      w_init_scale: DEPRECATED. Please use w_init instead.
      w_init: Initialiser for weights in the linear map.
      value_size: Optional size of the value projection (V). If None, defaults
        to the key size (K).
      model_size: Optional size of the output embedding (D'). If None, defaults
        to the key size multiplied by the number of heads (K * H).
      use_bias_p: Use bias parameters in the linear operations of the network.
      use_softmax: Use softmax instead of linear Transformer.
      use_non_lin_mix: Use softmax-linear mix Transformer.
      sum_normalization: Use sum normalization for the linear Transformer.
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads
    self.use_bias_p = use_bias_p
    self.use_softmax = use_softmax
    self.use_non_lin_mix = use_non_lin_mix
    self.sum_normalization = sum_normalization

    # Backwards-compatibility for w_init_scale.
    if w_init_scale is not None:
      warnings.warn(
          "w_init_scale is deprecated; please pass an explicit weight "
          "initialiser instead.", DeprecationWarning)
    if w_init and w_init_scale:
      raise ValueError("Please provide only `w_init`, not `w_init_scale`.")
    if w_init is None and w_init_scale is None:
      raise ValueError("Please provide a weight initializer: `w_init`.")
    if w_init is None:
      w_init = hk.initializers.VarianceScaling(w_init_scale)
    self.w_init = w_init

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Computes (optionally masked) MHA with queries, keys & values.

    This module broadcasts over zero or more 'batch-like' leading dimensions.
    Args:
      query: Embeddings sequence used to compute queries; shape [..., T', D_q].
      key: Embeddings sequence used to compute keys; shape [..., T, D_k].
      value: Embeddings sequence used to compute values; shape [..., T, D_v].
      mask: Optional mask applied to attention weights; shape [..., H=1, T', T].
    Returns:
      A new sequence of embeddings, consisting of a projection of the
        attention-weighted value projections; shape [..., T', D'].
    """

    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values
    query_heads = projection(query, self.key_size, self.use_bias_p, "query")
    key_heads = projection(key, self.key_size, self.use_bias_p, "key")
    if self.sum_normalization:
      query_heads=query_heads/(jnp.sum(query_heads,axis=-1,keepdims=True)+ 1e-6)
      key_heads = key_heads/(jnp.sum(key_heads, axis=-1)[..., None] + 1e-6)
    value_heads = projection(value, self.value_size, self.use_bias_p, "value")

    # Compute attention weights.

    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    if mask is not None:
      if mask.ndim != attn_logits.ndim:
        raise ValueError(
            f"Mask dimensionality {mask.ndim} must match logits dimensionality "
            f"{attn_logits.ndim}."
        )
      attn_logits = jnp.where(mask, attn_logits, -1e30)

     # [H, T', T]e
    if self.use_softmax:
      attn_weights = jax.nn.softmax(attn_logits/
                                    np.sqrt(self.key_size).astype(key.dtype))
    elif self.use_non_lin_mix:
      y = hk.Linear(1, with_bias=False,
                    w_init=self.w_init, name='non_lin_mix')(jnp.array([1.0])) 
      attn_weights = ((jax.nn.softmax(attn_logits/
                                    np.sqrt(self.key_size).astype(key.dtype)))
                      #*jnp.clip(y + 0.5, 0, 1) +
                      #(1-jnp.clip(y + 0.5, 0, 1))*attn_logits)
                      *jax.nn.sigmoid(y*10) +
                      (1-jax.nn.sigmoid(y*10))*attn_logits)
    else:
      attn_weights = attn_logits

    # Weight the values by the attention and flatten the head vectors.
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]
    # Apply another projection to get the final embeddings 
    final_projection = hk.Linear(self.model_size, w_init=self.w_init,
                                 with_bias=self.use_bias_p)
    attn = final_projection(attn)
    return attn, attn_weights  # [T', D']

  @hk.transparent
  def _linear_projection(
      self,
      x: jnp.ndarray,
      head_size: int,
      with_bias: Optional[bool] = False,
      name: Optional[str] = None,
  ) -> jnp.ndarray:
    y = hk.Linear(self.num_heads * head_size, with_bias=with_bias,
                  w_init=self.w_init, name=name)(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))


@dataclasses.dataclass
class MLP(hk.Module):
  """A multi layer perceptron.

  This module is fully connexted neural network, intented to process the
  result of the self-attention module. A couple of classic design choices
  have been already made such as using the gelu non-linearity,
  https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html, as well
  as fixing the depth to 2. Since the depth of the MLP is not part of our
  analyses (for now) we do not allow for this flexiblity.
  """

  def __init__(
      self,
      w_init: hk.initializers.Initializer,
      widening_factor: int = 4,
      second_layer: bool = False,
      use_bias_p: bool = False,
      outputdim: int = 0,
      name: Optional[str] = None
  ):
    """Initialises the module.

    Args:
      w_init: Initialiser for weights in the linear maps.
      widening_factor: Blow up in the hidden layer compared to input dimension.
      use_bias_p: Use pias parameters in linear layers.
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.w_init = w_init
    self.widening_factor = widening_factor
    self.second_layer = second_layer
    self.use_bias_p = use_bias_p
    self.outputdim = outputdim

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    x = hk.Linear(self.widening_factor * hiddens, with_bias=self.use_bias_p,
                  w_init=self.w_init)(x)
    x = jax.nn.gelu(x)
    if self.second_layer:
      x = hk.Linear(self.widening_factor * hiddens, with_bias=self.use_bias_p,
                    w_init=self.w_init)(x)
      x = jax.nn.gelu(x)
    if self.outputdim == 0:
      return hk.Linear(hiddens, with_bias=self.use_bias_p,
                       w_init=self.w_init)(x)
    else:
      return hk.Linear(self.outputdim, with_bias=self.use_bias_p,
                       w_init=self.w_init)(x)

@dataclasses.dataclass
class LNorm(hk.Module):
  """A layer norm class.
  """

  def __init__(
      self,
      name: Optional[str] = None
  ):
    """Initialises the module.

    Args:
      name: Optional name for this module.
    """
    super().__init__(name=name)

  def __call__(self, x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """Apply a LayerNorm operation to x with default settings."""
  return hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,
                      name=name)(x)


def create_pos_encoding(context_size, input_size, flip=False):
  """Create constant positional encoding."""
  pe = np.zeros((context_size, input_size))
  position = np.arange(0, context_size, dtype=np.float32)[:, None]
  div_term = np.exp(np.arange(0, input_size, 2) *
                    (-math.log(10000.0)/input_size))
  pe[:, 0::2] = np.sin(position * div_term)
  pe[:, 1::2] = np.cos(position * div_term)
  pe = pe[None]
  if flip:
    return jnp.flip(jax.numpy.squeeze(jax.device_put(pe), axis=0), 0)
  else:
    return jax.numpy.squeeze(jax.device_put(pe), axis=0)

def create_pos_encoding_diff(context_size, input_size):
  """Create constant positional encoding."""
  pe = np.zeros((context_size, input_size))
  position = np.arange(0, context_size, dtype=np.float32)[:, None]
  twoi = np.arange(0, input_size, 2)
  #div_term = np.exp(twoi * (-math.log(10000.0)/input_size))
  pe[:, 0::2] = np.sin(position / (10000**(twoi/input_size)))
  pe[:, 1::2] = np.cos(position / (10000**(twoi/input_size)))
  pe = pe[None]
  return jax.numpy.squeeze(jax.device_put(pe), axis=0)
  