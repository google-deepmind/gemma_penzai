# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Gemma architecture transformer variant with multimodal support.

Supports Gemma 3 multimodal transformer. Based on the
Flax reference implementation at https://github.com/google-deepmind/gemma.

This file defines the following classes:

  * MultiModalTransformerMetadata: Common axis sizes and other information for
    multi-modal transformer models.
  * MultiModalTransformerLM: Top-level multi-modal transformer wrapper.

And the following layers:

  * ApplyMultiModalCausalAttentionMask: Builds and applies a causal attention
    mask for multimodal transformer.
  * ApplyMultiModalCausalSlidingWindowAttentionMask: Builds and applies a
    sliding-window attention mask for multimodal transformer.
  * StitchEmbeddings: Stitches embeddings from text and vision.
  * TruncateVisionLogits: Truncates the vision logits to the original text
    logits.

And the model definitions for Gemma 3 with multimodal support.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Literal

from gemma_penzai.mllm.models import gemma_utils
from gemma_penzai.vision import siglip
from gemma_penzai.vision import vision_layers
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.core import named_axes
from penzai.core import struct
from penzai.models.transformer import model_parts
from penzai.models.transformer.variants import llamalike_common


AttentionType = llamalike_common.AttentionType
AttentionTypeSlidingWindowCausal = (
    llamalike_common.AttentionTypeSlidingWindowCausal
)
AttentionTypeGlobalCausal = llamalike_common.AttentionTypeGlobalCausal

build_mm_tokens = gemma_utils.build_mm_tokens
build_attention_mask = gemma_utils.build_attention_mask
build_token_positions = gemma_utils.build_token_positions
SOFT_TOKEN_PLACEHOLDER = gemma_utils.SOFT_TOKEN_PLACEHOLDER
START_OF_IMAGE = gemma_utils.START_OF_IMAGE
END_OF_IMAGE = gemma_utils.END_OF_IMAGE


@struct.pytree_dataclass
class ApplyMultiModalCausalAttentionMask(pz.nn.Layer):
  """Builds and applies a causal attention mask for multimodal transformer.

  The text part uses causal attention, while the image part uses
  bidirectional attention.

  If images are fed into the model, an attention mask (boolean named array) will
  be as input, then this class only applies the mask.

  If images are not fed into the model, the attention mask is built from the
  token positions.

  Masked out values are replaced with the ``masked_out_value`` attribute, which
  is usually a large (but finite) negative value.

  Attributes:
    query_positions_input_name: Key in the side input dictionary to use to
      identify the query token positions, which should be an integer array with
      the `seq_axis` axis.
    kv_positions_input_name: Key in the side input dictionary to use to identify
      the key/value token positions, which should be an integer array the
      `seq_axis` axis. (This axis will be renamed to match `kv_seq_axis`.)
    seq_axis: Name of the sequence axis, which should be present in both the
      query and key/value token position side inputs.
    kv_seq_axis: Name of the key/value sequence axis, which represents the keys
      and values in the input logits array.
  """

  masked_out_value: jax.typing.ArrayLike
  query_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  kv_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  seq_axis: str = dataclasses.field(
      default="seq", metadata={"pytree_node": False}
  )
  kv_seq_axis: str = dataclasses.field(
      default="kv_seq", metadata={"pytree_node": False}
  )
  attention_mask_input_name: str = dataclasses.field(
      default="attention_mask", metadata={"pytree_node": False}
  )

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Applies the attention mask to the input array.

    Args:
      x: The input array to mask. Usually the matrix of query-key dot products.
      **side_inputs: Side inputs. Must include ``query_positions_input_name``
        and ``kv_positions_input_name``.

    Returns:
      An adjusted matrix of logits, where any value where the mask is False has
      been replaced with the `masked_out_value` argument.
    """
    if self.attention_mask_input_name in side_inputs:
      mask = side_inputs[self.attention_mask_input_name]
      if mask is not None:
        return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)
    # if no mask is provided, we create one from the token positions
    query_positions = side_inputs[self.query_positions_input_name]
    kv_positions = (
        side_inputs[self.kv_positions_input_name]
        .untag(self.seq_axis)
        .tag(self.kv_seq_axis)
    )
    mask = (query_positions >= kv_positions) & (kv_positions >= 0)
    return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)


@struct.pytree_dataclass
class ApplyMultiModalCausalSlidingWindowAttentionMask(pz.nn.Layer):
  """Builds and applies a sliding-window attention mask for multimodal transformer.

  The text part uses causal sliding-window attention, while the image part uses
  bidirectional sliding-window attention.

  If images are fed into the model, an attention mask (boolean named array) will
  be as input, then this class creates the sliding window mask and only applies
  both sliding window mask and the attention mask.

  If images are not fed into the model, the attention mask is built from the
  token positions and window size.

  Masked out values are replaced with the ``masked_out_value`` attribute, which
  is usually a large (but finite) negative value.

  Attributes:
    masked_out_value: The value to substitute for masked-out locations.
    sliding_window_size: The size of the sliding window.
    query_positions_input_name: Key in the side input dictionary to use to
      identify the query token positions, which should be an integer array with
      the `seq_axis` axis.
    kv_positions_input_name: Key in the side input dictionary to use to identify
      the key/value token positions, which should be an integer array the
      `seq_axis` axis. (This axis will be renamed to match `kv_seq_axis`.)
    seq_axis: Name of the sequence axis, which should be present in both the
      query and key/value token position side inputs.
    kv_seq_axis: Name of the key/value sequence axis, which represents the keys
      and values in the input logits array.
  """

  masked_out_value: jax.typing.ArrayLike
  sliding_window_size: int | jax.typing.ArrayLike
  query_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  kv_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  seq_axis: str = dataclasses.field(
      default="seq", metadata={"pytree_node": False}
  )
  kv_seq_axis: str = dataclasses.field(
      default="kv_seq", metadata={"pytree_node": False}
  )
  attention_mask_input_name: str = dataclasses.field(
      default="attention_mask", metadata={"pytree_node": False}
  )

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Applies the attention mask to the input array.

    Args:
      x: The input array to mask. Usually the matrix of query-key dot products.
      **side_inputs: Side inputs. Must include ``query_positions_input_name``
        and ``kv_positions_input_name``.

    Returns:
      An adjusted matrix of logits, where any value where the mask is False has
      been replaced with the `masked_out_value` argument.
    """
    query_positions = side_inputs[self.query_positions_input_name]
    kv_positions = (
        side_inputs[self.kv_positions_input_name]
        .untag(self.seq_axis)
        .tag(self.kv_seq_axis)
    )
    if self.attention_mask_input_name in side_inputs:
      mask = side_inputs[self.attention_mask_input_name]
      if mask is not None:
        slide_window_mask = (
            (query_positions - kv_positions < self.sliding_window_size)
            & (query_positions - kv_positions > -self.sliding_window_size)
            & (kv_positions >= 0)
        )
        mask = mask & slide_window_mask
        return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)
    # if no mask is provided, we create one from the token positions
    mask = (
        (query_positions >= kv_positions)
        & (query_positions - kv_positions < self.sliding_window_size)
        & (kv_positions >= 0)
    )
    return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)


@struct.pytree_dataclass
class StitchEmbeddings(pz.nn.Layer):
  """Stitches embeddings from text and vision."""

  vision_input_name: str = dataclasses.field(
      default="images", metadata={"pytree_node": False}
  )
  mask_input_name: str = dataclasses.field(
      default="images_mask", metadata={"pytree_node": False}
  )

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Stitches text embeddings with the vision embeddings.

    Args:
      x: The input array to stitch. Usually the tokens embeddings.
      **side_inputs: Side inputs. Must include ``vision_input_name`` and
        ``token_input_name``.

    Returns:
      The stitched array with the vision embeddings inserted.
    """
    vision_embeddings = side_inputs[self.vision_input_name]
    if vision_embeddings is None:
      return x

    mask = side_inputs[self.mask_input_name]
    target_pos = named_axes.nmap(jnp.nonzero)(
        mask.untag("seq"),
        size=vision_embeddings.named_shape["seq"],
    )
    first_pos = x.untag("seq")[0]
    merged_embeddings = (
        x.untag("seq").at[target_pos].set(vision_embeddings.untag("seq"))
    )
    merged_embeddings = merged_embeddings.at[0].set(first_pos)
    merged_embeddings = merged_embeddings.tag("seq")
    return merged_embeddings


@struct.pytree_dataclass
class TruncateVisionLogits(pz.nn.Layer):
  """Truncates the vision logits to the original text logits."""

  vision_input_name: str = dataclasses.field(
      default="images", metadata={"pytree_node": False}
  )
  mask_input_name: str = dataclasses.field(
      default="images_start_mask", metadata={"pytree_node": False}
  )
  num_tokens_per_image: int = dataclasses.field(
      default=256, metadata={"pytree_node": False}
  )

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Truncates the vision logits to the original text logits.

    Args:
      x: The input array to truncate. Usually the logits with MM
      **side_inputs: Side inputs. Must include ``vision_input_name``.

    Returns:
      The truncated array with the vision logits removed.
    """
    vision_embeddings = side_inputs[self.vision_input_name]
    mask = side_inputs[self.mask_input_name]
    if vision_embeddings is None:
      return x

    # Shift the original tokens, to recover the original position.
    offset_by = self.num_tokens_per_image + 3
    offset = named_axes.nmap(jnp.cumsum)(mask.untag("seq"), axis=0) * offset_by
    offset = offset.tag("seq")
    new_positions = named_axes.wrap(jnp.arange(mask.named_shape["seq"])).tag(
        "seq"
    )
    new_positions = new_positions + offset - offset_by * mask
    logits = named_axes.nmap(jnp.take_along_axis)(
        x.untag("seq"),
        new_positions.untag("seq"),
        axis=0,
    )
    logits = logits.tag("seq")
    return logits


@dataclasses.dataclass
class MultiModalTransformerMetadata:
  """Common axis sizes and other information for multi-modal transformer models.

  These values are kept on the main transformer object to simplify model
  transformations that depend on axis sizes or dtypes, by making it possible
  to infer the shape of intermediate activations in advance.

  Attributes:
    common_head_axes: A map of axis names to sizes for head axes that are common
      to queries, keys, and values.
    query_only_head_axes: A map of axis names to sizes for head axes that are
      only used for queries.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    vocab_size: Number of tokens in the vocabulary.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
    num_tokens_per_image: Number of tokens per image.
  """

  common_head_axes: dict[str, int]
  query_only_head_axes: dict[str, int]
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  vocab_size: int
  activation_dtype: jax.typing.DTypeLike
  num_tokens_per_image: int | None = None


@pz.pytree_dataclass
class MultiModalTransformerLM(pz.nn.Layer):
  """Top-level multi-modaltransformer wrapper.

  This class is a simple wrapper that holds configuration data and runs safety
  checks.

  Attributes:
    body: The implementation of the transformer.
    metadata: The axis size and dtype info for the transformer.
  """

  vision_transformer: pz.nn.Layer
  vision_projection: pz.nn.Layer
  body: pz.nn.Layer
  metadata: MultiModalTransformerMetadata = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(
      self,
      tokens: pz.nx.NamedArray,
      *,
      token_positions: pz.nx.NamedArray | None = None,
      attention_mask: pz.nx.NamedArray | None = None,
      images: pz.nx.NamedArray | None = None,
      images_mask: pz.nx.NamedArray | None = None,
      images_start_mask: pz.nx.NamedArray | None = None,
      **side_inputs,
  ) -> pz.nx.NamedArray:
    """Scores log-probabilities for the given inputs.

    Args:
      tokens: Array of token IDs, as an integer named array with a "seq" axis
        and possibly batch axes. Usually starts with the beginning-of-sequence
        token.
      token_positions: Array of token positions, as an integer named array with
        a "seq" axis and possibly batch axes. Usually starts with 0. Inferred to
        start from 0 and increment along the "seq" axis if not provided.
      attention_mask: Array of attention mask, as a boolean named array with a
        "seq" axis and possibly batch axes. If not provided, it will be inferred
        from the token positions. Gemma 3 attention mask is a little
        complicated, as the text uses causal attention, while the images use
        bidirectional attention.
      images: Array of images, as a float named array with a "height" and
        "width" axis and possibly batch (or frame) axes.
      images_mask: Array of boolean mask indicating the token positions of image
        embeddings.
      images_start_mask: Array of boolean mask indicating the token positions of
        image starts.
      **side_inputs: Side inputs, which will be forwarded to the body.

    Returns:
      The final matrix of logits from the embedding decoding layer, which
      (in the normal configuration) will have axes "seq" and "vocabulary".
    """
    if images is None:
      if token_positions is None:
        token_positions = pz.nx.arange("seq", tokens.named_shape["seq"])
      vision_embeddings = None
    else:
      if attention_mask is None:
        # prepare the input
        # we first get the mask indicating the start of each image
        images_start_mask = tokens == START_OF_IMAGE
        # we then unwrap tokens into jax arrays for easier manipulation
        tokens = tokens.untag("batch", "seq").unwrap()
        # prepare tokens with MM tokens
        tokens = build_mm_tokens(
            tokens,
            max_num_images=images.named_shape["frame"],
            num_tokens_per_image=self.metadata.num_tokens_per_image,
        )
        # prepare attention mask
        attention_mask = build_attention_mask(tokens)
        # prepare the token positions
        token_positions = build_token_positions(tokens)
        # wrap tokens back to named array
        tokens = pz.nx.wrap(tokens).tag("batch", "seq")
        token_positions = pz.nx.wrap(token_positions).tag("batch", "seq")
        attention_mask = pz.nx.wrap(attention_mask).tag(
            "batch", "seq", "kv_seq"
        )
        # prepare masks related to images
        images_mask = tokens == SOFT_TOKEN_PLACEHOLDER
      # run the vision transformer on the images and project the embeddings
      vision_embeddings = self.vision_transformer(images, **side_inputs)
      vision_embeddings = self.vision_projection(
          vision_embeddings, **side_inputs
      )
    # run the transformer on the merged embeddings
    value = self.body(
        tokens,
        token_positions=token_positions,
        attention_mask=attention_mask,
        images=vision_embeddings,
        images_mask=images_mask,
        images_start_mask=images_start_mask,
        **side_inputs,
    )
    return value


@dataclasses.dataclass(kw_only=True)
class GemmaTransformerConfig:
  """Common configuration parameters for a gemma transformer.

  These are held in a single configuration object to simplify argument passing
  during construction of the model.

  Attributes:
    num_kv_heads: The number of key-value attention heads or head groups.
    query_head_multiplier: The number of query heads for each KV head.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections. Usually
      ``embedding_dim // num_heads``.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    num_decoder_blocks: Number of transformer decoder blocks in the model.
    vocab_size: Number of tokens in the vocabulary.
    mlp_variant: Gated linear unit variant for MLPs.
    tie_embedder_and_logits: Whether to tie the weights of the input token
      embedding and output logit layers. If True, also scales down input token
      embeddings by sqrt(embedding_dim). (This is used by Gemma.)
    rope_wavelength: Wavelength for global RoPE layers (and for local RoPE
      layers if local_rope_wavelength is not set).
    rms_norm_eps: Epsilon for RMSNorm layers.
    attention_type: A single attention type or sequence of per-layer attention
      types. If a sequence, its length should evenly divide the number of
      decoder blocks, and will be repeated to match the number of blocks.
    use_post_attn_norm: Whether to add a normalization layer after the attention
      block.
    use_post_ffw_norm: Whether to add a normalization layer after the
      feedforward block.
    final_logit_softcap: If not None, used as the tanh soft cap for the final
      transformer logits.
    attn_logits_soft_cap: If not None, used as the tanh soft cap for the
      attention logits.
    query_scaling_factor: Scaling factor for the query vectors. If "default",
      defaults to 1 / sqrt(projection_dim).
    parameter_dtype: Floating dtype to use for all parameters.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
    use_layer_stack: Whether to stack the blocks together using a LayerStack.
    use_qk_norm: Whether to use QK normalization.
    global_scale_factor: Scale factor for the global RoPE layers (scale factor
      for the local RoPE layers is set as 1.0 by default).
    local_rope_wavelength: Wavelength for the local RoPE layers. If None, local
      RoPE layers will use the same wavelength as global RoPE layers
      (config.rope_wavelength).
    vision_config: Configuration for the vision transformer.
  """

  num_kv_heads: int
  query_head_multiplier: int
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  num_decoder_blocks: int
  vocab_size: int
  mlp_variant: Literal["geglu_approx", "swiglu"]
  tie_embedder_and_logits: bool
  rope_wavelength: float = 10_000
  rms_norm_eps: float = 1e-6
  attention_type: AttentionType | Sequence[AttentionType] = (
      AttentionTypeGlobalCausal()
  )
  use_post_attn_norm: bool = False
  use_post_ffw_norm: bool = False
  final_logit_softcap: float | None = None
  attn_logits_soft_cap: float | None = None
  query_scaling_factor: float | Literal["default"] = "default"
  parameter_dtype: jax.typing.DTypeLike = jnp.float32
  activation_dtype: jax.typing.DTypeLike = jnp.float32
  use_layer_stack: bool = False
  use_qk_norm: bool = False
  global_scale_factor: float | None = None
  local_rope_wavelength: float | None = None

  vision_config: siglip.Gemma3SigLipConfig | None = None


def build_gemma_feedforward(
    name: str,
    init_base_rng: jax.Array | None,
    config: GemmaTransformerConfig,
) -> model_parts.TransformerFeedForward:
  """Creates a feedforward block.

  This family of models use gated linear units, as proposed by Shazeer (2020).
  We represent this computation as a composition of simpler Penzai primitives,
  to enable patching and post-processing of the various internal activations.

  Args:
    name: Name of the feedforward block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    An instance of TransformerFeedForward containing the GELU MLP blocks.
  """
  if config.mlp_variant == "geglu_approx":
    # Approximate is already the default in JAX, but we specify it explicitly
    # because defaults differ between JAX and PyTorch.
    act_fn = functools.partial(jax.nn.gelu, approximate=True)
  elif config.mlp_variant == "swiglu":
    act_fn = jax.nn.silu
  else:
    raise ValueError(f"Unsupported MLP variant {config.mlp_variant}")

  return model_parts.TransformerFeedForward([
      pz.nn.BranchAndMultiplyTogether(
          branches=[
              pz.nn.NamedGroup(
                  "gate",
                  [
                      pz.nn.Linear.from_config(
                          name=f"{name}/gating_linear",
                          init_base_rng=init_base_rng,
                          input_axes={"embedding": config.embedding_dim},
                          output_axes={"neurons": config.mlp_hidden_dim},
                          dtype=config.parameter_dtype,
                      ),
                      pz.nn.Elementwise(act_fn),
                  ],
              ),
              pz.nn.Linear.from_config(
                  name=f"{name}/value_linear",
                  init_base_rng=init_base_rng,
                  input_axes={"embedding": config.embedding_dim},
                  output_axes={"neurons": config.mlp_hidden_dim},
                  dtype=config.parameter_dtype,
              ),
          ]
      ),
      pz.nn.Linear.from_config(
          name=f"{name}/out_linear",
          init_base_rng=init_base_rng,
          input_axes={"neurons": config.mlp_hidden_dim},
          output_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
      ),
  ])


def _head_info(config: GemmaTransformerConfig):
  """Computes query, key, and value head axes and einsum names."""
  if config.query_head_multiplier == 1:
    common_head_axes = {"heads": config.num_kv_heads}
    qkv_einsum = {"heads": "h"}
    query_only_head_axes = {}
    q_einsum = {}
  elif config.num_kv_heads == 1:
    common_head_axes = {}
    qkv_einsum = {}
    query_only_head_axes = {"query_heads": config.query_head_multiplier}
    q_einsum = {"query_heads": "h"}
  else:
    common_head_axes = {"head_groups": config.num_kv_heads}
    qkv_einsum = {"head_groups": "hg"}
    query_only_head_axes = {"query_heads": config.query_head_multiplier}
    q_einsum = {"query_heads": "hq"}
  return (common_head_axes, qkv_einsum, query_only_head_axes, q_einsum)


def build_gemma_attention(
    name: str,
    init_base_rng: jax.Array | None,
    config: GemmaTransformerConfig,
    block_index: int | None = None,
) -> pz.nn.Attention:
  """Builds an attention block from a configuration.

  Args:
    name: Name of the attention block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.
    block_index: The index of the transformer block in the list of blocks. Can
      be None if the attention type doesn't depend on the block index.

  Returns:
    An Attention block.
  """
  embedding_dim = config.embedding_dim
  projection_dim = config.projection_dim

  common_head_axes, qkv_einsum, query_only_head_axes, q_einsum = _head_info(
      config
  )

  if config.query_scaling_factor == "default":
    query_scaling_factor = projection_dim**-0.5
  else:
    query_scaling_factor = config.query_scaling_factor

  # As used in https://github.com/google-deepmind/gemma.
  # (This exact value is probably not important.)
  masked_out_value = jnp.array(-2.3819763e38, dtype=config.activation_dtype)

  if isinstance(config.attention_type, AttentionType):
    attention_type = config.attention_type
  else:
    if block_index is None:
      raise ValueError(
          "block_index must be specified if attention_type is a sequence."
      )
    attention_type = config.attention_type[
        block_index % len(config.attention_type)
    ]

  if isinstance(attention_type, AttentionTypeSlidingWindowCausal):
    attn_masker = ApplyMultiModalCausalSlidingWindowAttentionMask(
        sliding_window_size=attention_type.window_size,
        masked_out_value=masked_out_value,
    )
    # Decide which wavelength to use for local RoPE.
    if config.local_rope_wavelength is not None:
      wavelength = config.local_rope_wavelength
    else:
      wavelength = config.rope_wavelength
    scale_factor = 1.0
  elif isinstance(attention_type, AttentionTypeGlobalCausal):
    attn_masker = ApplyMultiModalCausalAttentionMask(
        masked_out_value=masked_out_value,
    )
    wavelength = config.rope_wavelength
    # Decide which scale factor to use for global RoPE.
    if config.global_scale_factor is not None:
      scale_factor = config.global_scale_factor
    else:
      scale_factor = 1.0
  else:
    raise ValueError(f"Unsupported attention type {attention_type}")

  query_key_to_attn_sublayers = [
      pz.nn.NamedEinsum(
          (
              {"seq": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
              {"seq": "tkv", **qkv_einsum, "projection": "p"},
          ),
          {"seq": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
      ),
  ]
  if config.attn_logits_soft_cap is not None:
    query_key_to_attn_sublayers.append(
        pz.nn.TanhSoftCap(
            soft_cap=jnp.array(
                config.attn_logits_soft_cap, dtype=config.activation_dtype
            )
        )
    )
  query_key_to_attn_sublayers.extend([
      attn_masker,
      pz.nn.Softmax("kv_seq"),
  ])

  # add qk norm if needed in the module of input_to_query sublayers
  input_to_query_sublayers = [
      pz.nn.Linear.from_config(
          name=f"{name}/query",
          init_base_rng=init_base_rng,
          input_axes={"embedding": embedding_dim},
          output_axes={
              **common_head_axes,
              **query_only_head_axes,
              "projection": projection_dim,
          },
          dtype=config.parameter_dtype,
      ),
  ]
  if config.use_qk_norm:
    input_to_query_sublayers.append(
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/query_norm",
            init_base_rng=init_base_rng,
            across_axes={"projection": config.projection_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        ),
    )
  input_to_query_sublayers.extend([
      pz.nn.ApplyRoPE(
          positions_input_name="token_positions",
          embedding_axis="projection",
          max_wavelength=wavelength,
          scale_factor=scale_factor,
      ),
      pz.nn.ConstantRescale(
          by=jnp.array(query_scaling_factor, dtype=config.activation_dtype)
      ),
  ])

  # add qk norm if needed in the module of input_to_key sublayers
  input_to_key_sublayers = [
      pz.nn.Linear.from_config(
          name=f"{name}/key",
          init_base_rng=init_base_rng,
          input_axes={"embedding": embedding_dim},
          output_axes={**common_head_axes, "projection": projection_dim},
          dtype=config.parameter_dtype,
      ),
  ]
  if config.use_qk_norm:
    input_to_key_sublayers.append(
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/key_norm",
            init_base_rng=init_base_rng,
            across_axes={"projection": config.projection_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        ),
    )
  input_to_key_sublayers.append(
      pz.nn.ApplyRoPE(
          positions_input_name="token_positions",
          embedding_axis="projection",
          max_wavelength=wavelength,
          scale_factor=scale_factor,
      ),
  )

  return pz.nn.Attention(
      input_to_query=pz.nn.Sequential(input_to_query_sublayers),
      input_to_key=pz.nn.Sequential(input_to_key_sublayers),
      input_to_value=pz.nn.Sequential([
          pz.nn.Linear.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      query_key_to_attn=pz.nn.Sequential(query_key_to_attn_sublayers),
      attn_value_to_output=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"seq": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
                  {"seq": "tkv", **qkv_einsum, "projection": "p"},
              ),
              {"seq": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
          ),
          pz.nn.Linear.from_config(
              name=f"{name}/output",
              init_base_rng=init_base_rng,
              input_axes={
                  **common_head_axes,
                  **query_only_head_axes,
                  "projection": projection_dim,
              },
              output_axes={"embedding": embedding_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
  )


def build_gemma_block(
    name: str,
    init_base_rng: jax.Array | None,
    config: GemmaTransformerConfig,
    block_index: int | None = None,
) -> model_parts.TransformerBlock:
  """Builds a transformer block from a configuration.

  Args:
    name: Name of the block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.
    block_index: The index of the transformer block in the list of blocks. Can
      be None if the attention type doesn't depend on the block index.

  Returns:
    A full transformer block.
  """
  attn_sequence = [
      pz.nn.RMSLayerNorm.from_config(
          name=f"{name}/pre_attention_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
          epsilon=config.rms_norm_eps,
      ),
      build_gemma_attention(
          f"{name}/attention",
          init_base_rng,
          config,
          block_index=block_index,
      ),
  ]
  if config.use_post_attn_norm:
    attn_sequence.append(
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/post_attention_norm",
            init_base_rng=init_base_rng,
            across_axes={"embedding": config.embedding_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        )
    )
  ffw_sequence = [
      pz.nn.RMSLayerNorm.from_config(
          name=f"{name}/pre_ffw_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
          epsilon=config.rms_norm_eps,
      ),
      build_gemma_feedforward(f"{name}/mlp", init_base_rng, config),
  ]
  if config.use_post_ffw_norm:
    ffw_sequence.append(
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/post_ffw_norm",
            init_base_rng=init_base_rng,
            across_axes={"embedding": config.embedding_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        )
    )
  return model_parts.TransformerBlock(
      sublayers=[
          pz.nn.Residual(pz.nn.Sequential(attn_sequence)),
          pz.nn.Residual(pz.nn.Sequential(ffw_sequence)),
      ],
  )


def build_gemma_multimodal_transformer(
    config: GemmaTransformerConfig,
    init_base_rng: jax.Array | None = None,
    name: str = "transformer",
    vision_config: siglip.Gemma3SigLipConfig | None = None,
) -> MultiModalTransformerLM:
  """Builds a Gemma-style multimodal transformer model from a configuration.

  Args:
    config: The configuration of the model.
    init_base_rng: Base RNG for initializing the parameters.
    name: Name for the top-level model, used as a prefix for all parameters.
    vision_config: Configuration for the vision transformer. If None, the model
      will not include a vision transformer and will be text-only.

  Returns:
    A full transformer model.
  """

  # Embedding table is shared between first and last layers.
  emb_table = pz.nn.EmbeddingTable.from_config(
      name=f"{name}/embedder",
      init_base_rng=init_base_rng,
      vocab_size=config.vocab_size,
      embedding_axes={"embedding": config.embedding_dim},
      dtype=config.parameter_dtype,
  )
  sublayers = []
  sublayers.append(pz.nn.EmbeddingLookup(emb_table))
  if config.activation_dtype != config.parameter_dtype:
    sublayers.append(pz.nn.CastToDType(config.activation_dtype))

  if config.tie_embedder_and_logits:
    sublayers.append(
        pz.nn.ConstantRescale(
            by=jnp.sqrt(config.embedding_dim).astype(config.activation_dtype)
        )
    )

  if vision_config is not None:
    vision_transformer_layers = siglip.build_siglip_vision_transformer(
        vision_config, init_base_rng, name=f"{name}/vision_encoder"
    )
    vision_projection_layers = pz.nn.Sequential([
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/mm_soft_embedding_norm",
            init_base_rng=init_base_rng,
            across_axes={"embedding": config.embedding_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        ),
        pz.nn.RenameAxes(
            old="embedding",
            new="vision_embedding",
        ),
        pz.nn.Linear.from_config(
            name=f"{name}/mm_input_projection",
            init_base_rng=init_base_rng,
            input_axes={"vision_embedding": vision_config.embedding_dim},
            output_axes={"embedding": config.embedding_dim},
            dtype=config.parameter_dtype,
        ),
        vision_layers.MergeAxes(
            input_axes=("frame", "patch"),
            output_axes="seq",
        ),
    ])
    sublayers.append(
        StitchEmbeddings(
            vision_input_name="images",
            mask_input_name="images_mask",
        )
    )
  else:
    vision_transformer_layers = None
    vision_projection_layers = None

  if config.use_layer_stack:
    if not isinstance(config.attention_type, AttentionType):
      raise ValueError(
          "Layer stack does not currently support per-layer attention types."
      )
    sublayers.append(
        pz.nn.LayerStack.from_sublayer_builder(
            builder=build_gemma_block,
            stack_axis="blocks",
            stack_axis_size=config.num_decoder_blocks,
            init_base_rng=init_base_rng,
            builder_kwargs=dict(name=f"{name}/blocks", config=config),
        )
    )
  else:
    if not isinstance(config.attention_type, AttentionType):
      if config.num_decoder_blocks % len(config.attention_type) != 0:
        raise ValueError(
            "Per-layer attention types must have a length that divides the"
            " number of blocks."
        )
    for block_index in range(config.num_decoder_blocks):
      sublayers.append(
          build_gemma_block(
              f"{name}/block_{block_index}", init_base_rng, config, block_index
          )
      )

  sublayers.append(
      pz.nn.RMSLayerNorm.from_config(
          name=f"{name}/final_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
          epsilon=config.rms_norm_eps,
      )
  )

  if config.tie_embedder_and_logits:
    sublayers.append(pz.nn.EmbeddingDecode(emb_table))
  else:
    sublayers.append(
        pz.nn.Linear.from_config(
            name=f"{name}/lm_head",
            init_base_rng=init_base_rng,
            input_axes={"embedding": config.embedding_dim},
            output_axes={"vocabulary": config.vocab_size},
        )
    )

  if config.final_logit_softcap:
    sublayers.append(
        pz.nn.TanhSoftCap(
            soft_cap=jnp.array(
                config.final_logit_softcap, dtype=config.activation_dtype
            )
        )
    )

  if vision_config is not None:
    sublayers.append(
        TruncateVisionLogits(
            vision_input_name="images",
            mask_input_name="images_start_mask",
            num_tokens_per_image=vision_config.num_mm_tokens_per_image,
        )
    )

  common_head_axes, _, query_only_head_axes, _ = _head_info(config)
  return MultiModalTransformerLM(
      metadata=MultiModalTransformerMetadata(
          common_head_axes=common_head_axes,
          query_only_head_axes=query_only_head_axes,
          embedding_dim=config.embedding_dim,
          projection_dim=config.projection_dim,
          mlp_hidden_dim=config.mlp_hidden_dim,
          vocab_size=config.vocab_size,
          activation_dtype=config.activation_dtype,
          num_tokens_per_image=vision_config.num_mm_tokens_per_image
          if vision_config is not None
          else None,
      ),
      body=pz.nn.Sequential(sublayers),
      vision_transformer=vision_transformer_layers,
      vision_projection=vision_projection_layers,
  )
