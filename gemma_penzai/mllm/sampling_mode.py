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

"""Sampling-mode adapters for TransformerLM / TransformerMultiModalLM models.

This file includes the kv-cache sampling mode of the base TransformerLM model
and the base TransformerMultiModalLM model (multimodal support).

This mode is intended to be hot-swapped for the main TransformerLM /
TransformerMultiModalLM implementation: you should generally start by loading a
`TransformerLM` or `TransformerMultiModalLM` and then converting it to a
`KVCachingTransformerLM` or `KVCachingTransformerMultiModalLM` using
`from_uncached`.

The layers defined here follow the same conventions documented in the module
docstring for `model_parts`. In addition:

* Where applicable, "kv_token_positions" is the name of the side input that
  provides the position of each token for the purposes of positional embeddings.

* Where applicable, "cache_end_index" is the name of the side input that
  identifies the current length of the key/value cache state.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from gemma_penzai.mllm.models import gemma_multimodal
from gemma_penzai.mllm.models import gemma_utils
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts


build_mm_tokens = gemma_utils.build_mm_tokens
build_attention_mask = gemma_utils.build_attention_mask
SOFT_TOKEN_PLACEHOLDER = gemma_utils.SOFT_TOKEN_PLACEHOLDER
START_OF_IMAGE = gemma_utils.START_OF_IMAGE
END_OF_IMAGE = gemma_utils.END_OF_IMAGE


@pz.pytree_dataclass
class KVCachingTransformerLM(pz.nn.Layer):
  """Top-level transformer in (stateful) cached autoregressive sampling mode.

  This class represents the sampling mode of the model, and manages the sampling
  state. It is designed to be loaded from an existing `Transformer`. If you want
  to load this from the pretrained checkpoint, first load a `Transformer`, then
  call `KVCachingTransformer.from_uncached`.

  This class handles and automatically increments token positions based on the
  tokens it has generated so far.

  Attributes:
    body: The implementation of the transformer. Usually a nested set of state
      and side-effect handlers wrapping the main sequence of transformer blocks,
      but may be modified after the model is loaded due to patching.
    cache_end_index: A variable containing the current end index of the caches.
    previous_tokens: A variable containing all previously-seen tokens.
    metadata: The configuration for the transformer.
    cache_len: The length of the internal key-value caches.
    batch_axes: The batch axes of the internal key-value caches.
    pad_id: Token ID that indicates padding.
  """

  body: pz.nn.Layer
  cache_end_index: pz.StateVariable[int]
  previous_tokens: pz.StateVariable[pz.nx.NamedArray]
  metadata: model_parts.TransformerMetadata = dataclasses.field(
      metadata={"pytree_node": False}
  )
  cache_len: int = dataclasses.field(metadata={"pytree_node": False})
  batch_axes: dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  pad_id: int = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self, tokens: pz.nx.NamedArray, **extra_side_inputs: dict[Any, Any]
  ) -> pz.nx.NamedArray:
    """Processes a new subsequence of tokens and adds them to the K/V cache.

    When called, the internal variables tracking the key-value cache will be
    updated with the new state.

    Args:
      tokens: Array of token IDs, as an integer named array with a "seq" axis
        and possibly batch axes. The batch axes must match the `batch_axes`
        attribute. Padding tokens are ignored.
      **extra_side_inputs: Extra side inputs, which will be forwarded on to the
        body. The "token_positions", "kv_token_positions", and "cache_end_index"
        inputs will be added automatically and do not need to be provided.

    Returns:
      Matrix of logits from the embedding decoding layer, which (in the
      normal configuration) will have axes "seq" and "vocabulary".
    """
    pz.chk.check_structure(
        tokens,
        pz.chk.ArraySpec(
            named_shape={**self.batch_axes, "seq": pz.chk.var("S")}
        ),
    )

    # Update the written tokens.
    updated_tokens = pz.nx.nmap(jax.lax.dynamic_update_slice)(
        self.previous_tokens.value.untag("seq"),
        tokens.untag("seq"),
        (self.cache_end_index.value,),
    ).tag("seq")
    # Query position and key position are both based on the number of preceding
    # padding tokens (after updating the written tokens). Compute by summing
    # up non-padding tokens, then slicing the part we are updating.
    kv_nonpad_mask = updated_tokens != self.pad_id
    kv_nonpad_so_far_inclusive = pz.nx.nmap(jnp.cumsum)(
        kv_nonpad_mask.untag("seq"), dtype=jnp.int32
    ).tag("seq")
    key_value_positions = pz.nx.nmap(jnp.where)(
        kv_nonpad_mask, kv_nonpad_so_far_inclusive - 1, -1
    )
    query_positions = pz.nx.nmap(jax.lax.dynamic_slice)(
        key_value_positions.untag("seq"),
        (self.cache_end_index.value,),
        (tokens.named_shape["seq"],),
    ).tag("seq")
    # Run the model.
    outs = self.body(
        tokens,
        token_positions=query_positions,
        kv_token_positions=key_value_positions,
        cache_end_index=self.cache_end_index.value,
        **extra_side_inputs,
    )
    # Update the state variables.
    self.previous_tokens.value = updated_tokens
    self.cache_end_index.value = (
        self.cache_end_index.value + tokens.named_shape["seq"]
    )

    return outs

  @classmethod
  def from_uncached(
      cls,
      uncached: model_parts.TransformerLM,
      cache_len: int,
      batch_axes: dict[str, int],
      pad_id: int = 0,
      variable_name_prefix: str = "sampler",
  ) -> KVCachingTransformerLM:
    """Transforms a `Transformer` into cached sampling mode.

    This constructor hot-swaps all `pz.nn.Attention` layers in the
    original model to enable key-value caching, then installs new handlers to
    update their states appropriately. Note that any modifications to the
    uncached model will persist in the decoding mode.

    Args:
      uncached: The original `Transformer` model.
      cache_len: Maximum sequence length for the key/value caches.
      batch_axes: Names and sizes for the batch axes that will be used for
        sampling. Required for initializing the key/value caches.
      pad_id: ID for the padding token.
      variable_name_prefix: Prefix for cached sampling variable names.

    Returns:
      A KVCachingTransformer.
    """

    def _fix_attn_mask(masker):
      if masker.kv_positions_input_name != "token_positions":
        raise ValueError(
            "Could not automatically convert attention mask layer with"
            f" non-standard positions input name: {masker}"
        )
      return dataclasses.replace(
          masker, kv_positions_input_name="kv_token_positions"
      )

    cached_axes = {
        **batch_axes,
        **uncached.metadata.common_head_axes,
        "projection": uncached.metadata.projection_dim,
    }
    attn_sel = pz.select(uncached.body).at_instances_of(pz.nn.Attention)
    fixed_attns = {}
    for ix, (keypath, attn) in enumerate(attn_sel.selected_by_path.items()):
      attn_with_new_kv_positions = (
          pz.select(attn)
          .at_instances_of(
              pz.nn.ApplyCausalAttentionMask
              | pz.nn.ApplyCausalSlidingWindowAttentionMask
          )
          .apply(_fix_attn_mask)
      )
      fixed_attns[keypath] = pz.nn.KVCachingAttention.from_uncached(
          attn_with_new_kv_positions,
          cache_len=cache_len,
          cached_axes=cached_axes,
          cache_dtype=uncached.metadata.activation_dtype,
          sequence_axis="seq",
          cache_end_index_key="cache_end_index",
          cache_label=f"{variable_name_prefix}/cache_{ix}",
          layerstack_axes=pz.nn.layerstack_axes_from_keypath(keypath),
      )
    caching_body = attn_sel.set_by_path(fixed_attns)
    return cls(
        metadata=uncached.metadata,
        cache_len=cache_len,
        batch_axes=batch_axes,
        pad_id=pad_id,
        previous_tokens=pz.StateVariable(
            value=pz.nx.zeros(
                {**batch_axes, "seq": cache_len}, dtype=jnp.int32
            ),
            label=f"{variable_name_prefix}/previous_tokens",
        ),
        cache_end_index=pz.StateVariable(
            value=0,
            label=f"{variable_name_prefix}/cache_end_index",
        ),
        body=caching_body,
    )


@pz.pytree_dataclass
class KVCachingTransformerMultiModalLM(pz.nn.Layer):
  """Top-level multi-modal transformer in (stateful) cached autoregressive sampling mode.

  This class represents the sampling mode of the model, and manages the sampling
  state. It is designed to be loaded from an existing `Transformer`. If you want
  to load this from the pretrained checkpoint, first load a `Transformer`, then
  call `KVCachingTransformer.from_uncached`.

  This class handles and automatically increments token positions based on the
  tokens it has generated so far.

  Attributes:
    body: The implementation of the transformer. Usually a nested set of state
      and side-effect handlers wrapping the main sequence of transformer blocks,
      but may be modified after the model is loaded due to patching.
    cache_end_index: A variable containing the current end index of the caches.
    previous_tokens: A variable containing all previously-seen tokens.
    metadata: The configuration for the transformer.
    cache_len: The length of the internal key-value caches.
    batch_axes: The batch axes of the internal key-value caches.
    pad_id: Token ID that indicates padding.
  """

  body: pz.nn.Layer
  cache_end_index: pz.StateVariable[int]
  previous_tokens: pz.StateVariable[pz.nx.NamedArray]
  metadata: gemma_multimodal.MultiModalTransformerMetadata = dataclasses.field(
      metadata={"pytree_node": False}
  )
  cache_len: int = dataclasses.field(metadata={"pytree_node": False})
  batch_axes: dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  pad_id: int = dataclasses.field(metadata={"pytree_node": False})
  num_tokens_per_image: int = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self,
      tokens: pz.nx.NamedArray,
      images: pz.nx.NamedArray | None = None,
      **extra_side_inputs: dict[Any, Any],
  ) -> pz.nx.NamedArray:
    """Processes a new subsequence of tokens and adds them to the K/V cache.

    When called, the internal variables tracking the key-value cache will be
    updated with the new state.

    Args:
      tokens: Array of token IDs, as an integer named array with a "seq" axis
        and possibly batch axes. The batch axes must match the `batch_axes`
        attribute. Padding tokens are ignored.
      images: Array of image patches, as a float named array with a "frame" axis
        and possibly batch axes.
      **extra_side_inputs: Extra side inputs, which will be forwarded on to the
        body. The "token_positions", "kv_token_positions", and "cache_end_index"
        inputs will be added automatically and do not need to be provided.

    Returns:
      Matrix of logits from the embedding decoding layer, which (in the
      normal configuration) will have axes "seq" and "vocabulary".
    """
    pz.chk.check_structure(
        tokens,
        pz.chk.ArraySpec(
            named_shape={**self.batch_axes, "seq": pz.chk.var("S")}
        ),
    )

    if images is not None:
      # we first get the mask indicating the start of each image
      images_start_mask = tokens == START_OF_IMAGE
      # we then unwrap tokens into jax arrays for easier manipulation
      assert len(self.batch_axes) == 1
      tokens = tokens.untag(*self.batch_axes.keys(), "seq").unwrap()
      # prepare tokens with MM tokens
      tokens = build_mm_tokens(
          tokens,
          max_num_images=images.named_shape["frame"],
          num_tokens_per_image=self.num_tokens_per_image,
      )
      # prepare attention mask
      attention_mask = build_attention_mask(tokens)
      attention_mask = jnp.concatenate(
          [
              attention_mask,
              jnp.zeros(
                  (
                      attention_mask.shape[0],
                      attention_mask.shape[1],
                      self.cache_len - attention_mask.shape[2],
                  ),
                  dtype=attention_mask.dtype,
              ),
          ],
          axis=-1,
      )
      # wrap tokens back to named array
      tokens = pz.nx.wrap(tokens).tag(*self.batch_axes.keys(), "seq")
      attention_mask = pz.nx.wrap(attention_mask).tag(
          *self.batch_axes.keys(), "seq", "kv_seq"
      )
      # prepare masks related to images
      images_mask = tokens == SOFT_TOKEN_PLACEHOLDER
    else:
      images_start_mask = None
      attention_mask = None
      images_mask = None

    # Update the written tokens.
    updated_tokens = pz.nx.nmap(jax.lax.dynamic_update_slice)(
        self.previous_tokens.value.untag("seq"),
        tokens.untag("seq"),
        (self.cache_end_index.value,),
    ).tag("seq")
    # Query position and key position are both based on the number of preceding
    # padding tokens (after updating the written tokens). Compute by summing
    # up non-padding tokens, then slicing the part we are updating.
    kv_nonpad_mask = updated_tokens != self.pad_id
    kv_nonpad_so_far_inclusive = pz.nx.nmap(jnp.cumsum)(
        kv_nonpad_mask.untag("seq"), dtype=jnp.int32
    ).tag("seq")
    key_value_positions = pz.nx.nmap(jnp.where)(
        kv_nonpad_mask, kv_nonpad_so_far_inclusive - 1, -1
    )
    query_positions = pz.nx.nmap(jax.lax.dynamic_slice)(
        key_value_positions.untag("seq"),
        (self.cache_end_index.value,),
        (tokens.named_shape["seq"],),
    ).tag("seq")
    # Run the model.
    outs = self.body(
        tokens,
        images=images,
        images_mask=images_mask,
        images_start_mask=images_start_mask,
        token_positions=query_positions,
        attention_mask=attention_mask,
        kv_token_positions=key_value_positions,
        cache_end_index=self.cache_end_index.value,
        **extra_side_inputs,
    )
    # Update the state variables.
    self.previous_tokens.value = updated_tokens
    self.cache_end_index.value = (
        self.cache_end_index.value + tokens.named_shape["seq"]
    )

    return outs

  @classmethod
  def from_uncached(
      cls,
      uncached: gemma_multimodal.MultiModalTransformerLM,
      cache_len: int,
      batch_axes: dict[str, int],
      pad_id: int = 0,
      variable_name_prefix: str = "sampler",
      num_tokens_per_image: int = 256,
  ) -> KVCachingTransformerMultiModalLM:
    """Transforms a `Transformer` into cached sampling mode.

    This constructor hot-swaps all `pz.nn.Attention` layers in the
    original model to enable key-value caching, then installs new handlers to
    update their states appropriately. Note that any modifications to the
    uncached model will persist in the decoding mode.

    Args:
      uncached: The original `Transformer` model.
      cache_len: Maximum sequence length for the key/value caches.
      batch_axes: Names and sizes for the batch axes that will be used for
        sampling. Required for initializing the key/value caches.
      pad_id: ID for the padding token.
      variable_name_prefix: Prefix for cached sampling variable names.
      num_tokens_per_image: Number of tokens per image.

    Returns:
      A KVCachingTransformer.
    """

    def _fix_attn_mask(masker):
      if masker.kv_positions_input_name != "token_positions":
        raise ValueError(
            "Could not automatically convert attention mask layer with"
            f" non-standard positions input name: {masker}"
        )
      return dataclasses.replace(
          masker, kv_positions_input_name="kv_token_positions"
      )

    cached_axes = {
        **batch_axes,
        **uncached.metadata.common_head_axes,
        "projection": uncached.metadata.projection_dim,
    }
    attn_sel = (
        pz.select(uncached)
        .at(lambda x: x.body)
        .at_instances_of(pz.nn.Attention)
    )
    fixed_attns = {}
    for ix, (keypath, attn) in enumerate(attn_sel.selected_by_path.items()):
      attn_with_new_kv_positions = (
          pz.select(attn)
          .at_instances_of(
              gemma_multimodal.ApplyMultiModalCausalAttentionMask
              | gemma_multimodal.ApplyMultiModalCausalSlidingWindowAttentionMask
          )
          .apply(_fix_attn_mask)
      )
      fixed_attns[keypath] = pz.nn.KVCachingAttention.from_uncached(
          attn_with_new_kv_positions,
          cache_len=cache_len,
          cached_axes=cached_axes,
          cache_dtype=uncached.metadata.activation_dtype,
          sequence_axis="seq",
          cache_end_index_key="cache_end_index",
          cache_label=f"{variable_name_prefix}/cache_{ix}",
          layerstack_axes=pz.nn.layerstack_axes_from_keypath(keypath),
      )
    caching_body = attn_sel.set_by_path(fixed_attns)
    return cls(
        metadata=uncached.metadata,
        cache_len=cache_len,
        batch_axes=batch_axes,
        pad_id=pad_id,
        previous_tokens=pz.StateVariable(
            value=pz.nx.zeros(
                {**batch_axes, "seq": cache_len}, dtype=jnp.int32
            ),
            label=f"{variable_name_prefix}/previous_tokens",
        ),
        cache_end_index=pz.StateVariable(
            value=0,
            label=f"{variable_name_prefix}/cache_end_index",
        ),
        body=caching_body,
        num_tokens_per_image=num_tokens_per_image,
    )
