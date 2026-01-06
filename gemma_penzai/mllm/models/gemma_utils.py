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

"""Tokens manipulation utils for Gemma 3 multimodal transformer.

This file implements the following functions:
  * build_mm_tokens: Add image placeholder tokens to the text tokens.
  * build_token_positions: Build token positions after adding image plaeholder
  tokens.
  * build_attention_mask: Build the attention mask, the text part follows causal
  attention mask, while the image part follows bidirectional attention mask.

The inputs are jax arrays, instead of named arrays in penzai, for convenience.
The functions are mainly borrowed from
https://github.com/google-deepmind/gemma/blob/main/gemma/gm/vision/_token_utils.py.
"""

from typing import Union

import jax
import jax.numpy as jnp
import jaxtyping


Float = jaxtyping.Float
Int = jaxtyping.Int
Bool = jaxtyping.Bool
ArrayT = jaxtyping.Array

# `\n\n` token for Gemma 3 tokenizer.
_DOUBLE_NEW_LINE_TOKEN = 108
# Padding token for Gemma 3 tokenizer.
_PADDING_ID = 0

# This is not a real token, but a placeholder to indicate the position of the
# MM tokens. Those placeholders are later replaced by the MM tokens from the
# vision encoder.
# This should never be manipulated by the end-user.
SOFT_TOKEN_PLACEHOLDER = -2
START_OF_IMAGE = 255999
END_OF_IMAGE = 256000


def build_mm_tokens(
    tokens: Int[ArrayT, 'B L'],
    *,
    max_num_images: int,
    num_tokens_per_image: int,
) -> Int[ArrayT, 'B L+(max_num_images * (num_tokens_per_image + 3))']:
  r"""Add the extra image tokens to the text tokens.

  If the model has images, we expand each `<start_of_image>` token by the image
  placeholder tokens.

  Example:

  ```python
  input = [..., x, <start_of_image>, y, ...]
  output = [
      ..., x, \n\n, <start_of_image>, SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, ..., SOFT_TOKEN_PLACEHOLDER,
      SOFT_TOKEN_PLACEHOLDER, <end_of_image>, \n\n, y, ...
  ]
  ```

  The `\n\n` tokens are added to match how the model was trained.

  Args:
    tokens: The text tokens.
    max_num_images: The maximum number of images in the batch.
    num_tokens_per_image: The number of soft tokens per image.

  Returns:
    The text tokens with the extra image tokens.
  """

  # New tokens which will be inserted for each image.
  mm_tokens = [
      _DOUBLE_NEW_LINE_TOKEN,
      START_OF_IMAGE,
      *[SOFT_TOKEN_PLACEHOLDER] * num_tokens_per_image,
      END_OF_IMAGE,
      _DOUBLE_NEW_LINE_TOKEN,
  ]

  return _insert_sequence(
      at=START_OF_IMAGE,
      sequence=mm_tokens,
      tokens=tokens,
      max_num_images=max_num_images,
  )


def build_token_positions(
    tokens_with_mm: Int[ArrayT, 'B L'],
) -> Int[ArrayT, 'B L']:
  """Build the token positions for the multimodal transformer.

  Args:
    tokens_with_mm: The text tokens with the extra image placeholder tokens.

  Returns:
    The token positions.
  """
  inputs_mask = tokens_with_mm != _PADDING_ID
  return _build_token_positions_from_mask(inputs_mask)


def build_attention_mask(
    tokens_with_mm: Int[ArrayT, 'B L'],
) -> Bool[ArrayT, 'B L L']:
  """Build the attention mask for the multimodal transformer.

  The text part follows causal attention mask, while the image part follows
  bidirectional attention mask.

  Args:
    tokens_with_mm: The text tokens with the extra image placeholder tokens.

  Returns:
    The attention mask.
  """
  inputs_mask = tokens_with_mm != _PADDING_ID
  bidirectional_mask = tokens_with_mm == SOFT_TOKEN_PLACEHOLDER
  return _make_causal_bidirectional_attention_mask(
      inputs_mask, bidirectional_mask=bidirectional_mask
  )


def _get_num_mm_tokens(
    *,
    max_num_images: int,
    num_tokens_per_image: int,
) -> int:
  # +3 as `\n\n`, '\n\n' `<end_of_image>` are inserted. The
  # `<start_of_image>` token is already present in the text tokens, so is not
  # counted.
  return max_num_images * (num_tokens_per_image + 3)


def _insert_sequence(
    tokens: Int[ArrayT, 'B L'],
    *,
    at: int,
    sequence: Int[ArrayT, 'L'],
    max_num_images: int,
) -> Int[ArrayT, 'B L']:
  """Utility function for build_mm_tokens.

  Insert a sequence of tokens at a given position.

  Args:
    tokens: The text tokens.
    at: The position to insert the sequence at.
    sequence: The sequence to insert.
    max_num_images: The maximum number of images in the batch.

  Returns:
    The text tokens with the extra image tokens.
  """
  _, length = tokens.shape

  mm_tokens = jnp.array(sequence, dtype=jnp.int32)

  # `-1` because `<start_of_image>` is already present in the input tokens.
  offset_by = len(mm_tokens) - 1

  # Maximum length, if all images are present.
  length_with_mm = length + max_num_images * offset_by

  mm_start = tokens == at

  # Get the text tokens correctly placed at their final position.
  # The `<start_of_image>` are removed and expanded to leave space for the MM
  # tokens.
  # tokens = [..., x, <start_of_image>, y, ...]
  # new_text_tokens = [..., x, 0, 0, 0, ..., 0, 0, 0, y, ...]
  new_text_tokens = _get_new_text_tokens(
      mm_start=mm_start,
      text_tokens=tokens,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Get the mm tokens placeholders, correctly placed at their final position.
  # new_mm_tokens = [
  #     ..., 0, 0, \n\n, <start_of_image>, ..., <end_of_image>, \n\n, 0, 0, ...
  # ]
  new_mm_tokens = _get_new_mm_tokens(
      mm_start=mm_start,
      mm_tokens_to_insert=mm_tokens,
      max_num_images=max_num_images,
      offset_by=offset_by,
      length_with_mm=length_with_mm,
  )

  # Merge the text and MM tokens.
  return new_text_tokens + new_mm_tokens


def _get_new_text_tokens(
    *,
    mm_start: Bool[ArrayT, 'B L'],
    text_tokens: Int[ArrayT, 'B L'],
    offset_by: int,
    length_with_mm: int,
) -> Int[ArrayT, 'B max_num_images num_tokens_per_image+4']:
  """Utility function for _insert_sequence."""
  # Jax vmap does not support positional arguments, so need the
  # _get_new_text_tokens_inner indirection.
  return jax.vmap(_get_new_text_tokens_inner, in_axes=(0, 0, None, None))(
      mm_start, text_tokens, offset_by, length_with_mm
  )


def _get_new_text_tokens_inner(
    mm_start: Bool[ArrayT, 'B L'],
    text_tokens: Int[ArrayT, 'B L'],
    offset_by: int,
    length_with_mm: int,
) -> Int[ArrayT, 'L']:
  """`_get_new_text_tokens_positions` without batch dimension."""

  # Empty buffer in which text and MM tokens will be inserted.
  tokens_with_mm = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  # Shift the original tokens, so that the new soft tokens can be inserted.
  new_text_tokens_pos = _get_new_text_tokens_positions(
      offset_on=mm_start,
      offset_by=offset_by,
  )

  tokens_with_mm = tokens_with_mm.at[new_text_tokens_pos].set(text_tokens)

  # Remove the `<start_of_image>` tokens (will be added afterwards when
  # merging with `_get_new_mm_tokens`).
  first_mm_pos = tokens_with_mm[0]
  new_start_mm_pos = new_text_tokens_pos * mm_start
  tokens_with_mm = tokens_with_mm.at[new_start_mm_pos].set(0)
  tokens_with_mm = tokens_with_mm.at[0].set(first_mm_pos)

  return tokens_with_mm


def _get_new_text_tokens_positions(
    *,
    offset_on: Bool[ArrayT, 'L'],
    offset_by: int,
) -> Int[ArrayT, 'L']:
  """Create the positions of the new tokens.

  Input: `[x, x, x, offset_on, x, x, offset_on, x]`
  Output: `[0, 1, 2, 3, 4+Offset, 5+Offset, 6+Offset, 7+Offset^2]`

  Args:
    offset_on: The token to offset on.
    offset_by: The number of tokens to offset by.

  Returns:
    The new positions of the tokens.
  """
  offset = jnp.cumsum(offset_on, axis=-1) * offset_by
  new_positions = jnp.arange(offset_on.shape[-1]) + offset
  # Do not shift the `<start_of_image>` token, it will be overwritten by the MM
  # tokens.
  new_positions -= offset_by * offset_on
  return new_positions


def _get_new_mm_tokens(
    *,
    mm_start: Bool[ArrayT, 'B L'],
    mm_tokens_to_insert: Int[ArrayT, 'num_tokens_per_image+4'],
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> Int[ArrayT, 'B max_num_images num_tokens_per_image+4']:
  """Utility function for _insert_sequence."""
  # Jax vmap does not support positional arguments, so need the
  # _get_new_mm_tokens_inner indirection.
  return jax.vmap(
      _get_new_mm_tokens_inner, in_axes=(0, None, None, None, None)
  )(mm_start, mm_tokens_to_insert, max_num_images, offset_by, length_with_mm)


def _get_new_mm_tokens_inner(
    mm_start: Bool[ArrayT, 'L'],
    mm_tokens_to_insert: Int[ArrayT, 'num_tokens_per_image+4'],
    max_num_images: int,
    offset_by: int,
    length_with_mm: int,
) -> Int[ArrayT, 'max_num_images num_tokens_per_image+4']:
  """`_get_new_mm_tokens` without batch dimension."""
  # Empty buffer row, which will be merged with the final tokens.
  row = jnp.zeros((length_with_mm,), dtype=jnp.int32)

  ones = jnp.ones((len(mm_tokens_to_insert),), dtype=jnp.int32)

  (offset,) = jnp.nonzero(mm_start, size=max_num_images)

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  mask = offset != 0
  mask = jnp.einsum('...x,y->xy', mask, ones)

  # After the mask is created, offset each individual images
  offset += jnp.arange(len(offset)) * offset_by

  new_positions = jnp.einsum('x,y->xy', offset, ones)
  new_positions += jnp.arange(len(mm_tokens_to_insert))

  new_positions = new_positions * mask

  # Because not all elements in the batch do have the same number of images
  # we need to mask out the `offset == 0` values.
  # This means that `<start_of_images>` can never be the first token, but this
  # should never happen in practice as sequences should start by `<bos>`
  row = row.at[new_positions].set(mm_tokens_to_insert)
  row = row.at[0].set(0)
  return row


def _build_token_positions_from_mask(
    input_mask: Bool[ArrayT, 'B L'],
) -> Int[ArrayT, 'B L']:
  """Utility function for build_token_positions.

  Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  # The padded tokens are arbitrary position, but should not matter.
  # Could multiply the result by `input_mask` to set the position to 0.
  return positions - (positions >= 1)


def _make_causal_bidirectional_attention_mask(
    causal_mask: Bool[ArrayT, 'B L'],
    *,
    bidirectional_mask: Union[Bool[ArrayT, 'B L'], None] = None,
) -> Bool[ArrayT, 'B L L']:
  """Utility function for build_attention_mask.

  Make the attention mask for the transformer.

  Gemma transformer attention mask is a little complicated, as the text
  uses causal attention, while the images use bidirectional attention.

  Examples:

  ```python
  causal_mask =        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  bidirectional_mask = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

  attention_mask = [
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
  ]
  ```

  Args:
    causal_mask: The causal mask (to mask out future and padding tokens).
    bidirectional_mask: The bidirectional mask (location of the soft images
      tokens).

  Returns:
    The attention mask.
  """

  attention_mask = _make_causal_mask(causal_mask)

  # Add the bidirectional mask for images.
  if bidirectional_mask is not None:
    attention_mask = _add_bidirectional_mask(attention_mask, bidirectional_mask)

  return attention_mask


def _make_causal_mask(
    input_mask: Bool[ArrayT, 'B L'],
) -> Bool[ArrayT, 'B L L']:
  """Makes a causal attention mask.

  I.e., as in middle diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def _make_block_mask_indices(
    bidirectional_mask: Bool[ArrayT, 'B L'],
) -> Int[ArrayT, 'B L']:
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  # Left pad 0.
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _add_bidirectional_mask(
    attn_mask: Bool[ArrayT, 'B L L'],
    bidirectional_mask: Bool[ArrayT, 'B L'],
) -> Bool[ArrayT, 'B L L']:
  """Adds bidirectional mask to the attention mask."""
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  attn_mask = attn_mask | (
      (kv_block_indices[:, None, :] == q_block_indices[..., None])
      & (q_block_indices[..., None] > 0)
  )
  return attn_mask
