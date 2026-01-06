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

"""Core components and layers for Vision Transformers (ViTs).

Specific instantiations of the VisionTransformer model will use the following
axis naming conventions:

* "height", "width", and "channels" axes: Dimensions of the image pixels.

* "patch" is the temporal axis of the patches sequence after images are
converted to patches. Similar to "seq" in the TransformerLM models.

* "embedding" axis: Dimension of the embedding vectors and residual stream.

* "projection" is the axis for query, key, and value head projection vectors,
  i.e. the axis where query-key dot products happen, and for which the values of
  attention heads are retrieved.

* "heads" is the axis for attention heads, VisionTransformer always uses full
  multi-head attention, the "heads" axis appears in queries, keys, and values.

* "neurons" is the axis for the neurons in the MLP blocks, which have an
  activation function applied elementwise and therefore have a
  privileged basis.

Different from the TransformerLM models, the VisionTransformer models do not
use KV Caching logic.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from penzai import pz
from penzai.core import named_axes


@dataclasses.dataclass
class VisionTransformerMetadata:
  """Common axis sizes and other information for transformer models.

  These values are kept on the main transformer object to simplify model
  transformations that depend on axis sizes or dtypes, by making it possible
  to infer the shape of intermediate activations in advance.

  Attributes:
    common_head_axes: A map of axis names to sizes for head axes that are common
      to queries, keys, and values.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    num_patches_per_image: Number of patches per image in the attention
      sequence.
    parameter_dtype: Floating dtype to use for activations and parameters.
  """

  common_head_axes: dict[str, int]
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  num_patches_per_image: int
  parameter_dtype: jax.typing.DTypeLike


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)  # pytype: disable=wrong-keyword-args  # pylint: disable=line-too-long
class VisionTransformerFeedForward(pz.nn.Sequential):
  """Informatively-named Sequential subclass for feedforward/MLP layers."""


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)  # pytype: disable=wrong-keyword-args  # pylint: disable=line-too-long
class VisionTransformerBlock(pz.nn.Sequential):
  """Informatively-named Sequential subclass for the main ViT blocks."""

  def treescope_color(self):
    color = "oklch(0.785 0.103 186.9 / 1.0)"
    return color, f"color-mix(in oklab, {color} 25%, white)"


@pz.pytree_dataclass
class VisionTransformer(pz.nn.Layer):
  """Top-level vision transformer encoder wrapper.

  This class is a simple wrapper that holds configuration data and runs safety
  checks.

  Attributes:
    body: The implementation of the vision transformer.
    metadata: The axis size and dtype info for the transformer.
  """

  body: pz.nn.Layer
  metadata: VisionTransformerMetadata = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(
      self,
      images: pz.nx.NamedArray,
      **side_inputs,
  ) -> pz.nx.NamedArray:
    """Vision transformer forward pass.

    Args:
      images: Array of images, as a float named array with a "height" and
        "width" axis and possibly batch (or frame) axes.
      **side_inputs: Side inputs, which will be forwarded to the body.

    Returns:
      A sequence of image embeddings processed by the vision transformer.
    """
    return self.body(images, **side_inputs)


@pz.pytree_dataclass
class VisionExit(pz.nn.Layer):
  """The vision exit layer.

  This is typically used in the last block of the vision transformer. For
  Gemma3, this layer is used to downsample image tokens to required length.

  Attributes:
    output_length: The output length of the vision exit layer.
    across: The axis name of the input that contains the sequence.

  TODO(xiangming): Add other exit layers, such as classification head, etc.
  """

  output_length: int = dataclasses.field(
      default=256, metadata={"pytree_node": False}
  )
  across: str = dataclasses.field(
      default="patch", metadata={"pytree_node": False}
  )

  def __call__(
      self, value: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:

    assert (
        self.across in value.named_shape
    ), f"Axis {self.across} not found in input {value.named_shape}."
    cur_length = value.named_shape[self.across]
    if cur_length == self.output_length:
      return value

    cur_width = int(cur_length**0.5)
    assert (
        cur_width**2 == cur_length
    ), f"Current patch size {cur_length} is not a square root of an integer."
    output_width = int(self.output_length**0.5)
    assert output_width**2 == self.output_length, (
        f"Expected output length {self.output_length} is not a square root of"
        " an integer."
    )
    # re-arrange "b (h w) d" -> "b h w d"
    value = named_axes.nmap(jnp.reshape)(
        value.untag(self.across), (cur_width, cur_width)
    )

    # average pooling over patches
    if cur_width % output_width:
      raise ValueError(
          f"Current patch size {cur_width} is not divisible by output width"
          f" {output_width}."
      )
    window = cur_width // output_width
    window_shape = (window, window)
    # A manual implementation of the nn.avg_pool layer borrowed from Flax:
    # https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/pooling.html#avg_pool
    value = named_axes.nmap(jax.lax.reduce_window)(
        value,
        0.0,
        jax.lax.add,
        window_shape,
        window_shape,
        "VALID",
    )
    value = value / np.prod(window_shape)
    # re-arrange "b h w d" -> "b (h w) d"
    value = named_axes.nmap(jnp.reshape)(value, (self.output_length,)).tag(
        self.across
    )
    return value
