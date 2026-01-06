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

"""Useful layers for Vision Transformers.

These layers may also be useful for Transformer language models, but not
implemented in penzai.

normal_initializer is a normal initializer for linear and affine layers.

SplitAxes is used to split a single axis of an input into multipleaxes.
MergeAxes is used to merge multiple axes of an input into a single axis.

ApplySinCos2dPE is used to apply 2D sine-cosine positional embeddings to an
input, normally 2D images. ApplyLearnablePE is used to apply learnable
positional embeddings to an input, similar as AddBias.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.core import named_axes
from penzai.core import struct
from penzai.core import variables
from penzai.nn import layer as layer_base
from penzai.nn import linear_and_affine
from penzai.nn import parameters

check_unique_axis_names_for_initializer = (
    linear_and_affine.check_unique_axis_names_for_initializer
)
LinearOperatorWeightInitializer = (
    linear_and_affine.LinearOperatorWeightInitializer
)
NamedArray = named_axes.NamedArray
Parameter = variables.Parameter
ParameterValue = variables.ParameterValue


def normal_initializer(
    stddev: float | jnp.ndarray,
) -> LinearOperatorWeightInitializer:
  """Returns an initializer that uses a normal distribution."""

  def _initializer(
      key: jax.Array,
      *,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      parallel_axes: dict[str, int],
      convolution_spatial_axes: dict[str, int],
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    """Normal initializer for named arrays."""
    check_unique_axis_names_for_initializer(
        input_axes, output_axes, parallel_axes, convolution_spatial_axes
    )

    shape = []
    names = []
    input_axis_indices = []
    output_axis_indices = []
    parallel_axis_indices = []

    for name, size in parallel_axes.items():
      parallel_axis_indices.append(len(shape))
      names.append(name)
      shape.append(size)

    for name, size in convolution_spatial_axes.items():
      names.append(name)
      shape.append(size)

    for name, size in input_axes.items():
      input_axis_indices.append(len(shape))
      names.append(name)
      shape.append(size)

    for name, size in output_axes.items():
      output_axis_indices.append(len(shape))
      names.append(name)
      shape.append(size)

    array = jax.random.normal(key, shape, dtype=dtype) * jnp.array(
        stddev, dtype
    )
    return named_axes.wrap(array).tag(*names)

  return _initializer


@struct.pytree_dataclass
class MergeAxes(layer_base.Layer):
  """Merge axes of an input into a single axis.

  Attributes:
    input_axes: A tuple of input axis names to be merged or a dictionary of
      input axis names to their sizes.
    output_axes: The output axis name, or a dictionary of out axis name to its
      size.
  """

  input_axes: tuple[str, ...] | dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  output_axes: str | tuple[str, ...] | dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(self, value: NamedArray, **_unused_side_inputs) -> NamedArray:
    # Check input axes exist in the input and compute the output dimension.
    output_dim = 1
    if isinstance(self.input_axes, tuple):
      input_axis_names = self.input_axes
    else:
      input_axis_names = self.input_axes.keys()
    for axis_name in input_axis_names:
      assert (
          axis_name in value.named_shape
      ), f"Axis {axis_name} not found in input {value.named_shape}."
      output_dim *= value.named_shape[axis_name]
    # Check output axes.
    if isinstance(self.output_axes, str):
      output_axis_name = self.output_axes
    elif isinstance(self.output_axes, tuple):
      output_axis_name = self.output_axes[0]
      assert len(self.output_axes) == 1, "Only one output axis is supported."
    else:
      output_axis_name = list(self.output_axes.keys())[0]
      assert len(self.output_axes) == 1, "Only one output axis is supported."
    return named_axes.nmap(jnp.reshape)(
        value.untag(*input_axis_names), (output_dim,)
    ).tag(output_axis_name)


@struct.pytree_dataclass
class SplitAxes(layer_base.Layer):
  """Split a single axis of an input into axes.

  Attributes:
    input_axes: The input axis name or a dictionary of input axis names to be
      split.
    output_axes: A dictionary mapping output axis names to their sizes.
  """

  input_axes: str | tuple[str, ...] | dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  output_axes: dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(self, value: NamedArray, **_unused_side_inputs) -> NamedArray:
    # Check input axes exist in the input.
    if isinstance(self.input_axes, str):
      input_axis_name = self.input_axes
    elif isinstance(self.input_axes, tuple):
      input_axis_name = self.input_axes[0]
      assert len(self.input_axes) == 1, "Only one input axis is supported."
    else:
      input_axis_name = list(self.input_axes.keys())[0]
      assert len(self.input_axes) == 1, "Only one input axis is supported."
    assert (
        input_axis_name in value.named_shape
    ), f"Axis {input_axis_name} not found in input {value.named_shape}."
    return named_axes.nmap(jnp.reshape)(
        value.untag(input_axis_name), tuple(self.output_axes.values())
    ).tag(*self.output_axes.keys())


@struct.pytree_dataclass
class ApplySinCos2dPE(layer_base.Layer):
  """Applies 2D sine-cosine positional embeddings to an input.

  This layer generates and applies 2D sine-cosine positional embeddings,
  typically used in Vision Transformers.

  Attributes:
    embedding_axis: The axis name of the input that contains the embedding
      vector (e.g. "embedding" or "channels").
    height_axis: The axis name corresponding to the height dimension of the 2D
      input (e.g., "height" or "h").
    width_axis: The axis name corresponding to the width dimension of the 2D
      input (e.g., "width" or "w").
    sequence_axis: The name of the sequence axis after flattening height and
      width (e.g., "seq" or "patches").
    wavelength: The wavelength parameter for the sine-cosine positional
      embeddings, controlling the frequency range.
  """

  embedding_axis: str = dataclasses.field(
      default="embedding", metadata={"pytree_node": False}
  )
  height_axis: str = dataclasses.field(
      default="height", metadata={"pytree_node": False}
  )
  width_axis: str = dataclasses.field(
      default="width", metadata={"pytree_node": False}
  )
  sequence_axis: str = dataclasses.field(
      default="patch", metadata={"pytree_node": False}
  )
  wavelength: float = dataclasses.field(
      default=10_000.0, metadata={"pytree_node": False}
  )

  def __call__(
      self, inputs: named_axes.NamedArray, **_unused_side_inputs
  ) -> named_axes.NamedArray:
    if self.embedding_axis not in inputs.named_shape:
      raise ValueError(
          f"Input must have an embedding axis '{self.embedding_axis}'."
      )
    if self.height_axis not in inputs.named_shape:
      raise ValueError(f"Input must have a height axis '{self.height_axis}'.")
    if self.width_axis not in inputs.named_shape:
      raise ValueError(f"Input must have a width axis '{self.width_axis}'.")
    if self.sequence_axis in inputs.named_shape:
      raise ValueError(
          "Input should not already have a sequence axis"
          f" '{self.sequence_axis}'."
      )

    h = inputs.named_shape[self.height_axis]
    w = inputs.named_shape[self.width_axis]
    embedding_dim = inputs.named_shape[self.embedding_axis]
    dtype = inputs.dtype

    # Borrow code from Gemma:
    # https://github.com/google-deepmind/gemma/blob/main/gemma/multimodal/vision_utils.py
    if embedding_dim % 4 != 0:
      raise ValueError(
          "Embedding dimension must be a multiple of 4 for sincos positional"
          " embeddings."
      )

    y, x = jnp.mgrid[:h, :w]
    omega = jnp.arange(w // 4) / (w // 4 - 1)
    omega = 1.0 / (self.wavelength**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)

    pe = jnp.concatenate(
        [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1
    )
    pe = jnp.asarray(pe, dtype=dtype)

    # Create the positional embedding as a NamedArray with the sequence axis
    positional_embedding = named_axes.NamedArray(
        pe, (self.sequence_axis, self.embedding_axis)
    )

    # Reshape the input from (B, H, W, D) to (B, H*W, D)
    # First, rearrange axes to bring H and W together
    inputs = inputs.untag(self.height_axis, self.width_axis)
    reshaped_inputs = named_axes.nmap(jnp.reshape)(inputs, (h * w,))
    reshaped_inputs = reshaped_inputs.tag(self.sequence_axis)

    # Add positional embedding to the reshaped input.
    # NamedArray's addition handles broadcasting over the batch axis.
    return reshaped_inputs + positional_embedding


@struct.pytree_dataclass
class ApplyLearnablePE(layer_base.Layer):
  """Applies learnable positional embeddings to an input, similar as AddBias.

  This layer uses named arrays to automatically apply across the correct
  set of dimensions.

  Attributes:
    pos_embedding: The learnable positional embedding parameter.
    new_axis_names: The new axes in the output that we do not expect to see in
      the input.
  """

  pos_embedding: parameters.ParameterLike[NamedArray]
  new_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(self, value: NamedArray, **_unused_side_inputs) -> NamedArray:
    """Adds a learned bias to the value."""
    # Elementwise functions broadcast automatically
    return value + self.pos_embedding.value

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      positional_axes: dict[str, int],
      new_output_axes: dict[str, int] | None = None,
      initializer: LinearOperatorWeightInitializer = linear_and_affine.zero_initializer,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> ApplyLearnablePE:
    """Constructs an ``ApplyLearnablePE`` layer from a configuration.

    Args:
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      positional_axes: Names and lengths for the axes in the input that the
        positional embeddings should act over. Other axes will be broadcast
        over.
      new_output_axes: Names and lengths of new axes that should be introduced
        into the input.
      initializer: Function to use to initialize the weight. Only the output
        axes will be set.
      dtype: Dtype for the learnable positional embeddings.

    Returns:
      A new ``ApplyLearnablePE`` layer with an uninitialized positional
      embedding parameter.
    """
    if new_output_axes is None:
      new_output_axes = {}

    return cls(
        pos_embedding=parameters.make_parameter(
            f"{name}/pos_embedding",
            init_base_rng,
            initializer,
            input_axes={},
            output_axes={**positional_axes, **new_output_axes},
            parallel_axes={},
            convolution_spatial_axes={},
            dtype=dtype,
        ),
        new_axis_names=tuple(new_output_axes.keys()),
    )


@pz.pytree_dataclass
class StopGradient(pz.nn.Layer):
  """Stop gradient."""

  def __call__(
      self, value: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:
    return named_axes.nmap(jax.lax.stop_gradient)(value)
