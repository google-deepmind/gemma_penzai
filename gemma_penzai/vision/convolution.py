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

"""Definition of convolution layer and associated utilities.

Code is adapted based on the following PR:
https://github.com/google-deepmind/penzai/pull/120
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Sequence, cast

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import shapecheck
from penzai.core import struct
from penzai.core import variables
from penzai.nn import grouping
from penzai.nn import layer as layer_base
from penzai.nn import linear_and_affine
from penzai.nn import parameters


RenameAxes = linear_and_affine.RenameAxes
LinearOperatorWeightInitializer = (
    linear_and_affine.LinearOperatorWeightInitializer
)
xavier_uniform_initializer = linear_and_affine.xavier_uniform_initializer
NamedArray = named_axes.NamedArray
Parameter = variables.Parameter
ParameterValue = variables.ParameterValue


@struct.pytree_dataclass
class ConvInPlace(grouping.Sequential):
  """Container for "in-place" convolution operators that preserve axis names.

  This is used when initializing `Conv` layers that have overlapping names in
  their input and output specifications. We subclass `Sequential` to make
  this layer type easier to identify and manipulate.
  """

  sublayers: list[layer_base.Layer]

  def treescope_color(self) -> tuple[str, str]:
    return "#79eb75", "color-mix(in oklab, #79eb75 25%, white)"


@struct.pytree_dataclass
class ConvTransposeInPlace(grouping.Sequential):
  """Container for "in-place" transposed convolution operators that preserve axis names.

  This is used when initializing `ConvTranspose` layers that have overlapping
  names in their input and output specifications. We subclass `Sequential` to
  make this layer type easier to identify and manipulate.
  """

  sublayers: list[layer_base.Layer]

  def treescope_color(self) -> tuple[str, str]:
    return "#c7eb75", "color-mix(in oklab, #c7eb75 25%, white)"


def contract(
    names: str | Sequence[named_axes.AxisName],
    left: NamedArray, # type: ignore
    right: NamedArray, # type: ignore
) -> NamedArray: # type: ignore
  """Contracts two named arrays along the given axis names.

  Args:
    names: The axis names to contract. Can also be a single string axis name.
    left: The left-hand side of the contraction, as a NamedArray with only named
      axes.
    right: The right-hand side of the contraction, as a NamedArray with only
      named axes.

  Returns:
    Result of the contraction, which will have all names present in either
    `left` or `right`, except for the names in `names` which will be contracted
    away.
  """
  names = (names,) if isinstance(names, str) else names
  return named_axes.nmap(jnp.tensordot)(
      left.untag(*names), right.untag(*names), axes=len(names)
  )


def _maybe_rename_output_axes(
    input_axes: dict[str, int],
    output_axes: dict[str, int],
    parallel_axes: dict[str, int],
    parallel_broadcast_axes: dict[str, int],
    rename_outputs_if_necessary: bool,
):
  """Checks for name overlap between input and output axes, and renames if needed to avoid collisions.

  Args:
    input_axes: Names and lengths for axes that the linear operator should
      contract over.
    output_axes: Names and lengths for new axes that the linear operator should
      produce.
    parallel_axes: Names and lengths for axes that should be processed in
      parallel. These axes should appear in both the input and the output, and
      the resulting linear operator will apply a different operator to each
      slice. (This is similar to a block-diagonal matrix.)
    parallel_broadcast_axes: Names and lengths for axes that should be treated
      like `parallel_axes` but will only appear in the output. The input will be
      implicitly broadcast over these axes.
    rename_outputs_if_necessary: If True, renames output axes that overlap with
      input axes by appending "_out" to their names.

  Returns:
      A tuple containing:
          - The output axes after renaming.
          - A list of the primed names, if any renaming occurred.
          - A list of the original names, if any renaming occurred.
  """

  # By default no rename & no wrapping
  output_axes_after_rename = output_axes
  primed_names, original_names = None, None

  if any(name in input_axes for name in output_axes):
    # Name overlap!
    if rename_outputs_if_necessary:
      output_axes_after_rename = {}
      original_names = []
      primed_names = []

      for old_name in output_axes.keys():
        if old_name in input_axes:
          primed_name = old_name + "_out"
          if primed_name in input_axes:
            raise ValueError(
                f"Tried to rename {old_name} to {primed_name} to avoid a"
                " conflict, but both names are already in input_axes. Please"
                " rename axes manually to avoid this conflict."
            )
          original_names.append(old_name)
          primed_names.append(primed_name)
          output_axes_after_rename[primed_name] = output_axes[old_name]
        else:
          output_axes_after_rename[old_name] = output_axes[old_name]
    else:
      raise ValueError(
          "input_axes and output_axes must not overlap if"
          " rename_outputs_if_necessary is not set; got"
          f" input_axes={input_axes}, output_axes={output_axes}."
      )

  if set(parallel_axes).intersection(set(input_axes).union(output_axes)) or set(
      parallel_broadcast_axes
  ).intersection(set(input_axes).union(output_axes, parallel_axes)):
    raise ValueError(
        "parallel_axes and parallel_broadcast_axes must not overlap with"
        f" each other or with input/output axes; got input_axes={input_axes},"
        f" output_axes={output_axes}, parallel_axes={parallel_axes},"
        f" parallel_broadcast_axes={parallel_broadcast_axes}."
    )
  return output_axes_after_rename, primed_names, original_names


def _prepare_for_conv(
    inputs: NamedArray, # type: ignore
    kernel: NamedArray, # type: ignore
    spatial_axis_names: Sequence[str],
    in_axis_names: Sequence[str],
    out_axis_names: Sequence[str],
) -> tuple[NamedArray, NamedArray]: # type: ignore
  """Preprocess lhs and rhs for jax convolution operator.

  Merges the in axes of the inputs into a single in channel axis, and merges the
  out axes of the kernel into a single out channel axis. This is necessary to
  use the jax convolution operator, which expects the inputs to have a single
  in channel axis and the kernel to have a single out channel axis.

  Args:
    inputs: The input named array.
    kernel: The kernel named array.
    spatial_axis_names: Names of the spatial axes in the input and kernel.
    in_axis_names: Names of the input axes that will be contracted with the
      kernel.
    out_axis_names: Names of the output axes that will be produced by the
      convolution.
  Returns:
    A tuple of two named arrays. The first one is the conv input with the in
    axes merged into a single in channel axis. Its positional axis layout is
    [spatial_axes..., channel_axis]. The second one is the convolution kernel
    with the in axes merged into a single in channel axis and the out axes
    merged into a single out channel axis. Its positional axis layout is
    [spatial_axes..., in_channel_axis, out_channel_axis].
  """

  lhs = inputs
  rhs = kernel

  in_axis_name = named_axes.TmpPosAxisMarker()
  out_axis_name = named_axes.TmpPosAxisMarker()

  # merge in axes into one in channel axis for the inputs and the kernel
  lhs = lhs.untag(*in_axis_names).flatten().tag(in_axis_name)
  rhs = rhs.untag(*in_axis_names).flatten().tag(in_axis_name)

  # merge out axes into one out channels axis for jax convolution
  rhs = rhs.untag(*out_axis_names).flatten().tag(out_axis_name)

  # untag spatial axes
  lhs = lhs.untag(*spatial_axis_names, in_axis_name)
  rhs = rhs.untag(*spatial_axis_names, in_axis_name, out_axis_name)
  return lhs, rhs


def _get_named_axis_back_after_conv(
    result: NamedArray, # type: ignore
    spatial_axis_names: Sequence[str],
    out_axis_names: Sequence[str],
    out_axis_shape: Sequence[int],
) -> NamedArray: # type: ignore
  """Postprocess result from jax convolution operator.

  Restores the spatial axes and output axes to the result of the jax convolution
  operator. The spatial axes are tagged back, and the output axes are reshaped
  to the original shape and tagged back. It supposes that the result have a
  positional axis layout of [spatial_axes..., out_axis] with out_axis of
  size equals to the product of the dimensions in out_axis_shape. This is
  necessary to restore the desired shape of the output after the convolution
  operator has been applied, since the convolution operates on positional
  spatial axes and only outputs a single out_axis.

  Args:
    result: The result of the jax convolution operator.
    spatial_axis_names: Names of the spatial axes in the input and kernel.
    out_axis_names: Names of the output axes that will be produced by the
      convolution.
    out_axis_shape: The shape of the output axes, which will be used to reshape
      the result back to the original shape.
  Returns:
    A named array with the spatial axes and output axes tagged back, and the
    output axes reshaped to the original shape.
  """
  return (
      result.tag_prefix(*spatial_axis_names)
      .reshape(out_axis_shape)
      .tag(*out_axis_names)
  )


def _maybe_broadcast(value: int | Sequence[int], count: int) -> Sequence[int]:
  """Broadcasts a value to a sequence of the given count.

  If the value is an integer, it will be repeated `count` times.
  If the value is already a sequence, it will be returned as is.

  Args:
    value: The value to broadcast, either an integer or a sequence of integers.
    count: The number of times to repeat the value if it is an integer.
  Returns:
    A sequence of integers with the value repeated `count` times if it was an
    integer, or the original sequence if it was already a sequence.
  """

  if isinstance(value, int):
    return [value] * count
  else:
    assert (
        len(value) == count
    ), "If value is a sequence, it must match the count."
    return value


def _get_dimension_numbers(ndim) -> jax.lax.ConvDimensionNumbers:
  """Returns the dimension numbers for a convolution operator.

  Args:
    ndim: The number of spatial dimensions of the convolution operator.
  Returns:
    A `jax.lax.ConvDimensionNumbers` object that specifies the dimension numbers
    for the convolution operator. It assumes that the input and output have the
    following positional axis layout: [B, Spatial..., C] and the kernel has the
    following positional axis layout: [Spatial..., I, O], where B is the batch
    axis, C is the channel axis, I is the input channel axis, and O is the
    output channel axis. It matches the result of _prepare_for_conv.
  """

  return jax.lax.ConvDimensionNumbers(
      lhs_spec=(0, ndim + 1)
      + tuple(range(1, ndim + 1)),  # BCSpatial -> BCSpatial
      rhs_spec=(ndim + 1, ndim) + tuple(range(ndim)),  # SpatialIO -> OISpatial
      out_spec=(0, ndim + 1)
      + tuple(range(1, ndim + 1)),  # BSpatialC -> BCSpatial
  )


@struct.pytree_dataclass
class AbstractGeneralConv(layer_base.Layer):
  """Abstract base class for convolution operators, for named arrays.

  Applies an arbitrary convolution to the input `NamedArray` and a weight
  parameter. This can be used to express an arbitrary linear convolution
  operator.

  Attributes:
    kernel: The named array holding the kernel for the convolution operator.
    strides: The stride of the convolution operator
    padding: The padding to apply to the input before the convolution
    inputs_dilation: The input dilation of the convolution
    kernel_dilation: The kernel dilation of the convolution
    spatial_axis_names: The names of the spatial axes over which to
      apply the convolution operator
    in_axis_names: The names of the axes to contract with the input, removing
      them.
    out_axis_names: The names of the axes that should not appear in the input
      and will be inserted into the output.
  """
  kernel: parameters.ParameterLike[NamedArray] # type: ignore
  strides: Sequence[int] = dataclasses.field(metadata={"pytree_node": False})
  padding: str | Sequence[tuple[int, int]] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  spatial_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  in_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  out_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  kernel_dilation: Sequence[int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  inputs_dilation: Sequence[int] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(self, in_array: NamedArray, **_side_inputs) -> NamedArray: # type: ignore
    """Runs the Convolution operator."""
    in_struct = self._input_structure()

    # pytype: disable=attribute-error
    if isinstance(
        self.kernel,
        Parameter | ParameterValue,
    ) and self.kernel.label.endswith(".kernel"):
      error_prefix = f"({self.kernel.label[: 7]}) "
    else:
      error_prefix = ""
    # pytype: enable=attribute-error

    dimvars = shapecheck.check_structure(
        in_array, in_struct, error_prefix=error_prefix
    )

    lhs, rhs = _prepare_for_conv(
        in_array,
        self.kernel.value,
        self.spatial_axis_names,
        self.in_axis_names,
        self.out_axis_names,
    )

    if self._is_transposed():
      # Perform actual transposed convolution
      result = named_axes.nmap(
          lambda lhs, rhs: jax.lax.conv_transpose(
              lhs=lhs[None, ...],
              rhs=rhs,
              strides=self.strides,
              padding=self.padding,
              rhs_dilation=self.kernel_dilation,
              dimension_numbers=_get_dimension_numbers(
                  ndim=len(self.spatial_axis_names)
              ),
          )[0]
      )(lhs, rhs)
    else:
      # Perform actual convolution
      result = named_axes.nmap(
          lambda lhs, rhs: jax.lax.conv_general_dilated(
              lhs=lhs[None, ...],
              rhs=rhs,
              window_strides=self.strides,
              padding=self.padding,
              lhs_dilation=self.inputs_dilation,
              rhs_dilation=self.kernel_dilation,
              dimension_numbers=_get_dimension_numbers(
                  ndim=len(self.spatial_axis_names)
              ),
          )[0]
      )(lhs, rhs)

    result = _get_named_axis_back_after_conv(
        result,
        self.spatial_axis_names,
        self.out_axis_names,
        [self.output_axes[name] for name in self.out_axis_names],
    )

    out_struct = self._output_structure()
    shapecheck.check_structure(
        result, out_struct, known_vars=dimvars, error_prefix=error_prefix
    )
    return result

  @classmethod
  def _from_config(
      cls,
      inplace_class: type[ConvInPlace | ConvTransposeInPlace],
      name: str,
      init_base_rng: jax.Array | None,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      convolution_spatial_axes: dict[str, int],
      strides: int | Sequence[int] = 1,
      padding: str | Sequence[tuple[int, int]] = "SAME",
      inputs_dilation: int | Sequence[int] = 1,
      kernel_dilation: int | Sequence[int] = 1,
      parallel_axes: dict[str, int] | None = None,
      parallel_broadcast_axes: dict[str, int] | None = None,
      initializer: LinearOperatorWeightInitializer = xavier_uniform_initializer,
      dtype: jax.typing.DTypeLike = jnp.float32,
      rename_outputs_if_necessary: bool = True,
  ) -> AbstractGeneralConv | ConvInPlace | ConvTransposeInPlace:
    """Constructs a ``AbstractGeneralConv`` layer from a configuration.

    This can be used when building a new convolution or transposed convolution
    operator at the start of training. For more details see Conv or
    ConvTranspose.

    Args:
      inplace_class: The class to use when in place operation is needed.
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      input_axes: Names and lengths for axes that the linear operator should
        contract over.
      output_axes: Names and lengths for new axes that the linear operator
        should produce. If any axis names overlap with ``input_axes``, the
        argument ``rename_outputs_if_necessary`` must be True.
      convolution_spatial_axes: Names and lengths for the spatial axes of the
        convolution kernel.
      strides: strides of the convolution, if strides is an integer, it is
        broadcasted to every spatial dimensions
      padding: The padding to apply to the input before the convolution. Can be
        either the strings ‘SAME’, ‘SAME_LOWER’, or ‘VALID’, or a sequence
        of n (low, high) integer pairs that give the padding to apply before and
        after each spatial dimension. ‘SAME’ and ‘SAME_LOWER’ add padding to
        produce same output size as the input when the stride is one. The
        padding is split between the two sides equally or almost equally. In
        case the padding is an odd number, the extra padding is added at the end
        for ‘SAME’ and at the
        beginning for ‘SAME_LOWER’.
      inputs_dilation: inputs dilation of the convolution, if inputs_dilation is
        an integer, it is broadcasted to every spatial dimensions
      kernel_dilation: kernel dilation of the convolution, if kernel_dilation is
        an integer, it is broadcasted to every spatial dimensions
      parallel_axes: Names and lengths for axes that should be processed in
        parallel. These axes should appear in both the input and the output, and
        the resulting convolution operator will apply a different operator to
        each slice. (This is similar to a grouped convolution) Must not overlap
        with any axes named in ``input_axes`` or ``output_axes``.
      parallel_broadcast_axes: Names and lengths for axes that should be treated
        like ``parallel_axes`` but will only appear in the output. The input
        will be implicitly broadcast over these axes. Must not overlap with any
        axes named in ``input_axes``, ``output_axes`` or ``parallel_axes``.
      initializer: Function to use to initialize the kernel.
      dtype: Dtype for the kernel.
      rename_outputs_if_necessary: If True, and if ``output_axes`` and
        ``input_axes`` have overlapping names, avoids name conflicts by adding
        "primed" versions of the overlapping names, and returns an instance of
        `ConvInPlace` or `ConvTransposeInPlace` instead of a
        ``AbstractGeneralConv`` layer directly.
    """

    spatial_dim_count = len(convolution_spatial_axes)

    strides = _maybe_broadcast(strides, spatial_dim_count)
    inputs_dilation = _maybe_broadcast(inputs_dilation, spatial_dim_count)
    kernel_dilation = _maybe_broadcast(kernel_dilation, spatial_dim_count)

    if parallel_axes is None:
      parallel_axes = {}
    if parallel_broadcast_axes is None:
      parallel_broadcast_axes = {}

    output_axes_after_rename, primed_names, original_names = (
        _maybe_rename_output_axes(
            input_axes,
            output_axes,
            parallel_axes,
            parallel_broadcast_axes,
            rename_outputs_if_necessary,
        )
    )

    core_layer = cls(
        kernel=parameters.make_parameter(
            f"{name}.kernel",
            init_base_rng,
            initializer,
            input_axes=input_axes,
            output_axes=output_axes_after_rename,
            parallel_axes={**parallel_axes, **parallel_broadcast_axes},
            convolution_spatial_axes=convolution_spatial_axes,
            dtype=dtype,
        ),
        strides=strides,
        padding=padding,
        inputs_dilation=inputs_dilation,
        kernel_dilation=kernel_dilation,
        spatial_axis_names=tuple(convolution_spatial_axes.keys()),
        in_axis_names=tuple(input_axes.keys()),
        out_axis_names=(
            tuple(output_axes_after_rename.keys())
            + tuple(parallel_broadcast_axes.keys())
        ),
    )

    # if name overlap wrap layer
    if primed_names is not None and original_names is not None:
      return inplace_class(
          sublayers=[
              core_layer,
              RenameAxes(old=tuple(primed_names), new=tuple(original_names)),
          ],
      )

    return core_layer

  @abc.abstractmethod
  def _is_transposed(self) -> bool:
    ...

  def _input_structure(self):
    known_in_axes = {
        name: size
        for name, size in self.kernel.value.named_shape.items()
        if name not in self.out_axis_names
        and name not in self.spatial_axis_names
    }
    return shapecheck.ArraySpec(
        named_shape={**shapecheck.var("In"), **known_in_axes},
        dtype=jnp.floating,
    )

  def _output_structure(self):
    known_out_axes = {
        name: size
        for name, size in self.kernel.value.named_shape.items()
        if name not in self.in_axis_names
        and name not in self.spatial_axis_names
    }
    return shapecheck.ArraySpec(
        named_shape={**shapecheck.var("Out"), **known_out_axes},
        dtype=jnp.floating,
    )

  @property
  def input_axes(self) -> dict[str, int]:
    """The axis names and sizes that should appear in the input only."""
    return {  # pytype: disable=bad-return-type
        name: size
        for name, size in self.kernel.value.named_shape.items()
        if name in self.in_axis_names
    }

  @property
  def output_axes(self) -> dict[str, int]:
    """The axis names and sizes that will appear in the output only."""
    return {  # pytype: disable=bad-return-type
        name: size
        for name, size in self.kernel.value.named_shape.items()
        if name in self.out_axis_names
    }

  @property
  def parallel_axes(self) -> dict[str, int]:
    """The axis names and sizes that should appear in both input and output."""
    return {  # pytype: disable=bad-return-type
        name: size
        for name, size in self.kernel.value.named_shape.items()
        if name not in self.spatial_axis_names
        and name not in self.in_axis_names
        and name not in self.out_axis_names
    }

  @property
  def convolution_spatial_axes(self) -> dict[str, int]:
    """The spatial axis names and sizes of the convolution kernel.

    This should appear in both input and output.
    Note that that the sizes are only related to the kernel shape.
    """
    return {  # pytype: disable=bad-return-type
        name: size
        for name, size in self.kernel.value.named_shape.items()
        if name in self.spatial_axis_names
    }


@struct.pytree_dataclass
class Conv(AbstractGeneralConv):
  """A general convolution operator, for named arrays.

  Applies an arbitrary contraction to the input `NamedArray` and a weight
  parameter. This can be used to express an arbitrary linear convolution
  operator.

  Attributes:
    kernel: The named array holding the kernel for the convlution operator.
    strides: The stride of the convolution operator
    padding: The padding to apply to the input before the convolution
    inputs_dilation: The input dilation of the convolution
    kernel_dilation: The kernel dilation of the convolution
    convolution_spatial_axis_names: The names of the spatial axes over which to
      apply the convolution operator
    in_axis_names: The names of the axes to contract with the input, removing
      them.
    out_axis_names: The names of the axes that should not appear in the input
      and will be inserted into the output.
  """

  kernel: parameters.ParameterLike[NamedArray] # type: ignore
  strides: Sequence[int] = dataclasses.field(metadata={"pytree_node": False})
  padding: str | Sequence[tuple[int, int]] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  spatial_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  in_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  out_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  kernel_dilation: Sequence[int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  inputs_dilation: Sequence[int] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      convolution_spatial_axes: dict[str, int],
      strides: int | Sequence[int] = 1,
      padding: str | Sequence[tuple[int, int]] = "SAME",
      inputs_dilation: int | Sequence[int] = 1,
      kernel_dilation: int | Sequence[int] = 1,
      parallel_axes: dict[str, int] | None = None,
      parallel_broadcast_axes: dict[str, int] | None = None,
      initializer: LinearOperatorWeightInitializer = xavier_uniform_initializer,
      dtype: jax.typing.DTypeLike = jnp.float32,
      rename_outputs_if_necessary: bool = True,
  ) -> Conv | ConvInPlace:
    """Constructs a ``Conv`` layer from a configuration.

    This can be used when building a new convolution operator at the start of
    training.

    Note: For the purposes of the initializer, the ``parallel_axes`` and
    ``parallel_broadcast_axes`` are treated in the same way, without
    participating in output-dimension variance scaling. However, after
    initialization, the ``parallel_broadcast_axes`` will be treated like extra
    output axes (and assumed not to be present in the input).

    Args:
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      input_axes: Names and lengths for axes that the linear operator should
        contract over.
      output_axes: Names and lengths for new axes that the linear operator
        should produce. If any axis names overlap with ``input_axes``, the
        argument ``rename_outputs_if_necessary`` must be True.
      convolution_spatial_axes: Names and lengths for the spatial axes of the
        convolution kernel.
      strides: strides of the convolution, if strides is an integer, it is
        broadcasted to every spatial dimensions
      padding: The padding to apply to the input before the convolution. Can be
        either the strings ‘SAME’, ‘SAME_LOWER’, or ‘VALID’, or a sequence
        of n (low, high) integer pairs that give the padding to apply before and
        after each spatial dimension. ‘SAME’ and ‘SAME_LOWER’ add padding to
        produce same output size as the input when the stride is one. The
        padding is split between the two sides equally or almost equally. In
        case the padding is an odd number, the extra padding is added at the end
        for ‘SAME’ and at the
        beginning for ‘SAME_LOWER’.
      inputs_dilation: inputs dilation of the convolution, if inputs_dilation is
        an integer, it is broadcasted to every spatial dimensions
      kernel_dilation: kernel dilation of the convolution, if kernel_dilation is
        an integer, it is broadcasted to every spatial dimensions
      parallel_axes: Names and lengths for axes that should be processed in
        parallel. These axes should appear in both the input and the output, and
        the resulting convolution operator will apply a different operator to
        each slice. (This is similar to a grouped convolution) Must not overlap
        with any axes named in ``input_axes`` or ``output_axes``.
      parallel_broadcast_axes: Names and lengths for axes that should be treated
        like ``parallel_axes`` but will only appear in the output. The input
        will be implicitly broadcast over these axes. Must not overlap with any
        axes named in ``input_axes``, ``output_axes`` or ``parallel_axes``.
      initializer: Function to use to initialize the kernel.
      dtype: Dtype for the kernel.
      rename_outputs_if_necessary: If True, and if ``output_axes`` and
        ``input_axes`` have overlapping names, avoids name conflicts by adding
        "primed" versions of the overlapping names, and returns an instance of
        `ConvInPlace` instead of a ``Conv`` layer directly.

    Returns:
      A ``Conv`` layer with uninitialized kernel, or possibly a
      `ConvInPlace` layer if ``rename_outputs_if_necessary`` is True and
      ``input_axes`` overlaps with ``output_axes``.
    """

    layer = super()._from_config(
        inplace_class=ConvInPlace,
        name=name,
        init_base_rng=init_base_rng,
        input_axes=input_axes,
        output_axes=output_axes,
        convolution_spatial_axes=convolution_spatial_axes,
        strides=strides,
        padding=padding,
        inputs_dilation=inputs_dilation,
        kernel_dilation=kernel_dilation,
        parallel_axes=parallel_axes,
        parallel_broadcast_axes=parallel_broadcast_axes,
        initializer=initializer,
        dtype=dtype,
        rename_outputs_if_necessary=rename_outputs_if_necessary,
    )
    if isinstance(layer, AbstractGeneralConv):
      return cast(Conv, layer)
    assert isinstance(layer, ConvInPlace)
    return layer

  def _is_transposed(self):
    return False

  def treescope_color(self) -> str:
    return "#79eb75"


@struct.pytree_dataclass
class ConvTranspose(AbstractGeneralConv):
  """A general transposed convolution operator, for named arrays.

  Applies an arbitrary contraction to the input `NamedArray` and a kernel
  parameter. This can be used to express an arbitrary linear transposed
  convolution operator.

  Attributes:
    kernel: The named array holding the kernel for the convlution operator.
    strides: The stride of the convolution operator
    padding: The padding to apply to the input before the convolution
    kernel_dilation: The kernel dilation of the convolution
    convolution_spatial_axis_names: The names of the spatial axes over which to
      apply the convolution operator
    in_axis_names: The names of the axes to contract with the input, removing
      them.
    out_axis_names: The names of the axes that should not appear in the input
      and will be inserted into the output.
  """

  kernel: parameters.ParameterLike[NamedArray] # type: ignore
  strides: Sequence[int] = dataclasses.field(metadata={"pytree_node": False})
  padding: str | Sequence[tuple[int, int]] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  spatial_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  in_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  out_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  kernel_dilation: Sequence[int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  inputs_dilation: Sequence[int] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      convolution_spatial_axes: dict[str, int],
      strides: int | Sequence[int] = 1,
      padding: str | Sequence[tuple[int, int]] = "VALID",
      kernel_dilation: int | Sequence[int] = 1,
      parallel_axes: dict[str, int] | None = None,
      parallel_broadcast_axes: dict[str, int] | None = None,
      initializer: LinearOperatorWeightInitializer = xavier_uniform_initializer,
      dtype: jax.typing.DTypeLike = jnp.float32,
      rename_outputs_if_necessary: bool = True,
  ) -> ConvTranspose | ConvTransposeInPlace:
    """Constructs a ``Conv`` layer from a configuration.

    This can be used when building a new convolution operator at the start of
    training.

    Note: For the purposes of the initializer, the ``parallel_axes`` and
    ``parallel_broadcast_axes`` are treated in the same way, without
    participating in output-dimension variance scaling. However, after
    initialization, the ``parallel_broadcast_axes`` will be treated like extra
    output axes (and assumed not to be present in the input).

    Args:
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      input_axes: Names and lengths for axes that the linear operator should
        contract over.
      output_axes: Names and lengths for new axes that the linear operator
        should produce. If any axis names overlap with ``input_axes``, the
        argument ``rename_outputs_if_necessary`` must be True.
      convolution_spatial_axes: Names and lengths for the spatial axes of the
        convolution kernel.
      strides: strides of the convolution, if strides is an integer, it is
        broadcasted to every spatial dimensions
      padding: The padding to apply to the input before the convolution. Can be
        either the strings ‘SAME’, ‘SAME_LOWER’, or ‘VALID’, or a sequence
        of n (low, high) integer pairs that give the padding to apply before and
        after each spatial dimension. ‘SAME’ and ‘SAME_LOWER’ add padding to
        produce same output size as the input when the stride is one. The
        padding is split between the two sides equally or almost equally. In
        case the padding is an odd number, the extra padding is added at the end
        for ‘SAME’ and at the
        beginning for ‘SAME_LOWER’.
      kernel_dilation: kernel dilation of the convolution, if kernel_dilation is
        an integer, it is broadcasted to every spatial dimensions
      parallel_axes: Names and lengths for axes that should be processed in
        parallel. These axes should appear in both the input and the output, and
        the resulting convolution operator will apply a different operator to
        each slice. (This is similar to a grouped convolution) Must not overlap
        with any axes named in ``input_axes`` or ``output_axes``.
      parallel_broadcast_axes: Names and lengths for axes that should be treated
        like ``parallel_axes`` but will only appear in the output. The input
        will be implicitly broadcast over these axes. Must not overlap with any
        axes named in ``input_axes``, ``output_axes`` or ``parallel_axes``.
      initializer: Function to use to initialize the kernel.
      dtype: Dtype for the kernel.
      rename_outputs_if_necessary: If True, and if ``output_axes`` and
        ``input_axes`` have overlapping names, avoids name conflicts by adding
        "primed" versions of the overlapping names, and returns an instance of
        `ConvInPlace` instead of a ``Conv`` layer directly.

    Returns:
      A ``ConvTranspose`` layer with uninitialized kernel, or possibly a
      `ConvTransposeInPlace` layer if ``rename_outputs_if_necessary`` is True
      and ``input_axes`` overlaps with ``output_axes``.
    """
    layer = super()._from_config(
        inplace_class=ConvTransposeInPlace,
        name=name,
        init_base_rng=init_base_rng,
        input_axes=input_axes,
        output_axes=output_axes,
        convolution_spatial_axes=convolution_spatial_axes,
        strides=strides,
        padding=padding,
        kernel_dilation=kernel_dilation,
        inputs_dilation=1,  # not used for transposed convolutions
        parallel_axes=parallel_axes,
        parallel_broadcast_axes=parallel_broadcast_axes,
        initializer=initializer,
        dtype=dtype,
        rename_outputs_if_necessary=rename_outputs_if_necessary,
    )
    if isinstance(layer, AbstractGeneralConv):
      return cast(ConvTranspose, layer)

    assert isinstance(layer, ConvTransposeInPlace)
    return layer

  def _is_transposed(self):
    return True

  def treescope_color(self) -> str:
    return "#c7eb75"
