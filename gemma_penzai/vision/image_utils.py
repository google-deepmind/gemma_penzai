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

"""Utilities for image processing and patch extraction.

This module provides functions for:
  * Image Preprocessing: Resizing, normalizing, and clipping images.
  * Patch Extraction: Extracting patches from images using convolution.
  * Image Loading: Loading and processing image files from disk.
  * Image Processing: Processing jax arrays representing images.

The functions in this module are designed to be used as part of a vision
transformer model, and are optimized for use with JAX.

The module defines global constants for image mean and standard deviation,
and default image and patch sizes. These can be adjusted as needed.
"""

from collections.abc import Sequence
from typing import Union

import einops
from etils import epath
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
from PIL import Image


_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3
_DEFAULT_IMAGE_SIZE = 896  # SigLip expected input image size
_DEFAULT_PATCH_SIZE = 14   # SigLip expected patch size

Float = jaxtyping.Float
ArrayT = jaxtyping.Array
# A list of lists: (batch size) x (sequence size) of arrays (e.g., image arrays)
# or Nones.
BatchSeqArrayT = Sequence[Sequence[Union[ArrayT, None]]]
# A list of lists: (batch size) x (sequence size) of strings (e.g., image paths)
# or Nones.
BatchSeqStr = Sequence[Sequence[Union[str, None]]]


def _pre_process_image(
    image: Float[ArrayT, "H W C"],
    *,
    image_shape: tuple[int, int, int],
) -> Float[ArrayT, "H W C"]:
  """Pre-process image.

  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.

  Args:
    image: The image to pre-process.
    image_shape: The target shape (h, w, c) of the image (default to (896, 896,
      3)).

  Returns:
    The pre-processed image.
  """
  # TODO(epot): All inputs are expected to have been jpeg encoded with
  # TensorFlow.
  # tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)

  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image)
  image = jnp.clip(image, -1, 1)
  return image


def _patchify_images(
    images: Float[ArrayT, "B H W C"],
    *,
    patch_size: tuple[int, int],
    padding: str = "VALID",
) -> Float[ArrayT, "B P D"]:
  """Extract patches from images.

  This function is a wrapper for jax.lax.conv_general_dilated_patches
  to conform to the same interface as tf.image.extract_patches.
  The function extracts patches of shape sizes from the input images in the same
  manner as a convolution with kernel of shape sizes, stride equal to strides,
  and the given padding scheme.
  The patches are stacked in the channel dimension.

  Args:
    images: input batch of images of shape [B, H, W, C].
    patch_size: size of extracted patches.
    padding: padding algorithm to use.

  Returns:
    Tensor of shape [batch, num patches, patch_size * patch_size * C]
  """
  channels = images.shape[-1]
  patches = jax.lax.conv_general_dilated_patches(
      lhs=images,
      filter_shape=patch_size,
      window_strides=patch_size,
      padding=padding,
      rhs_dilation=[1, 1],
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
      precision=jax.lax.Precision.HIGH,
  )
  patches = einops.rearrange(
      patches, "b ph pw (c p) -> b (ph pw) (p c)", c=channels
  )
  return patches


def _normalize_images(
    images: Float[ArrayT, "H W C"],
) -> Float[ArrayT, "H W C"]:
  """Normalize the image to zero mean and unit variance.

  In order to change the image mean and std, we need to change the _IMAGE_MEAN
  and _IMAGE_STD global constants in this file.

  Args:
    images: The images to normalize.

  Returns:
    The normalized images.
  """
  images -= jnp.asarray(_IMAGE_MEAN)
  images /= jnp.asarray(_IMAGE_STD)
  return images


def load_and_process_image_files(
    img_paths: BatchSeqStr,
    image_resize_shape: tuple[int, int] = (
        _DEFAULT_IMAGE_SIZE,
        _DEFAULT_IMAGE_SIZE,
    ),
    patch_size: tuple[int, int] = (
        _DEFAULT_PATCH_SIZE,
        _DEFAULT_PATCH_SIZE,
    ),
) -> Union[Float[ArrayT, "B P D"], None]:
  """Loads image files.

  Args:
    img_paths: A list of list of image paths. Each element in the list is a list
      of image paths. We use a list of lists since we want to support batching
      (first list) and multiple images per sample (second list).
    image_resize_shape: The shape to resize the images to.
    patch_size: The size of the patches.

  Returns:
    The patches of the images of shape [batch size, num images, num patches,
    patch size * patch size * channels]
  """
  if len(img_paths) == 1 and len(img_paths[0]) == 1 and img_paths[0][0] is None:
    return None
  patches = []
  for imgs_path in img_paths:
    tmp = []
    for img_path in imgs_path:
      if img_path is None:
        raise ValueError(
            "some img_paths are None and some are not. we only support all None"
            " or all not None for now."
        )
      with epath.Path(img_path).open("rb") as f:
        img = _pre_process_image(
            np.array(Image.open(f).convert("RGB")),
            image_shape=image_resize_shape + (3,),
        )
      tmp.append(_patchify_images(img[None, ...], patch_size=patch_size))
    patches.append(jnp.concatenate(tmp, axis=0))
  patches = jnp.asarray(patches)
  return patches


def process_images(
    images: BatchSeqArrayT,
    image_resize_shape: tuple[int, int] = (
        _DEFAULT_IMAGE_SIZE,
        _DEFAULT_IMAGE_SIZE,
    ),
    patch_size: tuple[int, int] = (
        _DEFAULT_PATCH_SIZE,
        _DEFAULT_PATCH_SIZE,
    ),
) -> Union[Float[ArrayT, "B P D"], None]:
  """Loads image files.

  Args:
    images: A list of list of images. Each element in the list is a list of
      images with shape [H, W, C]. We use a list of lists since we want to
      support batching (first list) and multiple images per sample (second
      list).
    image_resize_shape: The shape to resize the images to.
    patch_size: The size of the patches.

  Returns:
    The patches of the images of shape [batch size, num images, num patches,
    patch size * patch size * channels]
  """
  if len(images) == 1 and len(images[0]) == 1 and images[0][0] is None:
    return None
  patches = []
  for imgs in images:
    tmp = []
    for img in imgs:
      if img is None:
        raise ValueError(
            "some imgs are None and some are not. we only support all None"
            " or all not None for now."
        )
      img = _pre_process_image(img, image_shape=image_resize_shape + (3,))
      tmp.append(_patchify_images(img[None, ...], patch_size=patch_size))
    patches.append(jnp.concatenate(tmp, axis=0))
  patches = jnp.asarray(patches)
  return patches
