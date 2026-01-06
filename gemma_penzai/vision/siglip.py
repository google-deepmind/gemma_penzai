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

"""Vision Transformer (SigLip) used by Gemma3.

The architecture is based on the Gemma3 SigLip model, including a vision
transformer and a vision exit layer to reduce the vision embedding length.

Features of the architecture:

- Input to SigLiPFromPatches is patches, the model will first transform the
  input patches to images, then feed the images into the vision transformer
  to get the soft tokens.
- Learnable positional embeddings or sinusoidal positional embeddings can
  be added to the vision transformer.
- Full multi-head attention without causal mask.
- All linear layers and normalization layers have biases.

This file also supports loading Gemma3 SigLip only from Gemma3 checkpoint by
using the function `gemma_vision_from_pretrained_checkpoint`.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Literal

from gemma_penzai.vision import convolution
from gemma_penzai.vision import vision_layers
from gemma_penzai.vision import vision_parts
import jax
import jax.numpy as jnp
from penzai import pz


normal_initializer = vision_layers.normal_initializer


@dataclasses.dataclass
class Gemma3SigLipConfig:
  """Configuration parameters for Gemma3 SigLip.

  This config encompass the parameters for the Vision Transformer family, mainly
  used by Gemma3 models.

  Attributes:
    image_height: The height of the image.
    image_width: The width of the image.
    image_channels: The number of channels in the image.
    patch_size: The size of the patches.
    num_mm_tokens_per_image_prepool: The number of mm tokens per image before
      vision exit layer.
    num_mm_tokens_per_image: The number of mm tokens per image after vision exit
      layer.
    mlp_variant: The variant of the MLP to use.
    pos_embed: The type of positional embedding to use.
    embedding_dim: The embedding dimension.
    mlp_hidden_dim: The hidden dimension of the MLP.
    num_encoder_blocks: The number of encoder blocks.
    num_heads: The number of heads.
    layer_norm_eps: The epsilon for the layer norm.
    wavelength: The wavelength for the sinusoidal positional embedding.
    parameter_dtype: The dtype of the parameters.
    apply_stop_gradient: When used by a multimodal language model, the gradients
      of the vision transformer should be stopped.
  """

  # image related
  image_height: int = 896
  image_width: int = 896
  image_channels: int = 3
  patch_size: tuple[int, int] = (14, 14)
  num_mm_tokens_per_image_prepool: int = 4096
  num_mm_tokens_per_image: int = 256
  # model related
  mlp_variant: Literal["geglu_approx", "swiglu"] = "geglu_approx"
  pos_embed: Literal["learned", "sinusoidal"] = "learned"
  embedding_dim: int = 1152
  mlp_hidden_dim: int = 4304
  num_encoder_blocks: int = 27
  num_heads: int = 16
  layer_norm_eps: float = 1e-6
  wavelength: float = 10_000.0
  parameter_dtype: jax.typing.DTypeLike = jnp.float32
  apply_stop_gradient: bool = True


@pz.pytree_dataclass
class SigLiPFromPatches(pz.nn.Layer):
  """SigLiP from patches.

  A SigLiP model from patches. The model is a stack of transformer blocks,
  followed by a vision exit layer to downsample the vision embeddings to a
  required output length.

  Attributes:
    siglip_encoder: The SigLiP encoder to extract vision embeddings from
      patches.
    siglip_exit: The SigLiP exit to re-organize the vision embeddings.
  """

  siglip_encoder: pz.nn.Layer
  siglip_exit: pz.nn.Layer

  def __call__(
      self,
      patches: pz.nx.NamedArray,
      **side_inputs,
  ) -> pz.nx.NamedArray:
    soft_tokens = self.siglip_encoder(patches, **side_inputs)
    soft_tokens = self.siglip_exit(soft_tokens, **side_inputs)
    return soft_tokens


def build_vision_transformer_attention(
    name: str,
    init_base_rng: jax.Array | None,
    config: Gemma3SigLipConfig,
) -> pz.nn.Attention:
  """Builds an attention block from a configuration.

  Args:
    name: Name of the attention block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    An Attention block.
  """
  embedding_dim = config.embedding_dim
  num_heads = config.num_heads
  assert embedding_dim % num_heads == 0
  projection_dim = embedding_dim // num_heads
  query_scaling_factor = (embedding_dim / num_heads) ** -0.5

  # This is equivalent to query_head_multiplier = 1
  common_head_axes = {"heads": num_heads}
  qkv_einsum = {"heads": "h"}
  query_only_head_axes = {}
  q_einsum = {}

  return pz.nn.Attention(
      input_to_query=pz.nn.Sequential([
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
          pz.nn.AddBias.from_config(
              name=f"{name}/query",
              init_base_rng=init_base_rng,
              biased_axes={
                  **common_head_axes,
                  **query_only_head_axes,
                  "projection": projection_dim,
              },
              dtype=config.parameter_dtype,
          ),
          pz.nn.ConstantRescale(
              by=jnp.array(query_scaling_factor, dtype=config.parameter_dtype)
          ),
      ]),
      input_to_key=pz.nn.Sequential([
          pz.nn.Linear.from_config(
              name=f"{name}/key",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
          pz.nn.AddBias.from_config(
              name=f"{name}/key",
              init_base_rng=init_base_rng,
              biased_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      input_to_value=pz.nn.Sequential([
          pz.nn.Linear.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
          pz.nn.AddBias.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              biased_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      query_key_to_attn=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"patch": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
                  {"patch": "tkv", **qkv_einsum, "projection": "p"},
              ),
              {"patch": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
          ),
          pz.nn.Softmax("kv_seq"),
      ]),
      attn_value_to_output=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"patch": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
                  {"patch": "tkv", **qkv_einsum, "projection": "p"},
              ),
              {"patch": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
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
          pz.nn.AddBias.from_config(
              name=f"{name}/output",
              init_base_rng=init_base_rng,
              biased_axes={"embedding": embedding_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
  )


def build_vision_transformer_feedforward(
    name: str,
    init_base_rng: jax.Array | None,
    config: Gemma3SigLipConfig,
) -> vision_parts.VisionTransformerFeedForward:
  """Builds a feedforward block from a configuration.

  Args:
    name: Name of the feedforward block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    An instance of VisionTransformerFeedForward containing the GELU MLP blocks.
  """
  if config.mlp_variant == "geglu_approx":
    # Approximate is already the default in JAX, but we specify it explicitly
    # because defaults differ between JAX and PyTorch.
    act_fn = functools.partial(jax.nn.gelu, approximate=True)
  elif config.mlp_variant == "swiglu":
    act_fn = jax.nn.silu
  else:
    raise ValueError(f"Unsupported MLP variant {config.mlp_variant}")

  return vision_parts.VisionTransformerFeedForward([
      pz.nn.Linear.from_config(
          name=f"{name}/dense_0",
          init_base_rng=init_base_rng,
          input_axes={"embedding": config.embedding_dim},
          output_axes={"neurons": config.mlp_hidden_dim},
          dtype=config.parameter_dtype,
      ),
      pz.nn.AddBias.from_config(
          name=f"{name}/dense_0",
          init_base_rng=init_base_rng,
          biased_axes={"neurons": config.mlp_hidden_dim},
          initializer=normal_initializer(
              stddev=1e-6
          ),  # following Gemma3 SigLip to set a normal initializer
          dtype=config.parameter_dtype,
      ),
      pz.nn.Elementwise(act_fn),
      pz.nn.Linear.from_config(
          name=f"{name}/dense_1",
          init_base_rng=init_base_rng,
          input_axes={"neurons": config.embedding_dim},
          output_axes={"embedding": config.mlp_hidden_dim},
          dtype=config.parameter_dtype,
      ),
      pz.nn.AddBias.from_config(
          name=f"{name}/dense_1",
          init_base_rng=init_base_rng,
          biased_axes={"embedding": config.mlp_hidden_dim},
          initializer=normal_initializer(
              stddev=1e-6
          ),  # following Gemma3 SigLip to set a normal initializer
          dtype=config.parameter_dtype,
      ),
  ])


def build_vision_transformer_block(
    name: str,
    init_base_rng: jax.Array | None,
    config: Gemma3SigLipConfig,
) -> vision_parts.VisionTransformerBlock:
  """Builds a visiontransformer block from a configuration.

  Args:
    name: Name of the block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    A full transformer block.
  """
  attn_sequence = [
      pz.nn.LayerNorm.from_config(
          name=f"{name}/pre_attention_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          epsilon=config.layer_norm_eps,
          dtype=config.parameter_dtype,
      ),
      build_vision_transformer_attention(
          f"{name}/attention",
          init_base_rng,
          config,
      ),
  ]
  ffw_sequence = [
      pz.nn.LayerNorm.from_config(
          name=f"{name}/pre_ffw_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          epsilon=config.layer_norm_eps,
          dtype=config.parameter_dtype,
      ),
      build_vision_transformer_feedforward(
          f"{name}/mlp", init_base_rng, config
      ),
  ]
  return vision_parts.VisionTransformerBlock(
      sublayers=[
          pz.nn.Residual(pz.nn.Sequential(attn_sequence)),
          pz.nn.Residual(pz.nn.Sequential(ffw_sequence)),
      ],
  )


def build_vision_transformer(
    config: Gemma3SigLipConfig,
    init_base_rng: jax.Array | None = None,
    name: str = "transformer",
) -> vision_parts.VisionTransformer:
  """Builds a Vision transformer model from a configuration.

  Args:
    config: The configuration of the model.
    init_base_rng: Base RNG for initializing the parameters.
    name: Name for the top-level model, used as a prefix for all parameters.

  Returns:
    A full transformer model.
  """
  sublayers = []
  # Re-arrange patches into images again before fed into the vision transformer
  rearrange_layers = []
  # Step I: re-arrange input from "b n (h w) c" - > "(b n) h w c"
  # Using named arrays, we don't need to flatten the batch and frame axes.
  num_patches_height = config.image_height // config.patch_size[0]
  num_patches_width = config.image_width // config.patch_size[1]
  rearrange_layers.append(
      vision_layers.SplitAxes(
          input_axes="patch",
          output_axes={
              "patch_height": num_patches_height,
              "patch_width": num_patches_width,
          },
      )
  )
  # Step II: re-arrange input from "b h w (p q c)" -> "b (h p) (w q) c"
  rearrange_layers.append(
      vision_layers.SplitAxes(
          input_axes="embedding",
          output_axes={
              "patch_size_height": config.patch_size[0],
              "patch_size_width": config.patch_size[1],
              "channel": config.image_channels,
          },
      ),
  )
  rearrange_layers.append(
      vision_layers.MergeAxes(
          input_axes=("patch_height", "patch_size_height"),
          output_axes="height",
      )
  )
  rearrange_layers.append(
      vision_layers.MergeAxes(
          input_axes=("patch_width", "patch_size_width"),
          output_axes="width",
      )
  )
  sublayers.extend(rearrange_layers)

  # Patch extraction
  patch_extraction_layer = [
      convolution.Conv.from_config(
          name=f"{name}/patch_extraction",
          init_base_rng=init_base_rng,
          input_axes={"channel": config.image_channels},
          output_axes={"embedding": config.embedding_dim},
          convolution_spatial_axes={
              "height": config.patch_size[0],
              "width": config.patch_size[1],
          },
          strides=config.patch_size,
          parallel_axes=None,
          parallel_broadcast_axes=None,
          rename_outputs_if_necessary=True,
          dtype=config.parameter_dtype,
      ),
      pz.nn.AddBias.from_config(
          name=f"{name}/patch_extraction",
          init_base_rng=init_base_rng,
          biased_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
      ),
      vision_layers.MergeAxes(
          input_axes=("height", "width"), output_axes="patch"
      ),
  ]
  sublayers.extend(patch_extraction_layer)

  # Positional embeddings
  if config.pos_embed == "learned":
    pos_emb_layer = vision_layers.ApplyLearnablePE.from_config(
        name=f"{name}",
        init_base_rng=init_base_rng,
        positional_axes={
            "patch": config.num_mm_tokens_per_image_prepool,
            "embedding": config.embedding_dim,
        },
        initializer=normal_initializer(
            stddev=1 / jnp.sqrt(config.embedding_dim)
        ),  # following Gemma3 SigLip to set a normal initializer
        dtype=config.parameter_dtype,
    )
  elif config.pos_embed == "sinusoidal":
    pos_emb_layer = vision_layers.ApplySinCos2dPE(
        wavelength=config.wavelength,
    )
  else:
    raise ValueError(
        f"Unsupported positional embedding type {config.pos_embed}"
    )
  sublayers.append(pos_emb_layer)

  # Add block layers
  for block_index in range(config.num_encoder_blocks):
    sublayers.append(
        build_vision_transformer_block(
            f"{name}/block_{block_index}", init_base_rng, config
        )
    )
  # Add final layer norm
  sublayers.append(
      pz.nn.LayerNorm.from_config(
          name=f"{name}/final_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          epsilon=config.layer_norm_eps,
          dtype=config.parameter_dtype,
      )
  )
  return vision_parts.VisionTransformer(
      metadata=vision_parts.VisionTransformerMetadata(
          common_head_axes={"heads": config.num_heads},
          embedding_dim=config.embedding_dim,
          mlp_hidden_dim=config.mlp_hidden_dim,
          projection_dim=config.embedding_dim // config.num_heads,
          parameter_dtype=config.parameter_dtype,
          num_patches_per_image=config.num_mm_tokens_per_image_prepool,
      ),
      body=pz.nn.Sequential(sublayers),
  )


def build_siglip_vision_transformer(
    config: Gemma3SigLipConfig,
    init_base_rng: jax.Array | None = None,
    name: str = "siglip",
) -> SigLiPFromPatches:
  """Builds a SigLip vision transformer model from a configuration.

  Args:
    config: The configuration of the model.
    init_base_rng: Base RNG for initializing the parameters.
    name: Name for the top-level model, used as a prefix for all parameters.

  Returns:
    A siglip vision transformer model.
  """
  return SigLiPFromPatches(
      siglip_encoder=build_vision_transformer(
          config,
          init_base_rng,
          name=name,
      ),
      siglip_exit=pz.nn.Sequential([
          vision_parts.VisionExit(
              output_length=config.num_mm_tokens_per_image,
          ),
          vision_layers.StopGradient()
          if config.apply_stop_gradient
          else pz.nn.Identity(),
      ]),
  )


def gemma_vision_from_pretrained_checkpoint(
    ckpt_params: dict[str, Any],
) -> SigLiPFromPatches:
  """Builds a Gemma3 vision model from a pretrained checkpoint.

  Args:
    ckpt_params: Nested dictionary of weights from the Gemma checkpoint.

  Returns:
    A Transformer model containing the loaded parameters.
  """
  params = {k.removeprefix("transformer/"): v for k, v in ckpt_params.items()}

  if "SigLiPFromPatches_0/siglip_encoder/embedding" not in params.keys():
    raise ValueError("Vision encoder not found in checkpoint.")
  vision_config = Gemma3SigLipConfig()

  model_def = build_siglip_vision_transformer(
      config=vision_config,
      init_base_rng=None,
      name="vision_encoder",
  )
  parameter_mapping = {}

  # load patch extraction parameters
  parameter_mapping["vision_encoder/patch_extraction.kernel"] = (
      pz.nx.NamedArray.wrap(
          params["SigLiPFromPatches_0/siglip_encoder/embedding"]["kernel"]
      ).tag("height", "width", "channel", "embedding")
  )
  parameter_mapping["vision_encoder/patch_extraction.bias"] = (
      pz.nx.NamedArray.wrap(
          params["SigLiPFromPatches_0/siglip_encoder/embedding"]["bias"]
      ).tag("embedding")
  )
  # load learnable positional embedding parameters
  if vision_config.pos_embed == "learned":
    parameter_mapping["vision_encoder/pos_embedding"] = pz.nx.NamedArray.wrap(
        params["SigLiPFromPatches_0/siglip_encoder"]["pos_embedding"][0]
    ).tag("patch", "embedding")

  # re-name the parameters for vision encoder
  vision_encoder_params = {
      k.removeprefix("SigLiPFromPatches_0/siglip_encoder/Transformer/"): v
      for k, v in ckpt_params.items()
      if k.startswith("SigLiPFromPatches_0/siglip_encoder/Transformer/")
  }
  # load final layer norm parameters
  parameter_mapping["vision_encoder/final_norm/scale.weights"] = (
      pz.nx.NamedArray.wrap(vision_encoder_params["encoder_norm"]["scale"]).tag(
          "embedding"
      )
  )
  parameter_mapping["vision_encoder/final_norm/shift.bias"] = (
      pz.nx.NamedArray.wrap(vision_encoder_params["encoder_norm"]["bias"]).tag(
          "embedding"
      )
  )

  # load vision transformer parameters
  for i in range(vision_config.num_encoder_blocks):
    # load pre-attention layer norm parameters
    parameter_mapping[
        f"vision_encoder/block_{i}/pre_attention_norm/scale.weights"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/LayerNorm_0"]["scale"]
    ).tag(
        "embedding"
    )
    parameter_mapping[
        f"vision_encoder/block_{i}/pre_attention_norm/shift.bias"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/LayerNorm_0"]["bias"]
    ).tag(
        "embedding"
    )
    # load pre-ffw layer norm parameters
    parameter_mapping[
        f"vision_encoder/block_{i}/pre_ffw_norm/scale.weights"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/LayerNorm_1"]["scale"]
    ).tag(
        "embedding"
    )
    parameter_mapping[f"vision_encoder/block_{i}/pre_ffw_norm/shift.bias"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[f"encoderblock_{i}/LayerNorm_1"]["bias"]
        ).tag("embedding")
    )
    # load feedforward network parameters
    parameter_mapping[
        f"vision_encoder/block_{i}/mlp/dense_0.weights"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/MlpBlock_0/Dense_0"]["kernel"]
    ).tag(
        "embedding",
        "neurons",
    )
    parameter_mapping[
        f"vision_encoder/block_{i}/mlp/dense_0.bias"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/MlpBlock_0/Dense_0"]["bias"]
    ).tag(
        "neurons",
    )
    parameter_mapping[
        f"vision_encoder/block_{i}/mlp/dense_1.weights"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/MlpBlock_0/Dense_1"]["kernel"]
    ).tag(
        "neurons",
        "embedding",
    )
    parameter_mapping[
        f"vision_encoder/block_{i}/mlp/dense_1.bias"
    ] = pz.nx.NamedArray.wrap(
        vision_encoder_params[f"encoderblock_{i}/MlpBlock_0/Dense_1"]["bias"]
    ).tag(
        "embedding",
    )
    # load attention parameters
    parameter_mapping[f"vision_encoder/block_{i}/attention/query.weights"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/query"
            ]["kernel"]
        ).tag("embedding", "heads", "projection")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/query.bias"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/query"
            ]["bias"]
        ).tag("heads", "projection")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/key.weights"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/key"
            ]["kernel"]
        ).tag("embedding", "heads", "projection")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/key.bias"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/key"
            ]["bias"]
        ).tag("heads", "projection")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/value.weights"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/value"
            ]["kernel"]
        ).tag("embedding", "heads", "projection")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/value.bias"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/value"
            ]["bias"]
        ).tag("heads", "projection")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/output.weights"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/out"
            ]["kernel"]
        ).tag("heads", "projection", "embedding")
    )
    parameter_mapping[f"vision_encoder/block_{i}/attention/output.bias"] = (
        pz.nx.NamedArray.wrap(
            vision_encoder_params[
                f"encoderblock_{i}/MultiHeadDotProductAttention_0/out"
            ]["bias"]
        ).tag("embedding")
    )

  # Create parameter objects for each parameter, and bind them to the model's
  # slots.
  model = pz.bind_variables(
      model_def,
      [pz.Parameter(value=v, label=k) for k, v in parameter_mapping.items()],
  )
  pz.nn.assert_no_parameter_slots(model)
  return model
