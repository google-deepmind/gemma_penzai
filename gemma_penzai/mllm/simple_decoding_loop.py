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

"""A simple decoding loop for the transformer model.

This can be used to sample from a transformer in decoding mode. Currently, it
incorporates the logic for temperature sampling, top-p sampling, top-k sampling,
and greedy sampling.
"""

from __future__ import annotations

from gemma_penzai.mllm import sampling_mode
import jax
import jax.numpy as jnp
from penzai import pz


KVCachingTransformerLM = sampling_mode.KVCachingTransformerLM
KVCachingTransformerMultiModalLM = (
    sampling_mode.KVCachingTransformerMultiModalLM
)


def temperature_sample_pyloop(
    model: KVCachingTransformerLM | KVCachingTransformerMultiModalLM,
    prompt: pz.nx.NamedArray,
    rng: jax.Array,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    images: pz.nx.NamedArray | None = None,
    max_sampling_steps: int | None = None,
) -> pz.nx.NamedArray:
  """Runs temperature sampling in a Python for loop.

  Args:
    model: The converted model we are running inference with.
    prompt: A named array of prompt tokens. Must have a "seq" axis along which
      the tokens for each batch element are arranged. Should always have at
      least one non-padding token along the "seq" axis (usually the beginning-
      of-sequence token).
    rng: JAX PRNGKey to use for sampling.
    temperature: Temperature to sample at.
    top_p: Top-p to sample at.
    top_k: Top-k to sample at.
    images: A named array of image patches.
    max_sampling_steps: Maximum number of sampling steps to run. If None,
      samples until filling up the key-value cache.

  Returns:
    A named array of continuations of the prompt.
  """
  if max_sampling_steps is None:
    max_sampling_steps = model.cache_len

  if isinstance(model, KVCachingTransformerLM):
    all_log_probs = model(prompt)
  elif isinstance(model, KVCachingTransformerMultiModalLM):
    all_log_probs = model(prompt, images=images)
  else:
    raise ValueError(f"Unsupported model type: {type(model)}")

  # Find the last non-padding token; this determines where we can find the
  # next log probs.
  non_padding_mask = prompt != model.pad_id
  max_non_padding_index = pz.nx.nmap(
      lambda x: jnp.argmax(jnp.where(x, jnp.arange(x.shape[0]), -1))
  )(non_padding_mask.untag("seq"))
  next_log_probs = all_log_probs[{"seq": max_non_padding_index}]

  sampled_count = 0
  initial_cache_index = model.cache_end_index.value
  while True:
    rng, key = jax.random.split(rng)
    # Split a key across named axes:
    batched_keys = pz.nx.random_split(key, model.batch_axes)
    # Sample the next token with different sampling methods.
    if temperature == 0.0:
      # greedy sampling
      del batched_keys  # Unused.
      next_token = pz.nx.nmap(jnp.argmax)(next_log_probs.untag("vocabulary"))
    else:
      # temperature scaling
      next_log_probs = next_log_probs.untag("vocabulary") / temperature
      if top_p < 1.0:
        # top-p sampling, borrowed from
        # https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampling.py
        assert top_k == -1, (
            "Temperature sampling with both top-p and top-k enabled is not"
            " supported yet."
        )
        sorted_log_probs = pz.nx.nmap(jnp.sort)(next_log_probs, axis=-1)
        sorted_probs = pz.nx.nmap(jax.nn.softmax)(sorted_log_probs, axis=-1)
        cumulative_probs = pz.nx.nmap(jnp.cumsum)(sorted_probs, axis=-1)
        # mask to remove tokens with cumulative probability > top-p.
        sorted_mask = cumulative_probs < top_p
        # get the index of the first token with cumulative probability < top-p.
        cutoff_index = pz.nx.nmap(jnp.sum)(
            sorted_mask < top_p, axis=-1, keepdims=True
        )
        # get the logit value where the cutoff is.
        cutoff_log_probs = pz.nx.nmap(jnp.take_along_axis)(
            sorted_log_probs, cutoff_index, axis=-1
        )
        # select logit values that are smaller than the cutoff logit.
        next_log_probs = pz.nx.nmap(jnp.where)(
            next_log_probs < cutoff_log_probs,
            jnp.finfo(next_log_probs.dtype).min,
            next_log_probs,
        )
        # sample the next token with new log probs.
        next_token = pz.nx.nmap(jax.random.categorical)(
            batched_keys, next_log_probs
        )
      elif top_k > 0:
        # top-k sampling, borrowed from
        # https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_sampling.py
        assert top_p == 1.0, (
            "Temperature sampling with both top-p and top-k enabled is not"
            " supported yet."
        )
        topk_values, topk_indices = pz.nx.nmap(jax.lax.top_k)(
            next_log_probs, top_k
        )
        next_token_indices = pz.nx.nmap(jax.random.categorical)(
            batched_keys, topk_values
        )
        next_token = topk_indices[next_token_indices]
      else:
        # random sampling
        next_token = pz.nx.nmap(jax.random.categorical)(
            batched_keys, next_log_probs
        )
    sampled_count += 1
    # Are we done?
    if (
        model.cache_end_index.value >= model.cache_len
        or sampled_count >= max_sampling_steps
    ):
      break
    next_log_probs = model(next_token[{"seq": None}]).untag("seq").squeeze(0)
  # Add the last token we sampled (which doesn't need to be run through the
  # model).
  start = initial_cache_index
  end = model.cache_end_index.value
  final_written = pz.nx.concatenate(
      [
          model.previous_tokens.value[{"seq": pz.slice[start:end]}],
          next_token[{"seq": None}],
      ],
      "seq",
  )
  return final_written  # pytype: disable=bad-return-type
