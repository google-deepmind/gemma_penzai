# Gemma in Penzai

---
`gemma_penzai` is JAX research toolkit for visualizing, manipulating and
understanding Gemma models with multi-modal support based on Penzai. The
original [Penzai](https://github.com/google-deepmind/penzai) mainly supports
text-only LLMs, including Gemma 1 and Gemma 2. Now we extend Penzai with vision
and multimodal support. Therefore, Gemma 3 is supported in this package for more
interpretability research. As a preliminary, the detailed documentation on
Penzai can be found at https://penzai.readthedocs.io.

---

[Gemma](https://ai.google.dev/gemma) is a family of open-weights Large Language
Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini
research and technology.

It has been implemented on different platforms:

*   [Gemma on JAX](https://github.com/google-deepmind/gemma)
*   [Gemma on PyTorch](https://github.com/google/gemma_pytorch)
*   [Gemma on Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/gemma)

However, using the above implementations makes it difficult to visualize the
internal mechanism of Gemma. Therefore, we extend the implementation of Gemma
on [Penzai](https://github.com/google-deepmind/penzai), a JAX research toolkit
for building, editing, and visualizing neural networks.

Gemma 1 and Gemma 2 have been supported in original Penzai package, here we
mainly extend the support of Gemma 3 with following new features:

- Vision Transformers (ViTs) and basic components.
- Multi-modal Large Language Models (MLLMs) with both vision encoders and LLM
backbone. The new attention mask is implemented.
- Decoding algorithms for MLLMs.

## Getting Started

If you haven't already installed JAX with TPU support, you should do that first,
since the installation process depends on your platform. You can find
instructions in the
[JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).
Afterwards, you can install our package as

```python
git clone https://github.com/google-deepmind/gemma_penzai.git
cd gemma_penzai
pip install --upgrade pip
pip install -e .
```

and import it and its dependence `penzai` using

```python
import penzai
from penzai import pz
from gemma_penzai import mllm, vision
```

(`penzai.pz` is an *alias namespace*, which makes it easier to reference
common Penzai objects.)

When working in a Colab or IPython notebook, we recommend also configuring
Treescope (Penzai's companion pretty-printer) as the default pretty printer, and
enabling some utilities for interactive use:

```python
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
```

## Using the package

We provide notebooks inside `./notebooks` about the basic usage of Gemma 3,
including the multimodal case. Inside the folder `./notebooks/mech_interp`, we also provide tutorials for [attention sink](https://arxiv.org/abs/2309.17453)), [logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [gemma scope 1](https://deepmind.google/models/gemma/gemma-scope/) and [2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/).

## Acknowledgements

Our codes are developed based on [Penzai](https://github.com/google-deepmind/penzai)
and [Gemma on JAX](https://github.com/google-deepmind/gemma).
