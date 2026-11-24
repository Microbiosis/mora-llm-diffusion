#!/usr/bin/env python3
"""Prepare and push CAD-BEBC checkpoint to Hugging Face Hub."""

import json
import os
from pathlib import Path

HF_README = '''---
language:
- en
license: mit
tags:
- diffusion
- autoregressive
- mora
- cad-bebc
datasets:
- roneneldan/TinyStories
---

# CAD-BEBC (Continuous Autoregressive Diffusion with Block-External Bidirectional Causality)

Small continuous-space block diffusion language model trained on a TinyStories sample.

## Model Details

- **Architecture**: CAD-BEBC unified system (block diffusion + internalized short paths + external expert)
- **Backbone**: Mini-LLaMA style (RoPE + GQA + SwiGLU)
- **Training data**: TinyStories sample (~2MB)
- **Tokenizer**: Byte-level (vocab=101)

## Usage

```bash
# Train
mora src/train.mora  # uses config/train_tinystories.json if configured

# Generate
mora src/generate.mora
```

See the [GitHub repo](https://github.com/Microbiosis/mora-llm-diffusion) for full architecture details.
'''

def main():
    out = Path("hf_model")
    out.mkdir(exist_ok=True)
    (out / "README.md").write_text(HF_README)
    print(f"Wrote {out / 'README.md'}")
    print("To push: huggingface-cli upload Microbiosis/cad-bebc-tinystories hf_model .")

if __name__ == "__main__":
    main()
