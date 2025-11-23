# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyVLA is a minimal Vision-Language-Action model designed for rapid experimentation on limited compute. The minimal variant (~14M parameters) is intended to be architecturally similar to state-of-the-art VLA models (SmolVLA, OpenVLA) but ~35x smaller for ultra-fast training iterations.

## Design Goals

- **Fast iteration**: Train in 1-2 minutes (minimal variant on RTX 3070 or free Colab)
- **Similar architecture to SOTA**: Uses ViT + Transformer decoder (like SmolVLA/OpenVLA). Same architecture as production VLAs, just smaller to allow for rapid iteration.
- **Easy to scale**: Same code structure as larger VLAs - just change config
- **Educational**: Clear, documented code for learning VLA fundamentals

## Setup

Uses `uv` for package management (Python 3.11+ required):

```bash
uv venv && source .venv/bin/activate && uv pip install -e .
```

Alternative with pip: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Common Commands

### Training

```bash
# Train with default config (minimal variant: 1-2 min on RTX 3070)
python train_tiny_vla.py

# Monitor training
tensorboard --logdir=logs
```

### Inference & Evaluation

```bash
# Interactive demo
python inference_tiny_vla.py

# Visualize predictions (creates predictions.png)
python inference_tiny_vla.py --visualize

# Evaluate on test set
python inference_tiny_vla.py --evaluate --num-samples 1000

# Custom checkpoint
python inference_tiny_vla.py --checkpoint checkpoints/checkpoint_epoch_10.pt --visualize
```

### Testing Setup

```bash
# Verify environment and estimate training time
python test_setup.py
```

### Dataset Visualization

```bash
# Generate sample visualization from included example dataset (creates sample_visualization.png)
python -c "from tiny_vla_dataset import BlockFindDataset; d=BlockFindDataset(10); d.visualize_sample(0)"
```

## Architecture

### Three-Component Pipeline

1. **Vision Encoder** (`TinyViT` in `tiny_vla_model.py:13-80`): ViT-style patch-based encoder
   - 64x64 images → 8x8 patches → transformer blocks
   - Returns: `(B, num_patches+1, vision_embed_dim)` where first token is CLS

2. **Language Model** (`TinyLanguageModel` in `tiny_vla_model.py:108-161`): Transformer decoder
   - Uses DistilBERT tokenizer (pretrained for convenience)
   - Returns: `(B, seq_len, lang_embed_dim)`

3. **Action Head** (`TinyVLA.action_head` in `tiny_vla_model.py:227-232`): Simple MLP
   - Input: Fused vision-language features
   - Output: Continuous actions (dx, dy) in range [-1, 1]

### Vision-Language Fusion Strategy

Current implementation uses **simple pooling + addition** (`tiny_vla_model.py:256-259`):

```python
vision_pooled = vision_features[:, 0, :]  # CLS token
lang_pooled = lang_features.mean(dim=1)   # Mean pooling
fused = vision_pooled + lang_pooled       # Element-wise addition
```

This is deliberately simple for fast iteration. For production scaling, consider:

- Cross-attention (queries from language, keys/values from vision)
- Learnable fusion tokens
- Gated fusion mechanisms

### Configuration System

All hyperparameters are in `train_tiny_vla.py:232-256`. To scale up:

```python
# Minimal variant (default): ~14M params
config['model'] = {
    'vision_embed_dim': 192,
    'vision_layers': 4,
    'lang_embed_dim': 256,
    'lang_layers': 4,
}

# MediumVLA: ~100M params
config['model'] = {
    'vision_embed_dim': 384,
    'vision_layers': 6,
    'lang_embed_dim': 512,
    'lang_layers': 8,
}

# SmolVLA-like: ~450M params
config['model'] = {
    'vision_embed_dim': 768,
    'vision_layers': 12,
    'lang_embed_dim': 2048,
    'lang_layers': 24,
}
```

## Dataset: BlockFind (Included Example Dataset)

**Purpose**: Toy problem / synthetic task that validates vision-language fusion works correctly.

**Structure** (`tiny_vla_dataset.py:13-153`):

- Generates 64x64 images with 3-4 colored blocks on an 8x8 grid
- Language instructions: "find the [color] block"
- Actions: Normalized (dx, dy) direction vectors from center to target block
- Pre-generated at initialization for consistency across epochs

**Key Validation Metric**: A vision-only or language-only model cannot solve this task perfectly. Success requires proper multimodal fusion.

**Dataset Generation**:

- Train: 8000 samples (seed=42)
- Val: 1000 samples (seed=43)
- Test: 1000 samples (seed=44)
- Generation is near-instant (~1 second for 10k samples)

## Training Loop Details

**Optimizer** (`train_tiny_vla.py:40-45`): AdamW with weight decay

- Learning rate: 3e-4 with linear warmup (100 steps)
- Gradient clipping: max_norm=1.0
- Weight decay: 0.01

**Loss Function**: MSE loss on continuous actions (L2 distance)

**Checkpointing Strategy**:

- Best model saved based on validation loss → `checkpoints/best_model.pt`
- Periodic checkpoints every 5 epochs → `checkpoints/checkpoint_epoch_X.pt`
- Config saved to `checkpoints/config.json`

**Expected Performance** (minimal variant on BlockFind dataset):

- Final validation L2 error: 0.05-0.10
- Direction accuracy (cosine similarity > 0.8): >95%

## File Organization

```Text
tiny_vla/
├── tiny_vla_model.py       # Core architecture (TinyViT, TinyLanguageModel, TinyVLA)
├── tiny_vla_dataset.py     # BlockFind synthetic dataset generator
├── train_tiny_vla.py       # Training loop with TensorBoard logging
├── inference_tiny_vla.py   # Evaluation and visualization
├── test_setup.py           # Environment verification
└── requirements.txt        # PyTorch, transformers, matplotlib, tensorboard
```

## Common Modifications

### Adding New Action Spaces

**Discrete actions** (e.g., {up, down, left, right}):

```python
# In tiny_vla_model.py, replace action_head:
self.action_head = nn.Sequential(
    nn.Linear(lang_embed_dim, num_discrete_actions),
    nn.LogSoftmax(dim=-1)
)

# In train_tiny_vla.py, change loss:
self.criterion = nn.NLLLoss()  # or CrossEntropyLoss
```

**Higher-dimensional continuous actions** (e.g., 7-DOF robot arm):

```python
config['model']['action_dim'] = 7  # x, y, z, roll, pitch, yaw, gripper
```

### Using Pretrained Components

To use pretrained vision/language models while keeping fast training:

```python
from transformers import AutoModel

# Replace TinyViT with pretrained SigLip
self.vision_encoder = AutoModel.from_pretrained('google/siglip-base-patch16-224')

# Replace TinyLanguageModel with Phi-2
self.language_model = AutoModel.from_pretrained('microsoft/phi-2')

# Freeze pretrained weights (train only projection + action head)
for param in self.vision_encoder.parameters():
    param.requires_grad = False
for param in self.language_model.parameters():
    param.requires_grad = False
```

This can give improved performance while maintaining fast training time!

### LoRA Fine-tuning for Large Models

When scaling to 1B+ parameters:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
# Now only ~1-5M parameters are trainable
```

## Debugging Tips

**Model not learning (loss not decreasing)**:

1. Verify dataset: `python tiny_vla_dataset.py` → check `sample_visualization.png`
2. Check for NaN gradients: Add `torch.autograd.set_detect_anomaly(True)`
3. Reduce learning rate or increase warmup steps
4. Verify data preprocessing (images should be in [0, 1], actions in [-1, 1])

**Out of GPU memory**:

1. Reduce batch size: `config['training']['batch_size'] = 32` (from 64)
2. Reduce model size: `config['model']['vision_embed_dim'] = 128` (from 192)
3. Set `num_workers=0` if DataLoader is causing issues

**Training too slow**:

1. Increase batch size if memory allows (faster with larger batches)
2. Reduce dataset size for prototyping: `config['training']['train_size'] = 2000`
3. Reduce num_workers if CPU is bottleneck
4. Verify CUDA is being used: `python -c "import torch; print(torch.cuda.is_available())"`

## Architecture Comparison to SOTA

| Component | TinyVLA | SmolVLA | OpenVLA |
|-----------|---------|---------|---------|
| Parameters | 14M | 450M | 7B |
| Vision | TinyViT (4 layers) | SigLip | SigLip-Large |
| Language | Custom (4 layers) | Phi-2 (32 layers) | Llama-2-7B |
| Fusion | Cross-Attention  | Cross-Attention | Cross-Attention |

**Key Insight**: The architectural *pattern* is similar across all sizes. You can prototype on TinyVLA in 1-2 mins and then scale up if desired.

## Extending to Real Robotics

To scale from toy problem to real robot deployment:

1. **Vision**: Replace TinyViT with pretrained SigLip/FasterViT (freeze or LoRA)
2. **Language**: Replace custom model with Phi-2/Llama (freeze or LoRA)
3. **Dataset**: Switch from BlockFind example dataset to Bridge, RT-1, or custom robot data
4. **Actions**: Extend to 7-DOF + gripper (change `action_dim`)
5. **Fusion**: Upgrade to cross-attention for better multimodal reasoning
6. **Training**: Use larger batch sizes, longer training (100+ epochs on real data)

The code structure remains the same—only config values and pretrained components change.
