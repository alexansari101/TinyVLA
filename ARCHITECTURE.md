# TinyVLA Architecture Deep Dive

## TinyVLA vs. State-of-the-Art (Small) Vision-Language-Action Models

This document shows how TinyVLA's architecture relates to alternative State-of-the-Art (small) VLAs.

## Architecture Comparison Table

| Component | TinyVLA (Minimal Variant) | SmolVLA | OpenVLA | RT-2 |
|-----------|---------------------------|---------|---------|------|
| **Total Parameters** | 13M | 450M | 7B | 55B |
| **Vision Encoder** | TinyViT (4 layers) | SigLip (12 layers) | SigLip-Large | PaLI-X ViT |
| **Vision Dim** | 192 | 384 | 1024 | 2048 |
| **Vision Patches** | 8x8 | 16x16 | 16x16 | 16x16 |
| **Language Model** | Tiny Transformer | Phi-2 | Llama-2-7B | PaLM-E |
| **Language Dim** | 256 | 2560 | 4096 | 8192 |
| **Language Layers** | 4 | 32 | 32 | 48 |
| **Fusion Method** | Pooling + Add | Cross-Attention | Cross-Attention | Cross-Attention |
| **Action Head** | 2-layer MLP | 3-layer MLP | 3-layer MLP | 4-layer MLP |
| **GPU Memory (Est.)** | 2-3 GB | 12-16 GB | 40+ GB | 80+ GB |
| **Typical Use Case** | Learning/Prototyping | Research/Fine-tuning | Production | Large-scale deployment |

## Architectural Patterns

Modern VLAs tend to follow this pattern:

```
Image → Vision Encoder → Vision Features
Text → Language Encoder → Language Features
Vision Features + Language Features → Fusion → Action Prediction
```

### 1. Vision Encoding
- **Pattern**: Patch-based (ViT-style) encoding
- **TinyVLA**: 64x64 image → 8x8 patches → 8x8 = 64 tokens
- **SmolVLA**: 224x224 image → 16x16 patches → 14x14 = 196 tokens
- **Shared**: Positional embeddings, CLS token, LayerNorm

### 2. Language Processing
- **Pattern**: Transformer decoder or encoder
- **TinyVLA**: 4-layer transformer with 256-dim embeddings
- **SmolVLA**: Phi-2 (32 layers, 2560-dim)
- **Shared**: Self-attention, feedforward layers, token embeddings

### 3. Vision-Language Fusion
 - **Cross-Attention**: ✓ Q from lang, K/V from vision

### 4. Action Prediction
- **Pattern**: MLP head on fused features

## Why is TinyVLA Fast?

1. **Small embedding dimensions**:
   - 192-256 vs 2048-4096
   - Matrix multiplications scale as O(d²)
   - 10x smaller dim → 100x fewer FLOPs per layer

2. **Fewer layers**:
   - 4 layers vs 32 layers
   - Linear speedup: 8x fewer layers → 8x faster

3. **Smaller attention**:
   - 3 heads vs 32 heads
   - Fewer attention computations

4. **Simple fusion**:
   - Pooling vs cross-attention
   - Cross-attention adds significant compute

### What Do You Lose?

| Capability | TinyVLA | SmolVLA/OpenVLA |
|------------|---------|-----------------|
| Simple tasks | ✓ | ✓ |
| Complex reasoning | Limited | ✓ |
| Long context | ~32 tokens | ~2048 tokens |
| Fine details | Lower | Higher |
| Generalization | Good for training distribution | Better out-of-distribution |
| Multi-step planning | Weak | Strong |

### What Do You Keep?

✓ **Same architecture patterns**: ViT + Transformer + Fusion

✓ **Same data format**: (image, text, action) tuples

✓ **Same evaluation metrics**: Success rate, action error

✓ **Same scaling laws**: More data + bigger model = better performance

## Bridging the Gap: Techniques

### 1. Use Pretrained Components
Instead of training from scratch:
```python
# Use pretrained vision
from transformers import AutoModel
vision = AutoModel.from_pretrained('google/siglip-base-patch16-224')

# Use pretrained language
language = AutoModel.from_pretrained('microsoft/phi-2')

# Only train projection + action head
for param in vision.parameters():
    param.requires_grad = False
for param in language.parameters():
    param.requires_grad = False
```

This gives you SmolVLA-like performance while maintaining TinyVLA training speed!

### 2. LoRA Fine-tuning
Train only 1-5M parameters even on 7B models:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(large_model, lora_config)
# Trains as fast as TinyVLA, performs like larger model!
```

### 3. Knowledge Distillation
Train TinyVLA to mimic a larger model:
```python
# Generate training data using large model
large_model_actions = large_model(images, texts)

# Train small model to match
loss = MSE(tiny_model(images, texts), large_model_actions)
```
---

Now go train your TinyVLA and see these patterns in action! 🚀
