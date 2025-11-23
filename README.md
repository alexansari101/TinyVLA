# TinyVLA: Fast-Iteration Vision-Language-Action Model

A minimal Vision-Language-Action model designed for **rapid experimentation** on laptops/Colab with limited compute. The minimal variant (~17.4M parameters) is ~25x smaller than alternatives like SmolVLA for ultra-fast training.

_Note: This project is still in **active development**_

## 🎯 Design Goals

- **Fast iteration**: Train in 1-2 minutes on a laptop (minimal variant on RTX 3070 GPU) or free Colab
- **Similar architecture to SOTA**: Uses ViT + Transformer decoder (like SmolVLA/OpenVLA)
- **Easy to scale**: Same code structure as larger VLAs - just change config
- **Educational**: Clear, documented code for learning VLA fundamentals

> **📖 For detailed architecture comparison with SmolVLA/OpenVLA and scaling strategies, see [ARCHITECTURE.md](ARCHITECTURE.md)**

## 📊 Model Architecture

```text
TinyVLA Minimal Variant (~17.4M parameters)
├── Vision Encoder: TinyViT (~1.8M params)
│   ├── Patch embedding (8x8 patches)
│   ├── 4 transformer layers
│   └── 3 attention heads
├── Vision-Language Projection (~0.1M params)
├── Language Model: TinyTransformer (~11M params)
│   ├── Token + position embeddings (7.8M for vocab)
│   ├── 4 transformer layers
│   └── 4 attention heads
├── Action Head: MLP (~0.03M params)
│   └── Predicts continuous actions (dx, dy)
└── Text Decoder: Transformer Decoder (~4.3M params)
    ├── Shares token embeddings with encoder (weight tying)
    ├── 4 decoder blocks with self-attn + cross-attn
    └── Generates text descriptions (e.g., "move right")
```

**Key design choices:**

- **ViT-style vision encoder**: Same patch-based approach as SigLip/FasterViT
- **Transformer decoder for language**: Similar to Phi-2 but much smaller
- **Cross-attention fusion**: Transformer decoder with self-attention and cross-attention to fused vision-language encoder features
- **Continuous action prediction**: MSE loss on (dx, dy) displacements
- **Text generation**: Causal transformer decoder for explaining actions

## 🚀 Quick Start

### Installation

**Option 1: Using `uv` (recommended - 10-100x faster)**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies (auto-detects Python 3.11 from .python-version)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project and dependencies
uv pip install -e .
```

**Option 2: Using `pip` (traditional)**

```bash
# Create virtual environment with Python 3.11+
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Setup

```bash
# Test environment and estimate training time
python test_setup.py

# Or with uv
uv run python test_setup.py
```

### Training (1-2 minutes for minimal variant)

```bash
# Train with default config
python train_tiny_vla.py

# Or with uv (runs in the project's virtual environment)
uv run python train_tiny_vla.py

# Monitor with tensorboard
tensorboard --logdir=logs
```

**Expected results for minimal variant (on a laptop with an RTX 3070):**

- Training time: ~1-2 minutes (20 epochs)
- Final val L2 error: <0.004
- GPU memory: ~0.7 GB
- Training speed: ~1400 samples/sec

### Inference & Evaluation

```bash
# Interactive demo
python inference_tiny_vla.py

# Visualize predictions
python inference_tiny_vla.py --visualize

# Evaluate on test set
python inference_tiny_vla.py --evaluate --num-samples 1000
```

## 🎮 Example Dataset: BlockFind (Toy Problem)

**Environment:**

- 64x64 top-down view of colored blocks on a grid
- 3-4 colored blocks per scene

**Language instructions:**

- "find the red block"
- "find the blue block"
- "find the green block"

**Actions:**

- Continuous (dx, dy) in range [-1, 1]
- Normalized direction vector from center to target block

**Dataset:**

- 8000 training samples
- 1000 validation samples
- 1000 test samples
- Generated synthetically in ~1 second

## 🔧 Customization

### Scaling Up

To scale to SmolVLA-like sizes, modify config in `train_tiny_vla.py`:

```python
config = {
    'model': {
        'vision_embed_dim': 768,      # 192 -> 768
        'vision_layers': 12,          # 4 -> 12
        'vision_heads': 12,           # 3 -> 12
        'lang_embed_dim': 2048,       # 256 -> 2048
        'lang_layers': 24,            # 4 -> 24
        'lang_heads': 16,             # 4 -> 16
    }
}
```

This would create a ~450M parameter model (like SmolVLA).

### Using Your Own Dataset

Replace the included `BlockFindDataset` example with your own:

```python
class YourDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'image': torch.tensor(...),      # (C, H, W)
            'instruction': "your text",      # string
            'action': torch.tensor([...])    # (action_dim,)
        }
```

### Different Action Spaces

**Discrete actions** (e.g., navigation):

```python
# In tiny_vla_model.py
self.action_head = nn.Sequential(
    nn.Linear(lang_embed_dim, num_discrete_actions),
    nn.LogSoftmax(dim=-1)
)

# Use CrossEntropyLoss instead of MSELoss
```

**High-dimensional actions** (e.g., 7-DOF robot):

```python
config = {
    'model': {
        'action_dim': 7,  # x, y, z, roll, pitch, yaw, gripper
    }
}
```

## 🔬 Extending to Real Robotics

Once you've validated your approach on the toy problem, scale up:

1. **Better vision encoder**: Replace TinyViT with pretrained SigLip/FasterViT
2. **Larger language model**: Use Phi-2 or Llama-based backbones
3. **Cross-attention fusion**: Replace simple pooling with cross-attention
4. **Real datasets**: Switch to Bridge, RT-1, or your own robot data
5. **More action dimensions**: Extend to 7-DOF + gripper

**Key architectural similarities maintained:**

- Patch-based vision encoding
- Transformer-based language processing
- Learned vision-language fusion
- Direct action prediction

> **📖 For detailed scaling path and code examples, see [ARCHITECTURE.md](ARCHITECTURE.md)**

## 💡 Learning Tips

### Quick experiments to try

1. **Architecture ablations** (~30 sec to 1 min each):
   - Remove vision encoder → see how much vision matters
   - Remove language model → test visual-only policy
   - Try different fusion strategies (concat vs. add vs. cross-attention)

2. **Data efficiency** (~1-2 min each):
   - Train on 1k, 2k, 4k, 8k samples
   - Plot learning curves

3. **Generalization** (~2-3 min):
   - Train on 3 colors, test on 4th color
   - Train on simple instructions, test on complex ones

4. **Compare to baselines**:
   - CNN + LSTM baseline
   - Separate vision and language encoders
   - End-to-end trained vs. pretrained components

## 📚 Related Work & Papers

**Key VLA papers to read:**

- [RT-1: Robotics Transformer](https://arxiv.org/abs/2212.06817) - Original VLA paper
- [RT-2: Vision-Language-Action Models](https://arxiv.org/abs/2307.15818) - Scaling to large LLMs
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)

**Toy problems in robotics:**

- [Behavioural Cloning (BC) on toy tasks](https://arxiv.org/abs/2302.14693)
- [Data-efficient imitation learning](https://arxiv.org/abs/2004.04906)

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `batch_size=32` or `vision_embed_dim=128` in config |
| **Loss is NaN** | Lower learning rate: `lr=1e-4` instead of `3e-4` |
| **Model not learning** | Check dataset: `python -c "from tiny_vla_dataset import BlockFindDataset; d=BlockFindDataset(10); d.visualize_sample(0)"` |
| **Training too slow** | Reduce `train_size=2000` for prototyping, or increase `batch_size=128` |
| **Import errors** | `pip install -r requirements.txt` |
| **Colab issues** | Set `num_workers=2` in config (Colab has limited CPU) |

## 🎓 Next Steps

Once you've mastered TinyVLA:

1. **Scale up**: Gradually increase model size (50M → 100M → 250M → 500M)
2. **Add pretraining**: Use pretrained SigLip + Phi-2
3. **Real robot**: Collect your own data or use Bridge/RT-1 datasets
4. **Advanced techniques**:
   - LoRA fine-tuning for efficiency
   - Diffusion policies for multi-modal actions
   - Mixture of experts for generalization

## 📄 License

MIT License

## 🙏 Acknowledgments

Inspired by:

- SmolVLA and OpenVLA architectures
- RT-1/RT-2 vision-language-action design
- HuggingFace Transformers library

---

## Happy experimenting! 🚀

For questions or issues, feel free to open an issue or reach out.
