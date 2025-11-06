"""
TinyVLA: Minimal Vision-Language-Action model
Architecture mirrors SmolVLA but with ~20-30M parameters for fast iteration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math


class TinyViT(nn.Module):
    """
    Tiny Vision Transformer encoder (similar to SigLip/ViT but smaller)
    ~5-10M parameters
    """
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 192,  # Much smaller than standard ViT
        num_layers: int = 4,   # vs 12 in ViT-Base
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding (like SigLip/ViT)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, num_patches + 1, embed_dim) encoded features
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLanguageModel(nn.Module):
    """
    Tiny transformer decoder for language processing
    ~10-15M parameters (much smaller than Phi-2's 2.7B)
    """
    def __init__(
        self,
        vocab_size: int = 30522,  # BERT vocab size
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Transformer decoder blocks (with causal masking)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            (B, seq_len, embed_dim)
        """
        B, seq_len = input_ids.shape
        
        # Token + position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer blocks (we're using self-attention, not causal for simplicity)
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class TinyVLA(nn.Module):
    """
    Tiny Vision-Language-Action model
    Total: ~20-30M parameters for fast iteration
    
    Architecture (similar to SmolVLA):
    1. Vision encoder (TinyViT) - encodes images
    2. Vision projection - projects vision features to language dimension
    3. Language model - processes text + vision features
    4. Action head - predicts continuous actions
    """
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        vision_embed_dim: int = 192,
        vision_layers: int = 4,
        vision_heads: int = 3,
        lang_embed_dim: int = 256,
        lang_layers: int = 4,
        lang_heads: int = 4,
        action_dim: int = 2,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        use_pretrained_tokenizer: bool = True
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = TinyViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            num_layers=vision_layers,
            num_heads=vision_heads,
            dropout=dropout
        )
        
        # Vision-to-language projection (like in SmolVLA)
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_embed_dim, lang_embed_dim),
            nn.GELU(),
            nn.Linear(lang_embed_dim, lang_embed_dim)
        )
        
        # Tokenizer (using pretrained for convenience)
        if use_pretrained_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            vocab_size = len(self.tokenizer)
        else:
            vocab_size = 30522
            self.tokenizer = None
        
        # Language model
        self.language_model = TinyLanguageModel(
            vocab_size=vocab_size,
            embed_dim=lang_embed_dim,
            num_layers=lang_layers,
            num_heads=lang_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Action head (predicts continuous actions)
        self.action_head = nn.Sequential(
            nn.Linear(lang_embed_dim, lang_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_embed_dim // 2, action_dim)
        )
        
        self.action_dim = action_dim
        
    def forward(self, images, input_ids, attention_mask=None):
        """
        Args:
            images: (B, C, H, W)
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            action_pred: (B, action_dim)
        """
        # Encode vision
        vision_features = self.vision_encoder(images)  # (B, num_patches + 1, vision_dim)
        
        # Project vision features to language dimension
        vision_features = self.vision_proj(vision_features)  # (B, num_patches + 1, lang_dim)
        
        # Encode language
        lang_features = self.language_model(input_ids, attention_mask)  # (B, seq_len, lang_dim)
        
        # Fuse vision and language (simple concatenation + pooling)
        # Alternative: cross-attention like SmolVLA, but this is simpler for toy problem
        vision_pooled = vision_features[:, 0, :]  # Use CLS token (B, lang_dim)
        lang_pooled = lang_features.mean(dim=1)    # Mean pooling (B, lang_dim)
        
        fused = vision_pooled + lang_pooled  # Simple fusion
        
        # Predict action
        action_pred = self.action_head(fused)  # (B, action_dim)
        
        return action_pred
    
    def count_parameters(self):
        """Count total parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def prepare_inputs(self, images, instructions):
        """
        Prepare inputs for the model
        Args:
            images: (B, C, H, W) tensor
            instructions: List of strings
        Returns:
            images, input_ids, attention_mask
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        # Tokenize instructions
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(images.device)
        attention_mask = encoded['attention_mask'].to(images.device)
        
        return images, input_ids, attention_mask


def create_tiny_vla(config=None):
    """Factory function to create TinyVLA with default or custom config"""
    if config is None:
        config = {
            'image_size': 64,
            'patch_size': 8,
            'vision_embed_dim': 192,
            'vision_layers': 4,
            'vision_heads': 3,
            'lang_embed_dim': 256,
            'lang_layers': 4,
            'lang_heads': 4,
            'action_dim': 2,
            'dropout': 0.1
        }
    
    model = TinyVLA(**config)
    
    print("=" * 60)
    print("TinyVLA Model Created")
    print("=" * 60)
    param_count = model.count_parameters()
    print(f"Total parameters: {param_count['total']:,}")
    print(f"Trainable parameters: {param_count['trainable']:,}")
    print(f"Model size: ~{param_count['total'] / 1e6:.1f}M parameters")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating TinyVLA model...")
    model = create_tiny_vla()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64)
    instructions = [
        "Push the red block up",
        "Move blue block left",
        "Push green block down",
        "Move yellow block right"
    ]
    
    images, input_ids, attention_mask = model.prepare_inputs(images, instructions)
    
    with torch.no_grad():
        actions = model(images, input_ids, attention_mask)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {actions.shape}")
    print(f"Sample action prediction: {actions[0]}")
    print("\nModel test successful!")
