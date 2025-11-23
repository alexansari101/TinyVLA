"""
TinyVLA: Minimal Vision-Language-Action model
Architecture mirrors SmolVLA but scaled down for ultra-fast iteration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math

def get_2d_sinusoidal_pos_embed(embed_dim, grid_size):
    """
    Generates 2D sinusoidal position embeddings (standard for ViT and MAE)
    using torch.meshgrid for a cleaner implementation.
    
    Args:
        embed_dim (int): Total embedding dimension (must be divisible by 4)
        grid_size (int): The grid size (e.g., 8 for 8x8 patches)
    
    Returns:
        (torch.Tensor): (grid_size*grid_size, embed_dim)
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"Embedding dimension {embed_dim} must be divisible by 4 "
            "to be split evenly for x/y sin/cos components."
        )

    # Half for y-coords, half for x-coords
    embed_dim_half = embed_dim // 2
    
    # Half of that for sin, half for cos
    channels = embed_dim_half // 2  # e.g., 192 -> 96 -> 48

    # (channels,)
    # Create the inverse frequency vector
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, dtype=torch.float32) / channels))

    # (grid_size, grid_size)
    # Create 2D coordinate grids
    # 'ij' indexing: pos_y varies with dim 0, pos_x varies with dim 1
    pos_y, pos_x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing='ij'
    )
    
    # (grid_size, grid_size, channels)
    # Apply frequencies to coordinates
    pos_x_emb = torch.einsum('ij,k->ijk', pos_x, inv_freq)
    pos_y_emb = torch.einsum('ij,k->ijk', pos_y, inv_freq)

    # (grid_size, grid_size, embed_dim_half)
    # Apply sin and cos
    pos_x_emb = torch.cat([torch.sin(pos_x_emb), torch.cos(pos_x_emb)], dim=-1)
    pos_y_emb = torch.cat([torch.sin(pos_y_emb), torch.cos(pos_y_emb)], dim=-1)

    # (grid_size, grid_size, embed_dim)
    # Concatenate the y and x components
    pos_emb_2d = torch.cat([pos_y_emb, pos_x_emb], dim=-1)

    # (num_patches, embed_dim)
    # Flatten to (grid_size*grid_size, embed_dim)
    pos_emb_1d = pos_emb_2d.reshape(grid_size * grid_size, embed_dim)
    
    return pos_emb_1d


class TinyViT(nn.Module):
    """
    Tiny Vision Transformer encoder (similar to SigLip/ViT but smaller)
    Default min config: ~1.8M parameters
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
        
        # Keep a learnable position for the [CLS] token
        self.pos_embed_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Create fixed 2D sinusoidal embeddings for patches
        grid_size = image_size // patch_size
        pos_embed_patches = get_2d_sinusoidal_pos_embed(embed_dim, grid_size) # (num_patches, embed_dim)
        
        # Register as a buffer (not a parameter) so it's not trained
        # but moves to the GPU with .to(device)
        self.register_buffer('pos_embed_patches', pos_embed_patches.unsqueeze(0), persistent=False)

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

        # Explicitly assign to a typed local variable to help the linter
        patch_pos_embed: torch.Tensor = self.pos_embed_patches

        pos_embed = torch.cat([
            self.pos_embed_cls.expand(B, -1, -1), 
            patch_pos_embed.expand(B, -1, -1)
        ], dim=1)
        
        x = x + pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=None, attn_mask=None)
        
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
        
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          key_padding_mask=key_padding_mask,
                          attn_mask=attn_mask)[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLanguageModel(nn.Module):
    """
    Tiny transformer encoder for language processing
    Default min config: ~11M parameters (much smaller than Phi-2's 2.7B)
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

        # Create the key_padding_mask (True for pad tokens, False for real tokens)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        
        # Token + position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer blocks (we're using self-attention, not causal for simplicity)
        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)
        
        x = self.norm(x)
        
        return x


class TinyLanguageDecoder(nn.Module):
    """
    Tiny transformer decoder for text generation.
    Conditions on fused VLA features via cross-attention.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Decoder blocks (Self-Attn + Cross-Attn + MLP)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, encoder_features, attention_mask=None):
        """
        Args:
            input_ids: (B, seq_len) target text tokens
            encoder_features: (B, 1, embed_dim) or (B, N, embed_dim) fused features to condition on
            attention_mask: (B, seq_len) mask for input_ids (0 for pad)
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, seq_len = input_ids.shape
        
        # Causal mask (tril)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        # Padding mask for self-attention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0) # (B, seq_len)
            
        # Embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Decoder blocks
        for block in self.blocks:
            x = block(
                x, 
                memory=encoder_features,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=key_padding_mask
            )
            
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention (causal)
        # Query/Key/Value: ALL from decoder tokens (x)
        x2 = self.norm1(x)
        x = x + self.self_attn(
            x2, x2, x2, 
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        
        # Cross-attention (conditioning on memory/encoder features)
        # Query: from decoder (what text we're generating)
        # Key/Value: from encoder (fused vision-language features)
        # Mask: NONE (can attend to ALL encoder features)
        x2 = self.norm2(x)
        x = x + self.cross_attn(
            query=x2, 
            key=memory, 
            value=memory
        )[0]
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class TinyVLA(nn.Module):
    """
    Tiny Vision-Language-Action model
    Default min config: ~14M parameters for ultra-fast iteration
    
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
        use_pretrained_tokenizer: bool = True,
        use_text_decoder: bool = True
    ):
        super().__init__()
        
        self.use_text_decoder = use_text_decoder
        
        # Vision encoder
        self.vision_encoder = TinyViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            num_layers=vision_layers,
            num_heads=vision_heads,
            dropout=dropout
        )
        
        # Vision-to-language projection (single layer to preserve color info)
        self.vision_proj = nn.Linear(vision_embed_dim, lang_embed_dim)

        self.vision_norm = nn.LayerNorm(lang_embed_dim)

        self.fusion_output_norm = nn.LayerNorm(lang_embed_dim)
        
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
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=lang_embed_dim, 
            num_heads=lang_heads, 
            dropout=dropout, 
            batch_first=True
        )

        # Text Decoder (Optional)
        if self.use_text_decoder:
            self.text_decoder = TinyLanguageDecoder(
                vocab_size=vocab_size,
                embed_dim=lang_embed_dim,
                num_layers=lang_layers, # Re-use same depth/width as encoder for simplicity
                num_heads=lang_heads,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
        else:
            self.text_decoder = None

        # # Action head (predicts continuous actions)
        # self.action_head = nn.Sequential(
        #     nn.Linear(lang_embed_dim, lang_embed_dim // 2),
        #     nn.GELU(),
        #     # nn.ReLU(),
        #     # nn.Dropout(dropout),
        #     nn.Linear(lang_embed_dim // 2, action_dim)
        # )
        self.action_head = nn.Linear(lang_embed_dim, action_dim)

        self.action_dim = action_dim
        
    def forward(self, images, input_ids, attention_mask=None):
        """
        Args:
            images: (B, C, H, W)
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            action_pred: (B, action_dim)
            text_logits: (B, seq_len, vocab_size) if target_text_ids is provided, else None
        """
        # 1. Encode vision - Get ALL patch tokens (B, num_patches+1, vision_dim)
        vision_features = self.vision_encoder(images)

        # 2. Encode language - Get the [CLS] token as the instruction summary
        lang_features = self.language_model(input_ids, attention_mask)
        lang_pooled = lang_features[:, 0, :]  # (B, lang_dim)

        # 3. Project vision *patch* features to lang_dim (discarding vision [CLS])
        # Note: We use vision_features[:, 1:, :] to skip the global [CLS] token
        vision_patches = vision_features[:, 1:, :]  # (B, num_patches, vision_dim)
        vision_patches_proj = self.vision_proj(vision_patches)  # (B, num_patches, lang_dim)
        # We can still normalize the patches
        vision_patches_norm = self.vision_norm(vision_patches_proj)

        # 4. Fuse with Cross-Attention
        # Query (Q) = What the instruction is asking for (lang_pooled)
        # Key (K)   = What we see in the image (vision_patches_norm)
        # Value (V) = What we see in the image (vision_patches_norm)

        # We unsqueeze lang_pooled to (B, 1, lang_dim) to act as a single query
        fused_features, _ = self.fusion_attention(
            query=lang_pooled.unsqueeze(1),
            key=vision_patches_norm,
            value=vision_patches_norm
        )

        # Squeeze out the query dimension (B, 1, lang_dim) -> (B, lang_dim)
        fused = fused_features.squeeze(1)

        fused_with_residual = lang_pooled + fused

        fused_norm = self.fusion_output_norm(fused_with_residual)

        # 5. Predict action from this spatially-aware fused representation
        # (We skip the manual vision/lang weights for this simpler, more powerful fusion)
        action_pred = self.action_head(fused_norm)

        return action_pred

    def forward(self, images, input_ids, attention_mask=None, target_text_ids=None):
        """
        Args:
            images: (B, C, H, W)
            input_ids: (B, seq_len) - Instruction text
            attention_mask: (B, seq_len) - Mask for instruction
            target_text_ids: (B, seq_len) - Optional target text for decoder training
        Returns:
            action_pred: (B, action_dim)
            text_logits: (B, seq_len, vocab_size) or None
        """
        # 1. Encode vision - Get ALL patch tokens (B, num_patches+1, vision_dim)
        vision_features = self.vision_encoder(images)

        # 2. Encode language - Get the [CLS] token as the instruction summary
        lang_features = self.language_model(input_ids, attention_mask)
        lang_pooled = lang_features[:, 0, :]  # (B, lang_dim)

        # 3. Project vision *patch* features to lang_dim (discarding vision [CLS])
        # Note: We use vision_features[:, 1:, :] to skip the global [CLS] token
        vision_patches = vision_features[:, 1:, :]  # (B, num_patches, vision_dim)
        vision_patches_proj = self.vision_proj(vision_patches)  # (B, num_patches, lang_dim)
        # We can still normalize the patches
        vision_patches_norm = self.vision_norm(vision_patches_proj)

        # 4. Fuse with Cross-Attention
        # Query (Q) = What the instruction is asking for (lang_pooled)
        # Key (K)   = What we see in the image (vision_patches_norm)
        # Value (V) = What we see in the image (vision_patches_norm)

        # We unsqueeze lang_pooled to (B, 1, lang_dim) to act as a single query
        fused_features, _ = self.fusion_attention(
            query=lang_pooled.unsqueeze(1),
            key=vision_patches_norm,
            value=vision_patches_norm
        )

        # Squeeze out the query dimension (B, 1, lang_dim) -> (B, lang_dim)
        fused = fused_features.squeeze(1)

        fused_with_residual = lang_pooled + fused

        fused_norm = self.fusion_output_norm(fused_with_residual)

        # 5. Predict action from this spatially-aware fused representation
        action_pred = self.action_head(fused_norm)

        # 6. Decode text (if decoder exists and targets provided)
        text_logits = None
        if self.text_decoder is not None and target_text_ids is not None:
            # Condition on the fused feature (unsqueeze to B, 1, dim)
            text_logits = self.text_decoder(
                target_text_ids, 
                fused_norm.unsqueeze(1)
            )

        return action_pred, text_logits

    def generate_text(self, images, input_ids, attention_mask=None, max_new_tokens=20, start_token_id=101):
        """
        Autoregressive text generation
        """
        if self.text_decoder is None:
            return None
            
        B = images.shape[0]
        device = images.device
        
        # Run encoder forward pass to get fused features
        # (Duplicating logic from forward() to avoid refactoring everything into sub-methods for now)
        vision_features = self.vision_encoder(images)
        lang_features = self.language_model(input_ids, attention_mask)
        lang_pooled = lang_features[:, 0, :]
        vision_patches = vision_features[:, 1:, :]
        vision_patches_proj = self.vision_proj(vision_patches)
        vision_patches_norm = self.vision_norm(vision_patches_proj)
        
        fused_features, _ = self.fusion_attention(
            query=lang_pooled.unsqueeze(1),
            key=vision_patches_norm,
            value=vision_patches_norm
        )
        fused = fused_features.squeeze(1)
        fused_with_residual = lang_pooled + fused
        fused_norm = self.fusion_output_norm(fused_with_residual)
        
        # Encoder memory: (B, 1, dim)
        memory = fused_norm.unsqueeze(1)
        
        # Start with [CLS] token (or whatever start_token_id is passed)
        curr_ids = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            # Forward pass through decoder
            logits = self.text_decoder(curr_ids, memory)
            
            # Greedy decode: take argmax of last token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Append
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Stop if all batches have hit [SEP] (102) - simplified check
            # For now, just generate fixed length
            
        return curr_ids
    
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
        "find the red block",
        "find the blue block",
        "find the green block",
        "find the yellow block"
    ]
    
    images, input_ids, attention_mask = model.prepare_inputs(images, instructions)
    
    # Dummy target text for testing decoder
    target_text = ["move right", "move left", "move up", "move down"]
    _, target_ids, _ = model.prepare_inputs(images, target_text) # Reuse prepare_inputs for targets

    with torch.no_grad():
        actions, text_logits = model(images, input_ids, attention_mask, target_text_ids=target_ids)
        
        # Test generation
        generated_ids = model.generate_text(images, input_ids, attention_mask)
        generated_text = model.tokenizer.batch_decode(generated_ids)

    print(f"Input shape: {images.shape}")
    print(f"Action Output shape: {actions.shape}")
    if text_logits is not None:
        print(f"Text Logits shape: {text_logits.shape}")
    print(f"Sample action prediction: {actions[0]}")
    print(f"Generated text sample: {generated_text[0]}")
    print("\nModel test successful!")
