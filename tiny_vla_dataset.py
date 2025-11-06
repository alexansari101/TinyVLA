"""
Synthetic Dataset Generator for TinyVLA
Creates a simple block-pushing task with images + language + actions
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
from typing import Dict

class BlockPushDataset(Dataset):
    """
    Toy dataset: Push colored blocks to target locations
    - Image: 64x64 top-down view with colored blocks
    - Language: "Push the [color] block [direction]"
    - Action: (dx, dy) continuous displacement
    """
    
    def __init__(
        self, 
        num_samples: int = 10000,
        image_size: int = 64,
        num_blocks: int = 3,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_blocks = num_blocks
        
        self.colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
        }
        self.color_names = list(self.colors.keys())
        
        self.directions = {
            'left': (-1, 0),
            'right': (1, 0),
            'up': (0, -1),
            'down': (0, 1)
        }
        
        # Pre-generate all samples for consistency
        random.seed(seed)
        np.random.seed(seed)
        self.samples = [self._generate_sample(i) for i in range(num_samples)]
    
    def _generate_sample(self, idx: int) -> Dict:
        """Generate a single training sample"""
        # Randomly place blocks on grid
        grid_size = 8  # 8x8 grid within 64x64 image
        block_positions = {}
        
        # Place blocks at random grid positions
        available_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        random.shuffle(available_positions)
        
        used_colors = random.sample(self.color_names, min(self.num_blocks, len(self.color_names)))
        
        for color in used_colors:
            pos = available_positions.pop()
            block_positions[color] = pos
        
        # Choose random target block and direction
        target_color = random.choice(used_colors)
        target_direction = random.choice(list(self.directions.keys()))
        
        # Create instruction
        instruction = f"Push the {target_color} block {target_direction}"
        
        # Calculate action (normalized displacement)
        dx, dy = self.directions[target_direction]
        action = np.array([dx, dy], dtype=np.float32)
        
        # Create image
        image = self._render_scene(block_positions, grid_size)
        
        return {
            'image': image,
            'instruction': instruction,
            'action': action,
            'block_positions': block_positions,
            'target_color': target_color,
            'target_direction': target_direction
        }
    
    def _render_scene(self, block_positions: Dict, grid_size: int) -> np.ndarray:
        """Render the scene as an image"""
        img = Image.new('RGB', (self.image_size, self.image_size), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        cell_size = self.image_size // grid_size
        
        # Draw grid (optional, subtle)
        for i in range(grid_size + 1):
            pos = i * cell_size
            draw.line([(pos, 0), (pos, self.image_size)], fill=(220, 220, 220), width=1)
            draw.line([(0, pos), (self.image_size, pos)], fill=(220, 220, 220), width=1)
        
        # Draw blocks
        for color_name, (gx, gy) in block_positions.items():
            x = gx * cell_size
            y = gy * cell_size
            color = self.colors[color_name]
            
            # Draw block as filled rectangle with border
            draw.rectangle(
                [x + 2, y + 2, x + cell_size - 2, y + cell_size - 2],
                fill=color,
                outline=(0, 0, 0),
                width=2
            )
        
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert to torch tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(sample['image']).permute(2, 0, 1)
        action = torch.from_numpy(sample['action'])
        
        return {
            'image': image,
            'instruction': sample['instruction'],
            'action': action
        }
    
    def visualize_sample(self, idx: int):
        """Visualize a sample for debugging"""
        import matplotlib.pyplot as plt
        
        sample = self.samples[idx]
        
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(sample['image'])
        ax.set_title(f"Instruction: {sample['instruction']}\nAction: {sample['action']}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved visualization to ./sample_visualization.png")


def create_dataloaders(
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    batch_size: int = 32,
    num_workers: int = 4
):
    """Create train/val/test dataloaders"""
    train_dataset = BlockPushDataset(num_samples=train_size, seed=42)
    val_dataset = BlockPushDataset(num_samples=val_size, seed=43)
    test_dataset = BlockPushDataset(num_samples=test_size, seed=44)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset generation
    print("Creating BlockPush dataset...")
    dataset = BlockPushDataset(num_samples=100)
    
    print(f"Dataset size: {len(dataset)}")
    print("\nSample 0:")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Action: {sample['action']}")
    
    # Visualize a sample
    dataset.visualize_sample(0)
    print("\nDataset creation successful!")
