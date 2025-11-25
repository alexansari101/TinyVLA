"""
Inference script for TinyVLA
Test the trained model and visualize predictions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tiny_vla_model import create_tiny_vla
from tiny_vla_dataset import BlockFindDataset


class TinyVLAInference:
    """Inference wrapper for TinyVLA"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = create_tiny_vla()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def predict(self, image, instruction):
        """
        Predict action for a single image-instruction pair
        
        Args:
            image: (C, H, W) tensor or (H, W, C) numpy array
            instruction: string
        Returns:
            action: (action_dim,) numpy array
        """
        # Convert numpy to torch if needed
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image)
        
        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)
        
        # Prepare inputs
        image, input_ids, attention_mask = self.model.prepare_inputs(
            image, [instruction]
        )
        
        # Predict
        actions, _ = self.model(image, input_ids, attention_mask)
        
        return actions.cpu().numpy()[0]
    
    @torch.no_grad()
    def predict_batch(self, images, instructions):
        """
        Predict actions for a batch
        
        Args:
            images: (B, C, H, W) tensor
            instructions: list of strings
        Returns:
            actions: (B, action_dim) numpy array
        """
        images = images.to(self.device)
        
        # Prepare inputs
        images, input_ids, attention_mask = self.model.prepare_inputs(
            images, instructions
        )
        
        # Predict
        actions = self.model(images, input_ids, attention_mask)
        
    @torch.no_grad()
    def generate_text(self, image, instruction):
        """
        Generate text description for a single image-instruction pair
        
        Args:
            image: (C, H, W) tensor or (H, W, C) numpy array
            instruction: string
        Returns:
            text: string
        """
        # Convert numpy to torch if needed
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image)
        
        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)
        
        # Prepare inputs
        image, input_ids, attention_mask = self.model.prepare_inputs(
            image, [instruction]
        )
        
        # Generate
        if getattr(self.model, 'use_text_decoder', False):
            generated_ids = self.model.generate_text(image, input_ids, attention_mask)
            text = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
        else:
            return "Text decoder not enabled"
    
    def visualize_predictions(self, dataset, num_samples=8, save_path=None):
        """
        Visualize model predictions on dataset samples
        
        Args:
            dataset: BlockFindDataset
            num_samples: number of samples to visualize
            save_path: path to save figure (optional)
        """
        # Sample random indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, ax in zip(indices, axes):
            sample = dataset[idx]
            
            # Get image and instruction
            image = sample['image']
            instruction = sample['instruction']
            action_gt = sample['action'].numpy()
            
            # Predict action and text
            action_pred = self.predict(image, instruction)
            text_pred = self.generate_text(image, instruction)
            
            # Convert image for display
            image_display = image.permute(1, 2, 0).numpy()
            
            # Calculate error
            error = np.linalg.norm(action_pred - action_gt)
            
            # Plot
            ax.imshow(image_display)
            ax.set_title(
                f"Instr: {instruction}\n"
                f"GT: [{action_gt[0]:.2f}, {action_gt[1]:.2f}]\n"
                f"Pred: [{action_pred[0]:.2f}, {action_pred[1]:.2f}]\n"
                f"Text: {text_pred}\n"
                f"Error: {error:.3f}",
                fontsize=8
            )
            ax.axis('off')
            
            # Draw arrows for actions
            center_x, center_y = 32, 32  # Image center
            arrow_scale = 15
            
            # Ground truth arrow (green)
            ax.arrow(
                center_x, center_y,
                action_gt[0] * arrow_scale, action_gt[1] * arrow_scale,
                head_width=3, head_length=3, fc='green', ec='green',
                linewidth=2, alpha=0.7, label='GT'
            )
            
            # Predicted arrow (red)
            ax.arrow(
                center_x, center_y,
                action_pred[0] * arrow_scale, action_pred[1] * arrow_scale,
                head_width=3, head_length=3, fc='red', ec='red',
                linewidth=2, alpha=0.7, linestyle='--', label='Pred'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
            print("Saved visualization to predictions.png")
        
        plt.close()
    
    def evaluate_accuracy(self, dataset, num_samples=1000):
        """
        Evaluate model accuracy on dataset
        
        Args:
            dataset: BlockFindDataset
            num_samples: number of samples to evaluate
        Returns:
            dict with metrics
        """
        print(f"\nEvaluating on {num_samples} samples...")
        
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        errors = []
        direction_accuracies = []
        
        for idx in indices:
            sample = dataset[idx]
            
            image = sample['image']
            instruction = sample['instruction']
            action_gt = sample['action'].numpy()
            
            # Predict
            action_pred = self.predict(image, instruction)
            
            # L2 error
            error = np.linalg.norm(action_pred - action_gt)
            errors.append(error)
            
            # Direction accuracy (cosine similarity)
            if np.linalg.norm(action_gt) > 0 and np.linalg.norm(action_pred) > 0:
                cos_sim = np.dot(action_pred, action_gt) / (
                    np.linalg.norm(action_pred) * np.linalg.norm(action_gt)
                )
                direction_accuracies.append(cos_sim)
        
        metrics = {
            'mean_l2_error': np.mean(errors),
            'std_l2_error': np.std(errors),
            'median_l2_error': np.median(errors),
            'mean_direction_accuracy': np.mean(direction_accuracies),
            'direction_correct_rate': np.mean([acc > 0.8 for acc in direction_accuracies])
        }
        
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key:30s}: {value:.4f}")
        print("="*50 + "\n")
        
        return metrics


def main():
    """Main inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TinyVLA Inference')
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Visualize predictions'
    )
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evaluate on test set'
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=1000,
        help='Number of samples for evaluation'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    inference = TinyVLAInference(args.checkpoint, device=device)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = BlockFindDataset(num_samples=args.num_samples, seed=44)
    
    # Visualize predictions
    if args.visualize:
        print("\nGenerating visualizations...")
        inference.visualize_predictions(test_dataset, num_samples=8)
    
    # Evaluate
    if args.evaluate:
        metrics = inference.evaluate_accuracy(test_dataset, num_samples=args.num_samples)
    
    # Interactive demo
    if not args.visualize and not args.evaluate:
        print("\nRunning interactive demo...")
        print("Testing on random samples...\n")
        
        for i in range(5):
            idx = np.random.randint(len(test_dataset))
            sample = test_dataset[idx]
            
            image = sample['image']
            instruction = sample['instruction']
            action_gt = sample['action'].numpy()
            
            action_pred = inference.predict(image, instruction)
            text_pred = inference.generate_text(image, instruction)
            error = np.linalg.norm(action_pred - action_gt)
            
            print(f"Sample {i+1}:")
            print(f"  Instruction: {instruction}")
            print(f"  Ground Truth: [{action_gt[0]:.3f}, {action_gt[1]:.3f}]")
            print(f"  Prediction:   [{action_pred[0]:.3f}, {action_pred[1]:.3f}]")
            print(f"  Text Output:  {text_pred}")
            print(f"  L2 Error: {error:.3f}")
            print()


if __name__ == "__main__":
    main()
