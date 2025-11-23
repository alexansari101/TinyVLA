"""
Training script for TinyVLA
Fast iteration training loop with logging and validation
"""

import os
# Disable tokenizer parallelism to avoid fork warnings with DataLoader multiprocessing.
# This has negligible impact on speed for short instructions (~5 tokens) with num_workers>0.
# Only matters for: long documents (1000+ tokens), num_workers=0, or heavy preprocessing.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from tqdm import tqdm
import json

from tiny_vla_model import create_tiny_vla
from tiny_vla_dataset import create_dataloaders


class TinyVLATrainer:
    """Trainer class for TinyVLA"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=3e-4,
        weight_decay=0.01,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        warmup_steps=100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer (AdamW like in SmolVLA)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Loss function (MSE for continuous actions)
        # Loss function (MSE for continuous actions)
        self.criterion_action = nn.MSELoss()
        # Loss function for text (CrossEntropy)
        self.criterion_text = nn.CrossEntropyLoss(ignore_index=0) # 0 is pad token
        
        # # Learning rate scheduler with warmup
        # self.warmup_steps = warmup_steps
        # self.base_lr = lr

        self.warmup_steps = warmup_steps
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Add 1 to step due to 0-indexing
                return float(step + 1) / float(self.warmup_steps)
            return 1.0 # Full base_lr after warmup

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        self.global_step = 0
        self.epoch = 0
        
    # def get_lr(self):
    #     """Learning rate with linear warmup"""
    #     if self.global_step < self.warmup_steps:
    #         return self.base_lr * (self.global_step / self.warmup_steps)
    #     return self.base_lr
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch in pbar:
            # Get batch data
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions_gt = batch['action'].to(self.device)
            descriptions = batch['description']
            
            # Prepare inputs
            images, input_ids, attention_mask = self.model.prepare_inputs(images, instructions)
            
            # Prepare targets for text decoder
            _, target_ids, _ = self.model.prepare_inputs(images, descriptions)
            
            # Shift tokens for teacher forcing
            decoder_input_ids = target_ids[:, :-1]
            decoder_targets = target_ids[:, 1:]
            
            # Forward pass
            actions_pred, text_logits = self.model(
                images, 
                input_ids, 
                attention_mask, 
                target_text_ids=decoder_input_ids
            )
            
            # Compute loss
            loss_action = self.criterion_action(actions_pred, actions_gt)
            
            # Logits: E.g. (B, 3, vocab_size) for [CLS, move, right]
            # Targets: E.g. (B, 3) for [move, right, SEP]
            # We predict next token at each position:
            #   - At pos 0 (CLS): predict "move"
            #   - At pos 1 (move): predict "right"
            #   - At pos 2 (right): predict "SEP"
            loss_text = self.criterion_text(
                text_logits.reshape(-1, text_logits.size(-1)), # (B*3, vocab_size)
                decoder_targets.reshape(-1) # (B*3,)
            )
            
            loss = loss_action + loss_text
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (helps stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # # Update learning rate with warmup
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = self.get_lr()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'act_loss': f'{loss_action.item():.4f}',
                'txt_loss': f'{loss_text.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_error = 0  # L2 distance between predicted and ground truth
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Get batch data
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions_gt = batch['action'].to(self.device)
            
            # Prepare inputs
            images, input_ids, attention_mask = self.model.prepare_inputs(images, instructions)
            
            # Forward pass
            # Forward pass
            # For validation, we can just check action loss, or both.
            # Let's check both.
            descriptions = batch['description']
            _, target_ids, _ = self.model.prepare_inputs(images, descriptions)
            
            decoder_input_ids = target_ids[:, :-1]
            decoder_targets = target_ids[:, 1:]
            
            actions_pred, text_logits = self.model(
                images, 
                input_ids, 
                attention_mask, 
                target_text_ids=decoder_input_ids
            )
            
            # Compute loss
            loss_action = self.criterion_action(actions_pred, actions_gt)
            loss_text = self.criterion_text(
                text_logits.reshape(-1, text_logits.size(-1)), 
                decoder_targets.reshape(-1)
            )
            
            loss = loss_action + loss_text

            total_loss += loss.item()
            
            # Compute L2 error
            error = torch.norm(actions_pred - actions_gt, dim=1).mean()
            total_error += error.item()
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_error = total_error / num_batches
        
        return avg_loss, avg_error
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_error = self.validate()
            
            # Log
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/error', val_error, epoch)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val L2 Error: {val_error:.4f}")
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from epoch {self.epoch}")


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'model': {
            'image_size': 64,
            'patch_size': 8,
            'vision_embed_dim': 192,
            'vision_layers': 4,
            'vision_heads': 3,
            'lang_embed_dim': 256,
            'lang_layers': 4,
            'lang_heads': 4,
            'action_dim': 2,
            'max_seq_len': 32,
            'dropout': 0.1,
            'use_text_decoder': True
        },
        'training': {
            'train_size': 8000,
            'val_size': 1000,
            'test_size': 1000,
            'batch_size': 64,  # Larger batch size for faster training
            'num_epochs': 20,
            'lr': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'num_workers': 4
        }
    }
    
    # Save config
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    print("\nCreating model...")
    model = create_tiny_vla(config['model'])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=config['training']['train_size'],
        val_size=config['training']['val_size'],
        test_size=config['training']['test_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Create trainer
    trainer = TinyVLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps']
    )
    
    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.val_loader = test_loader
    test_loss, test_error = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test L2 Error: {test_error:.4f}")


if __name__ == "__main__":
    main()
