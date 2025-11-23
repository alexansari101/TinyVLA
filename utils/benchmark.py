"""
Comprehensive benchmark for TinyVLA training
Measures parameter count, training time, and GPU memory usage
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import time
from tiny_vla_model import create_tiny_vla
from tiny_vla_dataset import create_dataloaders

def format_memory(bytes_):
    """Format bytes to human readable string"""
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024**2:
        return f"{bytes_/1024:.2f} KB"
    elif bytes_ < 1024**3:
        return f"{bytes_/(1024**2):.2f} MB"
    else:
        return f"{bytes_/(1024**3):.2f} GB"

def benchmark_training():
    """Run comprehensive benchmark"""

    print("=" * 70)
    print("TinyVLA Training Benchmark")
    print("=" * 70)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {gpu_name}")
        print(f"Total GPU Memory: {format_memory(gpu_memory)}")

    # Configuration (same as train_tiny_vla.py)
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
            'batch_size': 64,
            'num_epochs': 20,
            'lr': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'num_workers': 4
        }
    }

    # Create model
    print("\n" + "-" * 70)
    print("Model Configuration")
    print("-" * 70)
    model = create_tiny_vla(config['model'])
    model = model.to(device)

    # Create dataloaders (small subset for benchmark)
    print("\n" + "-" * 70)
    print("Creating dataloaders (small subset for benchmark)...")
    print("-" * 70)
    train_loader, val_loader, _ = create_dataloaders(
        train_size=500,  # Small subset for quick benchmark
        val_size=100,
        test_size=100,
        batch_size=config['training']['batch_size'],
        num_workers=0  # Avoid multiprocessing overhead for benchmark
    )

    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    criterion_action = torch.nn.MSELoss()
    criterion_text = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Benchmark training iterations
    print("\n" + "-" * 70)
    print("Benchmarking Training Speed")
    print("-" * 70)

    model.train()

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    batch_times = []

    # Warmup iterations
    print("Running warmup iterations...")
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break

        images = batch['image'].to(device)
        instructions = batch['instruction']
        actions_gt = batch['action'].to(device)
        descriptions = batch['description']

        images, input_ids, attention_mask = model.prepare_inputs(images, instructions)
        _, target_ids, _ = model.prepare_inputs(images, descriptions)

        decoder_input_ids = target_ids[:, :-1]
        decoder_targets = target_ids[:, 1:]

        actions_pred, text_logits = model(
            images, input_ids, attention_mask, target_text_ids=decoder_input_ids
        )

        loss_action = criterion_action(actions_pred, actions_gt)
        loss_text = criterion_text(
            text_logits.reshape(-1, text_logits.size(-1)),
            decoder_targets.reshape(-1)
        )
        loss = loss_action + loss_text

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark iterations
    print("Running benchmark iterations...")
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break

        start_time = time.time()

        images = batch['image'].to(device)
        instructions = batch['instruction']
        actions_gt = batch['action'].to(device)
        descriptions = batch['description']

        images, input_ids, attention_mask = model.prepare_inputs(images, instructions)
        _, target_ids, _ = model.prepare_inputs(images, descriptions)

        decoder_input_ids = target_ids[:, :-1]
        decoder_targets = target_ids[:, 1:]

        actions_pred, text_logits = model(
            images, input_ids, attention_mask, target_text_ids=decoder_input_ids
        )

        loss_action = criterion_action(actions_pred, actions_gt)
        loss_text = criterion_text(
            text_logits.reshape(-1, text_logits.size(-1)),
            decoder_targets.reshape(-1)
        )
        loss = loss_action + loss_text

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

    avg_batch_time = sum(batch_times) / len(batch_times)
    samples_per_sec = config['training']['batch_size'] / avg_batch_time

    # Calculate training time estimates
    total_samples = config['training']['train_size']
    batches_per_epoch = total_samples / config['training']['batch_size']
    time_per_epoch = batches_per_epoch * avg_batch_time
    total_training_time = time_per_epoch * config['training']['num_epochs']

    print(f"\nBatch size: {config['training']['batch_size']}")
    print(f"Average time per batch: {avg_batch_time*1000:.1f} ms")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")
    print(f"\nTime per epoch (125 batches): {time_per_epoch:.1f} seconds ({time_per_epoch/60:.2f} minutes)")
    print(f"Total training time (20 epochs): {total_training_time:.1f} seconds ({total_training_time/60:.2f} minutes)")

    # GPU Memory usage
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated()
        print("\n" + "-" * 70)
        print("GPU Memory Usage")
        print("-" * 70)
        print(f"Peak memory allocated: {format_memory(peak_memory)}")
        print(f"Peak memory reserved: {format_memory(torch.cuda.max_memory_reserved())}")
        print(f"Current memory allocated: {format_memory(torch.cuda.memory_allocated())}")

    # Validation benchmark
    print("\n" + "-" * 70)
    print("Benchmarking Validation Speed")
    print("-" * 70)

    model.eval()
    val_times = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 2:
                break

            start_time = time.time()

            images = batch['image'].to(device)
            instructions = batch['instruction']
            actions_gt = batch['action'].to(device)
            descriptions = batch['description']

            images, input_ids, attention_mask = model.prepare_inputs(images, instructions)
            _, target_ids, _ = model.prepare_inputs(images, descriptions)

            decoder_input_ids = target_ids[:, :-1]
            decoder_targets = target_ids[:, 1:]

            actions_pred, text_logits = model(
                images, input_ids, attention_mask, target_text_ids=decoder_input_ids
            )

            if device == 'cuda':
                torch.cuda.synchronize()

            val_time = time.time() - start_time
            val_times.append(val_time)

    avg_val_time = sum(val_times) / len(val_times)
    val_batches = config['training']['val_size'] / config['training']['batch_size']
    time_per_validation = val_batches * avg_val_time

    print(f"Average time per validation batch: {avg_val_time*1000:.1f} ms")
    print(f"Time per validation pass: {time_per_validation:.1f} seconds")

    # Total training time including validation
    total_with_validation = total_training_time + (time_per_validation * config['training']['num_epochs'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    param_count = model.count_parameters()
    print(f"\nModel Parameters: {param_count['total']:,} ({param_count['total']/1e6:.1f}M)")

    if device == 'cuda':
        print(f"\nGPU Memory (training): {format_memory(peak_memory)}")

    print(f"\nTraining Time:")
    print(f"  - Per epoch: {time_per_epoch/60:.2f} minutes")
    print(f"  - Total (20 epochs, training only): {total_training_time/60:.2f} minutes")
    print(f"  - Total (including validation): {total_with_validation/60:.2f} minutes")
    print(f"  - Throughput: {samples_per_sec:.1f} samples/sec")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    benchmark_training()
