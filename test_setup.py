"""
Quick test script to verify TinyVLA setup
Runs a minimal end-to-end test without training
"""

import torch
import sys

print("=" * 60)
print("TinyVLA Setup Verification")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from tiny_vla_dataset import BlockFindDataset
    from tiny_vla_model import create_tiny_vla
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Checking CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = 'cuda'
else:
    print("⚠ CUDA not available, will use CPU (slower)")
    device = 'cpu'

# Test 3: Create dataset
print("\n[3/5] Creating test dataset...")
try:
    dataset = BlockFindDataset(num_samples=10)
    sample = dataset[0]
    print("✓ Dataset created successfully")
    print(f"  Sample image shape: {sample['image'].shape}")
    print(f"  Sample instruction: {sample['instruction']}")
    print(f"  Sample action: {sample['action']}")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    sys.exit(1)

# Test 4: Create model
print("\n[4/5] Creating TinyVLA model...")
try:
    model = create_tiny_vla()
    model = model.to(device)
    param_count = model.count_parameters()
    print("✓ Model created successfully")
    print(f"  Total parameters: {param_count['total']:,}")
    print(f"  Model size: ~{param_count['total'] / 1e6:.1f}M")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test 5: Run forward pass
print("\n[5/5] Testing forward pass...")
try:
    batch_size = 2
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    instructions = [
        "Find the red block",
        "Find the blue block"
    ]
    
    with torch.no_grad():
        images, input_ids, attention_mask = model.prepare_inputs(images, instructions)
        actions = model(images, input_ids, attention_mask)
    
    print("✓ Forward pass successful")
    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {actions.shape}")
    print(f"  Sample prediction: {actions[0].cpu().numpy()}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 6: Estimate training time
print("\n[6/6] Estimating training time...")
try:
    # Time a few iterations
    import time
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.MSELoss()
    
    times = []
    for i in range(5):
        batch = {
            'image': torch.randn(64, 3, 64, 64).to(device),
            'instruction': ["Find the red block"] * 64,
            'action': torch.randn(64, 2).to(device)
        }
        
        start = time.time()
        
        images, input_ids, attention_mask = model.prepare_inputs(
            batch['image'], batch['instruction']
        )
        actions_pred = model(images, input_ids, attention_mask)
        loss = criterion(actions_pred, batch['action'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        if i > 0:  # Skip first iteration (warmup)
            times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    samples_per_sec = 64 / avg_time
    
    # Estimate total training time
    num_samples = 8000
    num_epochs = 20
    total_iterations = (num_samples * num_epochs) / 64
    estimated_time = (total_iterations * avg_time) / 60
    
    print(f"✓ Training speed: {samples_per_sec:.1f} samples/sec")
    print(f"  Time per batch (64 samples): {avg_time*1000:.1f}ms")
    print(f"  Estimated total training time: {estimated_time:.1f} minutes")
    
except Exception as e:
    print(f"⚠ Could not estimate training time: {e}")

print("\n" + "=" * 60)
print("Setup verification complete! ✓")
print("=" * 60)
print("\nYou're ready to train:")
print("  python train_tiny_vla.py")
print("\nOr visualize the dataset:")
print("  python -c \"from tiny_vla_dataset import BlockFindDataset; d = BlockFindDataset(100); d.visualize_sample(0)\"")
print("=" * 60)
