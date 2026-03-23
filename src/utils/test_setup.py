#!/usr/bin/env python3
"""
Quick Setup Verification Script

Tests that everything is configured correctly before launching full training.
Runs minimal checks on:
- Model import
- Zarr data access
- GPU availability
- Dataset creation
- Single training iteration

Usage: python test_setup.py --zarr-store /path/to/data.zarr
"""

import argparse
import sys
import os

def test_imports():
    """Test all required imports."""
    print("=" * 70)
    print("Testing imports...")
    print("=" * 70)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import zarr
        print(f"✓ Zarr {zarr.__version__}")
    except ImportError as e:
        print(f"✗ Zarr import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from src.models.fuxi_model import FuXi
        print(f"✓ FuXi model imported successfully")
        return True
    except ImportError as e:
        print(f"✗ FuXi model import failed: {e}")
        print("\nPlease ensure:")
        print("  1. model.py exists in the current directory")
        print("  2. It contains a class named 'FuXi'")
        print("  3. Or update the import in fuxi_train.py line 48")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\n" + "=" * 70)
    print("Testing GPU availability...")
    print("=" * 70)
    
    import torch
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        print("  Training will run on CPU (very slow)")
        return False
    
    n_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available with {n_gpus} GPU(s)")
    
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    
    # Test tensor creation
    try:
        x = torch.randn(2, 3, 4, 5).cuda()
        y = x * 2
        print(f"✓ GPU computation test passed")
        return True
    except Exception as e:
        print(f"✗ GPU computation test failed: {e}")
        return False


def test_zarr_access(zarr_path):
    """Test Zarr store access."""
    print("\n" + "=" * 70)
    print("Testing Zarr store access...")
    print("=" * 70)
    
    if not zarr_path:
        print("⚠ No zarr-store path provided, skipping")
        return True
    
    if not os.path.exists(zarr_path):
        print(f"✗ Zarr store not found: {zarr_path}")
        return False
    
    print(f"✓ Zarr store exists: {zarr_path}")
    
    try:
        import zarr
        store = zarr.open_group(zarr_path, mode='r')
        print(f"✓ Zarr store opened successfully")
        
        # Check for expected variables
        expected = ['time', 'latitude', 'longitude', 'level', 
                   'temperature', '2m_temperature']
        found = []
        missing = []
        
        for var in expected:
            alt_var = var.replace('latitude', 'lat').replace('longitude', 'lon')
            if var in store or alt_var in store:
                found.append(var)
            else:
                missing.append(var)
        
        print(f"✓ Found {len(found)}/{len(expected)} expected variables")
        if found:
            print(f"  Present: {', '.join(found[:5])}...")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
        
        # Check time coordinate
        if 'time' in store:
            time_arr = store['time']
            print(f"✓ Time dimension: {len(time_arr)} timesteps")
        
        # Check spatial dimensions
        lat = store['latitude'][:] if 'latitude' in store else store['lat'][:]
        lon = store['longitude'][:] if 'longitude' in store else store['lon'][:]
        print(f"✓ Spatial grid: {len(lat)} lat × {len(lon)} lon")
        
        return True
        
    except Exception as e:
        print(f"✗ Zarr access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation(zarr_path):
    """Test dataset creation."""
    print("\n" + "=" * 70)
    print("Testing dataset creation...")
    print("=" * 70)
    
    if not zarr_path:
        print("⚠ No zarr-store path provided, skipping")
        return True
    
    try:
        # Import the dataset class
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import from the training script
        import importlib.util
        spec = importlib.util.spec_from_file_location("fuxi_train", "fuxi_train.py")
        fuxi_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fuxi_train)
        
        FuXiZarrDataset = fuxi_train.FuXiZarrDataset
        
        print("Creating dataset (this may take a minute)...")
        dataset = FuXiZarrDataset(
            zarr_path,
            history_steps=2,
            time_start="2020-01-01",
            time_end="2020-01-31",
            stats_subsample=50,  # Small sample for speed
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Samples: {len(dataset)}")
        print(f"  Channels: {dataset.channels}")
        print(f"  Spatial: {dataset.spatial_shape}")
        
        # Test loading a sample
        print("\nTesting sample loading...")
        history, target = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  History shape: {history.shape}")
        print(f"  Target shape: {target.shape}")
        
        # Check for NaNs
        if history.isnan().any() or target.isnan().any():
            print("✗ WARNING: NaN values detected in data")
            return False
        
        print(f"✓ No NaN values detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model instantiation."""
    print("\n" + "=" * 70)
    print("Testing model creation...")
    print("=" * 70)
    
    try:
        import torch
        from src.models.fuxi_model import FuXi
        
        # Create a small model
        model = FuXi(
            num_variables=20,
            embed_dim=64,  # Small for testing
            num_heads=4,
            window_size=5,
            depth_pre=1,
            depth_mid=2,
            depth_post=1,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            input_height=121,
            input_width=240,
            use_checkpoint=False,
        )
        
        print(f"✓ Model created successfully")
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        x = torch.randn(batch_size, 20, 2, 121, 240)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            print("  Using GPU")
        
        with torch.no_grad():
            y = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape : {tuple(x.shape)}")
        print(f"  Output shape: {tuple(y.shape)}")
        
        expected_shape = (batch_size, 20, 121, 240)
        if y.shape == expected_shape:
            print(f"✓ Output shape correct: {expected_shape}")
        else:
            print(f"✗ Output shape mismatch!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {y.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation/forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_iteration(zarr_path):
    """Test a single training iteration."""
    print("\n" + "=" * 70)
    print("Testing training iteration...")
    print("=" * 70)
    
    if not zarr_path:
        print("⚠ No zarr-store path provided, skipping")
        return True
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.cuda.amp import autocast, GradScaler
        
        # Import from training script
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("fuxi_train", "fuxi_train.py")
        fuxi_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fuxi_train)
        
        FuXiZarrDataset = fuxi_train.FuXiZarrDataset
        LatitudeWeightedL1Loss = fuxi_train.LatitudeWeightedL1Loss
        
        from src.models.fuxi_model import FuXi
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        
        # Create tiny dataset
        print("\nCreating dataset...")
        dataset = FuXiZarrDataset(
            zarr_path,
            history_steps=2,
            time_start="2020-01-01",
            time_end="2020-01-03",
            stats_subsample=20,
        )
        
        history, target = dataset[0]
        history = history.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # Create small model
        print("Creating model...")
        model = FuXi(
            num_variables=dataset.channels,
            embed_dim=64,
            num_heads=4,
            window_size=5,
            depth_pre=1,
            depth_mid=2,
            depth_post=1,
            mlp_ratio=4.0,
            drop_path_rate=0.0,
            input_height=dataset.spatial_shape[0],
            input_width=dataset.spatial_shape[1],
            use_checkpoint=False,
        ).to(device)
        
        # Create optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = LatitudeWeightedL1Loss(
            num_lat=dataset.spatial_shape[0],
            lat_range=(-90, 90),
        ).to(device)
        
        scaler = GradScaler(enabled=(device.type == "cuda"))
        
        # Training iteration
        print("\nRunning training iteration...")
        model.train()
        optimizer.zero_grad()
        
        with autocast(enabled=(device.type == "cuda")):
            pred = model(history)
            loss = criterion(pred, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✓ Training iteration successful")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Prediction shape: {pred.shape}")
        
        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        if has_grads:
            print(f"✓ Gradients computed")
        else:
            print(f"✗ No gradients found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Training iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FuXi training setup")
    parser.add_argument("--zarr-store", type=str, default=None,
                       help="Path to Zarr store (optional)")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("FuXi Training Setup Verification")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['gpu'] = test_gpu()
    results['zarr_access'] = test_zarr_access(args.zarr_store)
    results['dataset'] = test_dataset_creation(args.zarr_store)
    results['model'] = test_model_creation()
    results['training'] = test_training_iteration(args.zarr_store)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test}")
    
    print("\n" + "=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Review README.md for usage instructions")
        print("  2. Update fuxi_sweep.sh with your paths")
        print("  3. Run: ./fuxi_sweep.sh")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Imports: Install missing packages with conda/pip")
        print("  - Model: Check model.py exists and has FuXi class")
        print("  - Zarr: Verify path and permissions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
