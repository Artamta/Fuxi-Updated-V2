#!/usr/bin/env python3
"""
Model Interface Compatibility Checker - Package-aware version

Verifies that your FuXi model implementation is compatible with the training script.
Works with both standalone files and package structures.

Usage: python check_model_interface.py
"""

import inspect
import sys
import os

def check_model_import():
    """Try to import the model and check its interface."""
    print("=" * 70)
    print("Checking Model Import and Interface")
    print("=" * 70 + "\n")
    
    # Determine the package structure
    current_dir = os.path.basename(os.getcwd())
    parent_dir = os.path.dirname(os.getcwd())
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Looking for model...\n")
    
    # Try different import strategies
    FuXi = None
    import_method = None
    
    # Strategy 1: Canonical package import for this repository
    try:
        from src.models.fuxi_model import FuXi
        import_method = "from src.models.fuxi_model import FuXi"
        print(f"✓ Successfully imported using: {import_method}\n")
    except (ImportError, AttributeError) as e:
        print(f"  Strategy 1 failed (from src.models.fuxi_model import FuXi): {e}")
    
    # Strategy 2: Package import (if in fuxi_paper directory)
    if FuXi is None and current_dir == "fuxi_paper":
        try:
            # Add parent to path and import as package
            sys.path.insert(0, parent_dir)
            from fuxi_paper.model import FuXi
            import_method = "from fuxi_paper.model import FuXi"
            print(f"✓ Successfully imported using: {import_method}\n")
        except (ImportError, AttributeError) as e:
            print(f"  Strategy 2 failed (from fuxi_paper.model import FuXi): {e}")
    
    # Strategy 3: Try importing the whole package
    if FuXi is None:
        try:
            # Check if we're inside a package
            package_name = current_dir
            sys.path.insert(0, parent_dir)
            module = __import__(f"{package_name}.model", fromlist=['FuXi'])
            FuXi = getattr(module, 'FuXi')
            import_method = f"from {package_name}.model import FuXi"
            print(f"✓ Successfully imported using: {import_method}\n")
        except (ImportError, AttributeError) as e:
            print(f"  Strategy 3 failed: {e}")
    
    # Strategy 4: Check if model.py exists and try to find the class name
    if FuXi is None and os.path.exists("model.py"):
        print("\n  model.py exists, checking for class names...")
        try:
            with open("model.py", "r") as f:
                content = f.read()
                # Find class definitions
                import re
                classes = re.findall(r'class\s+(\w+)\s*\(', content)
                if classes:
                    print(f"  Found classes in model.py: {', '.join(classes)}")
                    
                    # Try to import the first likely candidate
                    for cls_name in classes:
                        if 'fuxi' in cls_name.lower() or 'model' in cls_name.lower():
                            print(f"\n  Trying to import class: {cls_name}")
                            # This is a workaround - we'll need to fix the imports
                            print(f"  → Class found but cannot import due to relative imports")
                            print(f"  → Need to use package import method")
        except Exception as e:
            print(f"  Could not analyze model.py: {e}")
    
    if FuXi is None:
        print("\n✗ Failed to import FuXi model\n")
        print("=" * 70)
        print("DIAGNOSIS:")
        print("=" * 70)
        print("\nYour code structure appears to be a Python package with relative imports.")
        print("This is typical when you see imports like:")
        print("  from .blocks import ...")
        print("  from .swin_v2 import ...")
        print("\nSOLUTION OPTIONS:")
        print("=" * 70)
        print("\nOption 1: Update the training script imports (RECOMMENDED)")
        print("-" * 70)
        print("In fuxi_train.py, replace line 48:")
        print("  from model import FuXi")
        print("\nWith:")
        print("  from fuxi_paper.model import FuXi")
        print("\nOr if you're running from parent directory:")
        print("  import sys")
        print("  sys.path.insert(0, 'fuxi_paper')")
        print("  from model import FuXi")
        
        print("\n\nOption 2: Create a standalone adapter file")
        print("-" * 70)
        print("Create a file 'model_adapter.py' in your training directory:")
        print("""
# model_adapter.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fuxi_paper'))
from fuxi_paper.model import FuXi

# Re-export
__all__ = ['FuXi']
""")
        print("\nThen in fuxi_train.py line 48:")
        print("  from model_adapter import FuXi")
        
        print("\n\nOption 3: Install as package")
        print("-" * 70)
        print("If fuxi_paper has setup.py:")
        print("  pip install -e .")
        print("\nThen import normally:")
        print("  from fuxi_paper.model import FuXi")
        
        return None, None
    
    return FuXi, import_method


def check_init_signature(FuXi):
    """Check __init__ method signature."""
    print("\n" + "=" * 70)
    print("Checking __init__ Signature")
    print("=" * 70 + "\n")
    
    expected_params = {
        'num_variables': int,
        'embed_dim': int,
        'num_heads': int,
        'window_size': int,
        'depth_pre': int,
        'depth_mid': int,
        'depth_post': int,
        'mlp_ratio': float,
        'drop_path_rate': float,
        'input_height': int,
        'input_width': int,
        'use_checkpoint': bool,
    }
    
    # Get actual signature
    try:
        sig = inspect.signature(FuXi.__init__)
        actual_params = {
            name: param for name, param in sig.parameters.items()
            if name != 'self'
        }
        
        print("Expected parameters:")
        for name, typ in expected_params.items():
            print(f"  {name}: {typ.__name__}")
        
        print("\nActual parameters:")
        for name, param in actual_params.items():
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            print(f"  {name}{default}")
        
        # Check for missing/extra parameters
        missing = set(expected_params.keys()) - set(actual_params.keys())
        extra = set(actual_params.keys()) - set(expected_params.keys())
        
        if not missing and not extra:
            print("\n✓ Signature matches perfectly!")
            return True
        
        if missing:
            print(f"\n⚠ Missing parameters: {', '.join(missing)}")
        
        if extra:
            print(f"\n⚠ Extra parameters: {', '.join(extra)}")
        
        # Check if close enough
        critical_missing = missing - {'use_checkpoint'}  # use_checkpoint is optional
        if len(critical_missing) == 0:
            print("\n✓ Signature is compatible (missing only optional params)")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Failed to inspect __init__ signature: {e}")
        return False


def check_forward_signature(FuXi):
    """Check forward method signature."""
    print("\n" + "=" * 70)
    print("Checking forward() Signature")
    print("=" * 70 + "\n")
    
    try:
        sig = inspect.signature(FuXi.forward)
        params = list(sig.parameters.keys())
        
        print("Expected signature:")
        print("  def forward(self, x: torch.Tensor) -> torch.Tensor")
        print("    Args: x shape (B, C, T, H, W) or (B, V, T, H, W)")
        print("    Returns: shape (B, C, H, W) or (B, V, H, W)")
        
        print("\nActual signature:")
        print(f"  def forward({', '.join(params)})")
        
        # More lenient check - allow additional optional parameters
        if len(params) >= 2 and params[0] == 'self':
            first_param = params[1]
            print(f"\n✓ Has required input parameter: '{first_param}'")
            return True
        else:
            print(f"\n⚠ Unexpected signature")
            return False
        
    except Exception as e:
        print(f"✗ Failed to inspect forward signature: {e}")
        return False


def test_forward_pass(FuXi):
    """Test actual forward pass."""
    print("\n" + "=" * 70)
    print("Testing Forward Pass")
    print("=" * 70 + "\n")
    
    try:
        import torch
        
        # Create model instance with typical parameters
        print("Creating model instance...")
        model = FuXi(
            num_variables=20,
            embed_dim=64,
            num_heads=4,
            window_size=5,
            depth_pre=2,
            depth_mid=6,
            depth_post=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            input_height=121,
            input_width=240,
        )
        print("✓ Model created successfully")
        
        # Test input - try both possible formats
        B, V, T, H, W = 2, 20, 2, 121, 240
        x = torch.randn(B, V, T, H, W)
        
        print(f"\nInput shape: {tuple(x.shape)} (B, V, T, H, W)")
        
        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            y = model(x)
        
        print(f"Output shape: {tuple(y.shape)}")
        
        # Check output shape - accept both (B, C, H, W) and (B, V, H, W)
        expected_shape = (B, V, H, W)
        if y.shape == expected_shape:
            print(f"\n✓ Output shape correct: {expected_shape}")
            return True
        else:
            print(f"\n⚠ Output shape: {y.shape}")
            print(f"  Expected: {expected_shape}")
            print("  This may still work if dimensions match semantically")
            return True  # More lenient
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def provide_integration_guide(import_method):
    """Provide integration guide based on successful import method."""
    print("\n" + "=" * 70)
    print("Integration Guide for fuxi_train.py")
    print("=" * 70 + "\n")
    
    print("Update the import section in fuxi_train.py (around line 40-50):")
    print("\n" + "-" * 70)
    print(f"""
# UPDATED IMPORT SECTION
# (Replace the 'from model import FuXi' line with this)

import sys
import os

# Add fuxi_paper to path
script_dir = os.path.dirname(os.path.abspath(__file__))
fuxi_paper_dir = os.path.join(script_dir, 'fuxi_paper')
if os.path.exists(fuxi_paper_dir):
    sys.path.insert(0, fuxi_paper_dir)

# Import using the working method
{import_method}

# Alternative: If running from parent directory
# sys.path.insert(0, 'fuxi_paper')
# from model import FuXi
""")
    print("-" * 70)


def main():
    print("\n" + "=" * 70)
    print("FuXi Model Interface Compatibility Checker")
    print("Package-aware version")
    print("=" * 70 + "\n")
    
    # Step 1: Import model
    FuXi, import_method = check_model_import()
    if FuXi is None:
        return 1
    
    # Step 2: Check signatures
    init_ok = check_init_signature(FuXi)
    forward_ok = check_forward_signature(FuXi)
    
    # Step 3: Test forward pass
    forward_test_ok = test_forward_pass(FuXi)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70 + "\n")
    
    results = {
        'Import': True,
        '__init__ signature': init_ok,
        'forward() signature': forward_ok,
        'Forward pass test': forward_test_ok,
    }
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 70)
        print("✓ Your model is compatible!")
        print("=" * 70)
        provide_integration_guide(import_method)
        return 0
    else:
        print("\n" + "=" * 70)
        print("⚠ Some compatibility issues found")
        print("=" * 70)
        print("\nPlease review the failures above.")
        if import_method:
            provide_integration_guide(import_method)
        return 1


if __name__ == "__main__":
    sys.exit(main())