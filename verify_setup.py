#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that the experimental environment is correctly configured.
"""

import sys
from pathlib import Path
import config


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_repositories():
    """Check if repositories are cloned"""
    print("\nChecking repositories...")
    status = config.check_repositories_exist()

    if status['fastv']:
        print(f"  ✓ FastV found at: {config.get_fastv_path()}")
    else:
        print(f"  ✗ FastV not found at: {config.get_fastv_path()}")

    if status['prumerge']:
        print(f"  ✓ LLaVA-PruMerge found at: {config.get_prumerge_path()}")
    else:
        print(f"  ✗ LLaVA-PruMerge not found at: {config.get_prumerge_path()}")

    return status['all_present']


def check_modifications():
    """Check if code modifications are applied"""
    print("\nChecking modifications...")
    status = config.check_modifications_applied()

    if status.get('prumerge_modified', False):
        print("  ✓ PruMerge modifications applied")
    else:
        print("  ✗ PruMerge modifications NOT applied")
        print("    See MODIFICATIONS.md for details")

    if status.get('fastv_modified', False):
        print("  ✓ FastV modifications applied")
    else:
        print("  ✗ FastV modifications NOT applied")
        print("    See MODIFICATIONS.md for details")

    return status.get('all_modified', False)


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking dependencies...")
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow (PIL)',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy',
        'requests': 'Requests'
    }

    all_present = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            all_present = False

    if not all_present:
        print("\n  Install missing dependencies:")
        print("    pip install -r requirements.txt")

    return all_present


def check_prumerge_loading():
    """Test if PruMerge can be loaded"""
    print("\nTesting PruMerge loading...")
    try:
        import direct_prumerge_loader
        vision_tower, clip_encoder = direct_prumerge_loader.get_prumerge_vision_tower()
        print(f"  ✓ PruMerge loads successfully")
        print(f"    Device: {vision_tower.device}")
        print(f"    Model: {vision_tower.vision_tower_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load PruMerge: {e}")
        return False


def main():
    print("=" * 80)
    print("VLM RESEARCH SETUP VERIFICATION")
    print("=" * 80)

    results = {}
    results['python'] = check_python_version()
    results['repositories'] = check_repositories()
    results['modifications'] = check_modifications()
    results['dependencies'] = check_dependencies()
    results['prumerge'] = check_prumerge_loading()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_checks_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check.capitalize()}")

    print("\n" + "=" * 80)

    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - Ready to run experiments!")
        print("\nTry:")
        print("  python run_experiment.py --mode demo --images 3")
    else:
        print("✗ SOME CHECKS FAILED - See above for details")
        print("\nRefer to README.md and MODIFICATIONS.md for setup instructions")

    print("=" * 80)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
