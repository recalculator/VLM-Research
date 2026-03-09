"""
Direct loader for PruMerge clip_encoder to avoid import conflicts.

This module directly loads the clip_encoder.py file without going through
the llava package initialization, which has conflicts with FastV's transformers.
"""

import sys
import importlib.util
from pathlib import Path
import config

def load_clip_encoder_module():
    """
    Directly load clip_encoder.py as a module without package initialization.

    Returns:
        module: The clip_encoder module
    """
    # IMPORTANT: Remove FastV's transformers from sys.path temporarily
    # to avoid version conflicts
    fastv_paths = [p for p in sys.path if 'FastV' in p and 'transformers' in p]
    for p in fastv_paths:
        sys.path.remove(p)

    try:
        clip_encoder_path = (
            config.get_prumerge_path() /
            "llava" / "model" / "multimodal_encoder" / "clip_encoder.py"
        )

        if not clip_encoder_path.exists():
            raise FileNotFoundError(f"clip_encoder.py not found at {clip_encoder_path}")

        # Load module directly from file
        spec = importlib.util.spec_from_file_location("clip_encoder", clip_encoder_path)
        module = importlib.util.module_from_spec(spec)

        # Execute module
        spec.loader.exec_module(module)

        return module
    finally:
        # Restore FastV paths (in case needed later)
        for p in fastv_paths:
            if p not in sys.path:
                sys.path.insert(0, p)


def get_prumerge_vision_tower():
    """
    Load and return a PruMerge vision tower without package import conflicts.

    Returns:
        CLIPVisionTower instance
    """
    # Load the clip_encoder module directly
    clip_encoder = load_clip_encoder_module()

    # Create minimal args object with required attributes
    class Args:
        mm_vision_select_layer = -2  # Use penultimate layer
        mm_vision_select_feature = 'patch'  # Use patch tokens only

    args = Args()

    # Create vision tower
    vision_tower = clip_encoder.CLIPVisionTower(
        vision_tower="openai/clip-vit-large-patch14-336",
        args=args,
        delay_load=False
    )

    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower.eval()

    return vision_tower, clip_encoder


if __name__ == "__main__":
    print("Testing direct PruMerge loader...")
    try:
        vision_tower, clip_encoder_module = get_prumerge_vision_tower()
        print(f"✓ Successfully loaded PruMerge vision tower")
        print(f"  Device: {vision_tower.device}")
        print(f"  Model: {vision_tower.vision_tower_name}")
        print(f"  Global variable available: {hasattr(clip_encoder_module, 'kept_token_indices')}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
