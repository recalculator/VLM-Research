#!/usr/bin/env python3
"""
Visual Token Pruning Comparison - Main Experiment Runner

This script runs the complete end-to-end experiment comparing FastV and
LLaVA-PruMerge visual token pruning strategies.

Usage:
    python run_experiment.py                    # Run full experiment (if models available)
    python run_experiment.py --mode demo        # Run PruMerge-only demo
    python run_experiment.py --images 3         # Use 3 images
    python run_experiment.py --no-viz           # Skip visualizations
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import json

# Import configuration
import config
import direct_prumerge_loader

# Setup paths
print("Setting up paths...")
config.setup_python_paths()

# Verify setup
repo_status = config.check_repositories_exist()
mod_status = config.check_modifications_applied()

if not repo_status['all_present']:
    print("ERROR: Required repositories not found!")
    print(f"  FastV: {repo_status['fastv']}")
    print(f"  PruMerge: {repo_status['prumerge']}")
    print("\nPlease clone the repositories first.")
    sys.exit(1)

if not mod_status.get('all_modified', False):
    print("WARNING: Modifications may not be applied!")
    print(f"  PruMerge modified: {mod_status.get('prumerge_modified', False)}")
    print(f"  FastV modified: {mod_status.get('fastv_modified', False)}")
    print("\nSee MODIFICATIONS.md for details.")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def download_image(url: str) -> Image.Image:
    """Download image from URL"""
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content))
    return img.convert("RGB")


def compute_jaccard_similarity(set_a: List[int], set_b: List[int]) -> float:
    """Compute Jaccard similarity: |A ∩ B| / |A ∪ B|"""
    set_a = set(set_a)
    set_b = set(set_b)

    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def compute_jaccard_stats(set_a: List[int], set_b: List[int]) -> Dict:
    """Compute detailed Jaccard statistics"""
    set_a = set(set_a)
    set_b = set(set_b)

    intersection = set_a & set_b
    union = set_a | set_b

    return {
        "jaccard": len(intersection) / len(union) if len(union) > 0 else 0.0,
        "intersection_size": len(intersection),
        "union_size": len(union),
        "fastv_count": len(set_a),
        "prumerge_count": len(set_b),
        "only_fastv": len(set_a - set_b),
        "only_prumerge": len(set_b - set_a),
    }


def visualize_token_overlap(
    fastv_indices: List[int],
    prumerge_indices: List[int],
    title: str = "Token Selection Comparison",
    save_path: Optional[str] = None
):
    """Visualize token selection on 24x24 grid"""
    h, w = 24, 24

    # Create grids
    fastv_grid = np.zeros((h, w))
    prumerge_grid = np.zeros((h, w))

    for idx in fastv_indices:
        row, col = divmod(int(idx), w)
        if row < h and col < w:
            fastv_grid[row, col] = 1

    for idx in prumerge_indices:
        row, col = divmod(int(idx), w)
        if row < h and col < w:
            prumerge_grid[row, col] = 1

    # Create overlap grid
    overlap_grid = fastv_grid + prumerge_grid * 2

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # FastV
    axes[0].imshow(fastv_grid, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title(f'FastV\n{len(fastv_indices)} tokens', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # PruMerge
    axes[1].imshow(prumerge_grid, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title(f'PruMerge\n{len(prumerge_indices)} tokens', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Overlap
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red', 'blue', 'purple']
    cmap = ListedColormap(colors)

    axes[2].imshow(overlap_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)

    both = np.sum(overlap_grid == 3)
    only_fastv = np.sum(overlap_grid == 1)
    only_prumerge = np.sum(overlap_grid == 2)

    axes[2].set_title(
        f'Overlap\nBoth: {both} | FastV only: {only_fastv} | PruMerge only: {only_prumerge}',
        fontsize=14, fontweight='bold'
    )
    axes[2].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved visualization: {save_path}")

    plt.close(fig)
    return fig


# ============================================================================
# PRUMERGE INFERENCE
# ============================================================================

def load_prumerge_model():
    """Load LLaVA-PruMerge vision encoder"""
    print("\nLoading PruMerge model...")

    # Use direct loader to avoid import conflicts
    vision_tower, clip_encoder_module = direct_prumerge_loader.get_prumerge_vision_tower()

    print(f"  ✓ PruMerge loaded")
    print(f"    Device: {vision_tower.device}")
    print(f"    Model: {vision_tower.vision_tower_name}")

    return vision_tower, clip_encoder_module


def run_prumerge_inference(vision_tower, clip_encoder_module, image: Image.Image) -> List[int]:
    """Run PruMerge and extract kept token indices"""
    # Prepare image
    image_tensor = vision_tower.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(vision_tower.device, dtype=vision_tower.dtype)

    # Run inference
    with torch.no_grad():
        _ = vision_tower.token_prune_merge_advanced_plus(
            image_tensor,
            if_adaptive=True,
            reduction_ratio=1/8
        )

    # Extract indices from global variable
    kept_indices = clip_encoder_module.kept_token_indices["indices"]

    if kept_indices is None:
        raise RuntimeError("Failed to capture PruMerge indices")

    # Convert to list
    if len(kept_indices.shape) > 1:
        kept_indices = kept_indices[0]

    indices_list = kept_indices.numpy().tolist()
    return indices_list


# ============================================================================
# FASTV INFERENCE
# ============================================================================

def load_fastv_model():
    """Load FastV model (LLaVA with FastV modifications)"""
    print("\nLoading FastV model...")
    print("  Note: This requires LLaVA model weights (~13GB)")

    try:
        # Import LLaVA builder
        # This might fail if full LLaVA is not installed
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_path = "liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)

        print(f"  Loading from: {model_path}")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="auto"
        )

        # Configure FastV
        if hasattr(model.config, 'use_fast_v'):
            model.config.use_fast_v = True
            model.config.fast_v_sys_length = 35
            model.config.fast_v_image_token_length = 576
            model.config.fast_v_attention_rank = 144
            model.config.fast_v_agg_layer = 15
            model.config.fast_v_inplace = False

            print(f"  ✓ FastV loaded and configured")
            print(f"    Attention rank: {model.config.fast_v_attention_rank}")
            print(f"    Aggregation layer: {model.config.fast_v_agg_layer}")
        else:
            print("  ⚠ Warning: Model doesn't have FastV configuration")

        return tokenizer, model, image_processor, context_len

    except Exception as e:
        print(f"  ✗ Failed to load FastV: {e}")
        print(f"    This is expected if LLaVA model weights are not available")
        return None, None, None, None


def run_fastv_inference(model, tokenizer, image_processor, image: Image.Image) -> Optional[List[int]]:
    """Run FastV and extract kept token indices"""
    if model is None:
        return None

    try:
        # Import FastV's modified modeling_llama
        from transformers.models.llama import modeling_llama
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX

        # Prepare conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], "<image>\nDescribe this image briefly.")
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Process image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # Run inference
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=20,
                use_cache=True,
            )

        # Extract indices from global variable
        kept_indices = modeling_llama.kept_visual_token_indices["indices"]

        if kept_indices is None:
            raise RuntimeError("FastV indices not captured")

        indices_list = kept_indices.numpy().tolist()
        return indices_list

    except Exception as e:
        print(f"    ✗ FastV inference failed: {e}")
        return None


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    vision_tower,
    clip_encoder_module,
    fastv_model,
    fastv_tokenizer,
    fastv_image_processor,
    image_sources: List[Tuple[str, str]],
    visualize: bool = True
) -> List[Dict]:
    """Run experiment on multiple images"""

    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT")
    print("=" * 80)

    all_results = []
    viz_path = config.get_visualizations_path()

    for i, (name, source) in enumerate(image_sources, 1):
        print(f"\n[{i}/{len(image_sources)}] Processing: {name}")
        print("-" * 80)

        # Load image
        try:
            if source.startswith("http"):
                image = download_image(source)
            else:
                image = Image.open(source).convert("RGB")
            print(f"  ✓ Image loaded: {image.size}")
        except Exception as e:
            print(f"  ✗ Failed to load image: {e}")
            continue

        result = {"name": name, "source": source}

        # Run PruMerge
        try:
            prumerge_indices = run_prumerge_inference(vision_tower, clip_encoder_module, image)
            result["prumerge_indices"] = prumerge_indices
            result["prumerge_success"] = True
            print(f"  ✓ PruMerge: kept {len(prumerge_indices)} tokens")
        except Exception as e:
            print(f"  ✗ PruMerge failed: {e}")
            result["prumerge_success"] = False
            result["prumerge_indices"] = []
            continue

        # Run FastV
        if fastv_model is not None:
            try:
                fastv_indices = run_fastv_inference(fastv_model, fastv_tokenizer, fastv_image_processor, image)
                if fastv_indices is not None:
                    result["fastv_indices"] = fastv_indices
                    result["fastv_success"] = True
                    print(f"  ✓ FastV: kept {len(fastv_indices)} tokens")
                else:
                    result["fastv_success"] = False
                    result["fastv_indices"] = []
            except Exception as e:
                print(f"  ✗ FastV failed: {e}")
                result["fastv_success"] = False
                result["fastv_indices"] = []
        else:
            result["fastv_success"] = False
            result["fastv_indices"] = []

        # Compute Jaccard if both succeeded
        if result["prumerge_success"] and result.get("fastv_success", False):
            stats = compute_jaccard_stats(result["fastv_indices"], result["prumerge_indices"])
            result.update(stats)

            print(f"\n  RESULTS:")
            print(f"    FastV tokens:       {stats['fastv_count']}")
            print(f"    PruMerge tokens:    {stats['prumerge_count']}")
            print(f"    Intersection:       {stats['intersection_size']}")
            print(f"    Union:              {stats['union_size']}")
            print(f"    Jaccard similarity: {stats['jaccard']:.4f}")

            # Visualize
            if visualize:
                viz_file = viz_path / f"{name.replace(' ', '_')}_comparison.png"
                visualize_token_overlap(
                    result["fastv_indices"],
                    result["prumerge_indices"],
                    title=f"Token Selection: {name}",
                    save_path=str(viz_file)
                )
        else:
            result["jaccard"] = None
            print(f"  ⚠ Skipping Jaccard (one or both methods failed)")

        all_results.append(result)

    return all_results


def print_summary(results: List[Dict]):
    """Print experiment summary"""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    valid_results = [r for r in results if r.get("jaccard") is not None]

    print(f"\nTotal images processed: {len(results)}")
    print(f"Successful comparisons: {len(valid_results)}")

    if len(valid_results) == 0:
        print("\nNo valid comparisons. Running in demo mode (PruMerge only):")
        prumerge_only = [r for r in results if r.get("prumerge_success", False)]
        print(f"\n{'Image':<30} {'Tokens':>10}")
        print("-" * 80)
        for r in prumerge_only:
            print(f"{r['name']:<30} {len(r.get('prumerge_indices', [])):>10}")
        return

    # Statistics
    jaccards = [r["jaccard"] for r in valid_results]
    fastv_counts = [r["fastv_count"] for r in valid_results]
    prumerge_counts = [r["prumerge_count"] for r in valid_results]
    intersections = [r["intersection_size"] for r in valid_results]

    print(f"\nJaccard Similarity Statistics:")
    print(f"  Mean:   {np.mean(jaccards):.4f}")
    print(f"  Median: {np.median(jaccards):.4f}")
    print(f"  Std:    {np.std(jaccards):.4f}")
    print(f"  Min:    {np.min(jaccards):.4f}")
    print(f"  Max:    {np.max(jaccards):.4f}")

    print(f"\nToken Count Statistics:")
    print(f"  FastV (mean):        {np.mean(fastv_counts):.1f} ± {np.std(fastv_counts):.1f}")
    print(f"  PruMerge (mean):     {np.mean(prumerge_counts):.1f} ± {np.std(prumerge_counts):.1f}")
    print(f"  Intersection (mean): {np.mean(intersections):.1f} ± {np.std(intersections):.1f}")

    print(f"\nPer-Image Results:")
    print(f"{'Image':<30} {'FastV':>8} {'PruMerge':>10} {'Overlap':>8} {'Jaccard':>10}")
    print("-" * 80)
    for r in valid_results:
        print(f"{r['name']:<30} {r['fastv_count']:>8} {r['prumerge_count']:>10} "
              f"{r['intersection_size']:>8} {r['jaccard']:>10.4f}")


def save_results(results: List[Dict], filepath: str):
    """Save results to JSON file"""
    # Remove numpy types for JSON serialization
    serializable_results = []
    for r in results:
        r_copy = r.copy()
        if 'prumerge_indices' in r_copy:
            r_copy['prumerge_indices'] = [int(x) for x in r_copy['prumerge_indices']]
        if 'fastv_indices' in r_copy:
            r_copy['fastv_indices'] = [int(x) for x in r_copy['fastv_indices']]
        serializable_results.append(r_copy)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visual Token Pruning Comparison Experiment")
    parser.add_argument("--mode", choices=["full", "demo"], default="full",
                        help="Experiment mode: 'full' (both methods) or 'demo' (PruMerge only)")
    parser.add_argument("--images", type=int, default=3,
                        help="Number of images to process (default: 3)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("VISUAL TOKEN PRUNING COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Images: {args.images}")
    print(f"Visualizations: {'disabled' if args.no_viz else 'enabled'}")

    # Sample images from COCO
    all_images = [
        ("cats", "http://images.cocodataset.org/val2017/000000039769.jpg"),
        ("tennis", "http://images.cocodataset.org/val2017/000000397133.jpg"),
        ("pizza", "http://images.cocodataset.org/val2017/000000252219.jpg"),
        ("baseball", "http://images.cocodataset.org/val2017/000000087038.jpg"),
        ("giraffe", "http://images.cocodataset.org/val2017/000000174482.jpg"),
    ]

    image_sources = all_images[:args.images]

    # Load PruMerge
    vision_tower, clip_encoder_module = load_prumerge_model()

    # Load FastV (only in full mode)
    fastv_model = None
    fastv_tokenizer = None
    fastv_image_processor = None

    if args.mode == "full":
        fastv_tokenizer, fastv_model, fastv_image_processor, _ = load_fastv_model()
        if fastv_model is None:
            print("\n⚠ FastV failed to load. Running in demo mode (PruMerge only).")
    else:
        print("\n⚠ Running in demo mode (PruMerge only).")

    # Run experiment
    results = run_experiment(
        vision_tower,
        clip_encoder_module,
        fastv_model,
        fastv_tokenizer,
        fastv_image_processor,
        image_sources,
        visualize=not args.no_viz
    )

    # Print summary
    print_summary(results)

    # Save results
    output_path = config.get_outputs_path() / "results.json"
    save_results(results, str(output_path))

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Results: {output_path}")
    if not args.no_viz:
        print(f"  Visualizations: {config.get_visualizations_path()}")


if __name__ == "__main__":
    main()
