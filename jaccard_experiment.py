"""
Visual Token Pruning Comparison: FastV vs LLaVA-PruMerge
=========================================================

This script compares visual token pruning strategies between FastV and LLaVA-PruMerge
by computing Jaccard similarity on the selected token sets.

Author: Generated for research experiment
Date: 2024
"""

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple
import requests
from io import BytesIO

# Add repository paths
WORK_DIR = Path("/Users/ayaanchawla/Research")
FASTV_PATH = WORK_DIR / "FastV"
PRUMERGE_PATH = WORK_DIR / "LLaVA-PruMerge"

# Add to Python path
sys.path.insert(0, str(FASTV_PATH / "src" / "transformers" / "src"))
sys.path.insert(0, str(FASTV_PATH / "src" / "FastV"))
sys.path.insert(0, str(PRUMERGE_PATH))

print("=" * 80)
print("Visual Token Pruning Comparison Experiment")
print("=" * 80)
print(f"FastV path: {FASTV_PATH}")
print(f"PruMerge path: {PRUMERGE_PATH}")
print()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    only_a = set_a - set_b
    only_b = set_b - set_a

    return {
        "jaccard": len(intersection) / len(union) if len(union) > 0 else 0.0,
        "intersection_size": len(intersection),
        "union_size": len(union),
        "set_a_size": len(set_a),
        "set_b_size": len(set_b),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
    }


def download_image(url: str) -> Image.Image:
    """Download image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.convert("RGB")


def load_local_image(path: str) -> Image.Image:
    """Load image from local path"""
    img = Image.open(path)
    return img.convert("RGB")


def visualize_token_overlap(
    fastv_indices: List[int],
    prumerge_indices: List[int],
    grid_size: Tuple[int, int] = (24, 24),
    title: str = "Token Selection Comparison",
    save_path: str = None
):
    """
    Visualize token selection overlap on a 24x24 grid.

    Args:
        fastv_indices: Indices selected by FastV
        prumerge_indices: Indices selected by PruMerge
        grid_size: Grid dimensions (default 24x24)
        title: Plot title
        save_path: Optional path to save figure
    """
    h, w = grid_size

    # Create grids
    fastv_grid = np.zeros((h, w))
    prumerge_grid = np.zeros((h, w))

    # Fill grids
    for idx in fastv_indices:
        row, col = divmod(int(idx), w)
        if row < h and col < w:
            fastv_grid[row, col] = 1

    for idx in prumerge_indices:
        row, col = divmod(int(idx), w)
        if row < h and col < w:
            prumerge_grid[row, col] = 1

    # Create overlap grid: 0=neither, 1=FastV only, 2=PruMerge only, 3=both
    overlap_grid = fastv_grid + prumerge_grid * 2

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot FastV
    axes[0].imshow(fastv_grid, cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title(f'FastV\n{len(fastv_indices)} tokens', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot PruMerge
    axes[1].imshow(prumerge_grid, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title(f'PruMerge\n{len(prumerge_indices)} tokens', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Plot overlap
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red', 'blue', 'purple']
    cmap = ListedColormap(colors)

    im = axes[2].imshow(overlap_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)

    # Compute stats
    both = np.sum(overlap_grid == 3)
    only_fastv = np.sum(overlap_grid == 1)
    only_prumerge = np.sum(overlap_grid == 2)

    axes[2].set_title(
        f'Overlap\nBoth: {both} | FastV only: {only_fastv} | PruMerge only: {only_prumerge}',
        fontsize=14,
        fontweight='bold'
    )
    axes[2].axis('off')

    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    return fig


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_prumerge_model():
    """Load LLaVA-PruMerge model"""
    print("\n" + "=" * 80)
    print("Loading LLaVA-PruMerge Model...")
    print("=" * 80)

    # Import after path setup
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    # Initialize vision tower
    vision_tower = CLIPVisionTower(
        vision_tower="openai/clip-vit-large-patch14-336",
        args=None,
        delay_load=False
    )

    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower.eval()

    print(f"✓ PruMerge model loaded successfully")
    print(f"  - Vision tower: {vision_tower.vision_tower_name}")
    print(f"  - Device: {vision_tower.device}")
    print(f"  - Dtype: {vision_tower.dtype}")

    return vision_tower


def load_fastv_model():
    """Load FastV model (LLaVA with FastV modifications)"""
    print("\n" + "=" * 80)
    print("Loading FastV Model...")
    print("=" * 80)

    # Import after path setup
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    # Model configuration
    model_path = "liuhaotian/llava-v1.5-7b"  # Default LLaVA model
    model_name = get_model_name_from_path(model_path)

    print(f"Loading model: {model_path}")
    print(f"Model name: {model_name}")

    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="auto"
        )

        # Configure FastV settings
        if hasattr(model.config, 'use_fast_v'):
            model.config.use_fast_v = True
            model.config.fast_v_sys_length = 35  # Default system token length
            model.config.fast_v_image_token_length = 576  # 24x24 grid
            model.config.fast_v_attention_rank = 144  # Keep 25% of tokens
            model.config.fast_v_agg_layer = 15  # Aggregation layer
            model.config.fast_v_inplace = False

            print(f"✓ FastV model loaded successfully")
            print(f"  - Use FastV: {model.config.use_fast_v}")
            print(f"  - Attention rank: {model.config.fast_v_attention_rank}")
            print(f"  - Aggregation layer: {model.config.fast_v_agg_layer}")
        else:
            print("⚠ Warning: Model doesn't have FastV configuration")

        return tokenizer, model, image_processor, context_len

    except Exception as e:
        print(f"✗ Error loading FastV model: {e}")
        print("\nNote: This experiment requires the full model weights.")
        print("For a lightweight demo, we'll use only PruMerge.")
        return None, None, None, None


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def run_prumerge_inference(vision_tower, image: Image.Image) -> List[int]:
    """
    Run PruMerge inference and extract kept token indices.

    Returns:
        List of token indices in range [0, 575]
    """
    # Import the modified clip_encoder module to access global variable
    from llava.model.multimodal_encoder import clip_encoder

    # Prepare image
    image_tensor = vision_tower.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(vision_tower.device, dtype=vision_tower.dtype)

    # Run inference (this will populate the global variable)
    with torch.no_grad():
        _ = vision_tower.token_prune_merge_advanced_plus(
            image_tensor,
            if_adaptive=True,
            reduction_ratio=1/8
        )

    # Extract indices from global variable
    kept_indices = clip_encoder.kept_token_indices["indices"]
    reduction_ratio = clip_encoder.kept_token_indices["reduction_ratio"]

    if kept_indices is None:
        raise RuntimeError("Failed to capture PruMerge indices")

    # Convert to list (handle batch dimension)
    if len(kept_indices.shape) > 1:
        kept_indices = kept_indices[0]  # Take first batch

    indices_list = kept_indices.numpy().tolist()

    print(f"  PruMerge kept {len(indices_list)} tokens (reduction ratio: {reduction_ratio:.3f})")

    return indices_list


def run_fastv_inference(model, tokenizer, image_processor, image: Image.Image, prompt: str = "Describe this image.") -> List[int]:
    """
    Run FastV inference and extract kept token indices.

    Returns:
        List of token indices in range [0, 575]
    """
    # Import the modified modeling_llama module to access global variable
    from transformers.models.llama import modeling_llama

    # Prepare inputs
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    # Prepare conversation
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{prompt}")
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
            max_new_tokens=50,
            use_cache=True,
        )

    # Extract indices from global variable
    kept_indices = modeling_llama.kept_visual_token_indices["indices"]

    if kept_indices is None:
        raise RuntimeError("Failed to capture FastV indices")

    indices_list = kept_indices.numpy().tolist()

    print(f"  FastV kept {len(indices_list)} tokens")

    return indices_list


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_image_experiment(
    vision_tower,
    fastv_model,
    fastv_tokenizer,
    fastv_image_processor,
    image: Image.Image,
    image_name: str = "image"
) -> Dict:
    """
    Run experiment on a single image.

    Returns:
        Dictionary with results including indices and Jaccard similarity
    """
    print(f"\n{'-' * 80}")
    print(f"Processing: {image_name}")
    print(f"{'-' * 80}")

    results = {"image_name": image_name}

    # Run PruMerge
    try:
        prumerge_indices = run_prumerge_inference(vision_tower, image)
        results["prumerge_indices"] = prumerge_indices
        results["prumerge_success"] = True
    except Exception as e:
        print(f"  ✗ PruMerge failed: {e}")
        results["prumerge_success"] = False
        prumerge_indices = []
        results["prumerge_indices"] = []

    # Run FastV (if available)
    if fastv_model is not None:
        try:
            fastv_indices = run_fastv_inference(
                fastv_model, fastv_tokenizer, fastv_image_processor, image
            )
            results["fastv_indices"] = fastv_indices
            results["fastv_success"] = True
        except Exception as e:
            print(f"  ✗ FastV failed: {e}")
            results["fastv_success"] = False
            fastv_indices = []
            results["fastv_indices"] = []
    else:
        results["fastv_success"] = False
        fastv_indices = []
        results["fastv_indices"] = []

    # Compute Jaccard similarity
    if results["prumerge_success"] and results["fastv_success"]:
        stats = compute_jaccard_stats(fastv_indices, prumerge_indices)
        results.update(stats)

        print(f"\n  Results:")
        print(f"    FastV tokens:      {stats['set_a_size']}")
        print(f"    PruMerge tokens:   {stats['set_b_size']}")
        print(f"    Intersection:      {stats['intersection_size']}")
        print(f"    Union:             {stats['union_size']}")
        print(f"    Jaccard similarity: {stats['jaccard']:.4f}")
    else:
        results["jaccard"] = None
        print(f"\n  ⚠ Skipping Jaccard computation (one or both methods failed)")

    return results


def run_multi_image_experiment(
    vision_tower,
    fastv_model,
    fastv_tokenizer,
    fastv_image_processor,
    image_sources: List[Tuple[str, str]]
) -> List[Dict]:
    """
    Run experiment on multiple images.

    Args:
        vision_tower: PruMerge vision tower
        fastv_model: FastV model (can be None)
        fastv_tokenizer: FastV tokenizer (can be None)
        fastv_image_processor: FastV image processor (can be None)
        image_sources: List of (name, url_or_path) tuples

    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 80)
    print("Running Multi-Image Experiment")
    print("=" * 80)

    all_results = []

    for i, (name, source) in enumerate(image_sources, 1):
        print(f"\n[{i}/{len(image_sources)}]", end=" ")

        # Load image
        try:
            if source.startswith("http"):
                image = download_image(source)
            else:
                image = load_local_image(source)
        except Exception as e:
            print(f"Failed to load image {name}: {e}")
            continue

        # Run experiment
        results = run_single_image_experiment(
            vision_tower,
            fastv_model,
            fastv_tokenizer,
            fastv_image_processor,
            image,
            name
        )

        all_results.append(results)

    return all_results


def summarize_results(results: List[Dict]):
    """Print summary statistics across all images"""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    valid_results = [r for r in results if r.get("jaccard") is not None]

    if len(valid_results) == 0:
        print("No valid results to summarize.")
        return

    jaccards = [r["jaccard"] for r in valid_results]
    fastv_sizes = [r["set_a_size"] for r in valid_results]
    prumerge_sizes = [r["set_b_size"] for r in valid_results]
    intersections = [r["intersection_size"] for r in valid_results]

    print(f"\nProcessed {len(valid_results)} images successfully")
    print(f"\nJaccard Similarity:")
    print(f"  Mean:   {np.mean(jaccards):.4f}")
    print(f"  Median: {np.median(jaccards):.4f}")
    print(f"  Std:    {np.std(jaccards):.4f}")
    print(f"  Min:    {np.min(jaccards):.4f}")
    print(f"  Max:    {np.max(jaccards):.4f}")

    print(f"\nToken Counts:")
    print(f"  FastV (mean):      {np.mean(fastv_sizes):.1f} ± {np.std(fastv_sizes):.1f}")
    print(f"  PruMerge (mean):   {np.mean(prumerge_sizes):.1f} ± {np.std(prumerge_sizes):.1f}")
    print(f"  Intersection (mean): {np.mean(intersections):.1f} ± {np.std(intersections):.1f}")

    # Per-image breakdown
    print(f"\nPer-Image Results:")
    print(f"  {'Image':<30} {'FastV':>8} {'PruMerge':>10} {'Overlap':>8} {'Jaccard':>8}")
    print(f"  {'-' * 70}")
    for r in valid_results:
        print(f"  {r['image_name']:<30} {r['set_a_size']:>8} {r['set_b_size']:>10} "
              f"{r['intersection_size']:>8} {r['jaccard']:>8.4f}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Main experiment entry point"""

    # Sample images for testing
    sample_images = [
        ("COCO_sample_1", "http://images.cocodataset.org/val2017/000000039769.jpg"),
        ("COCO_sample_2", "http://images.cocodataset.org/val2017/000000397133.jpg"),
        ("COCO_sample_3", "http://images.cocodataset.org/val2017/000000252219.jpg"),
        ("COCO_sample_4", "http://images.cocodataset.org/val2017/000000087038.jpg"),
        ("COCO_sample_5", "http://images.cocodataset.org/val2017/000000174482.jpg"),
    ]

    # Load PruMerge model
    vision_tower = load_prumerge_model()

    # Load FastV model (optional - may fail without full weights)
    print("\nNote: FastV loading requires full model weights and may fail in this demo.")
    print("The experiment will continue with PruMerge only if FastV fails.")

    fastv_tokenizer = None
    fastv_model = None
    fastv_image_processor = None
    fastv_context_len = None

    # Uncomment to try loading FastV:
    # fastv_tokenizer, fastv_model, fastv_image_processor, fastv_context_len = load_fastv_model()

    # Run experiment
    results = run_multi_image_experiment(
        vision_tower,
        fastv_model,
        fastv_tokenizer,
        fastv_image_processor,
        sample_images
    )

    # Summarize
    summarize_results(results)

    # Visualize first result (if both methods succeeded)
    if len(results) > 0 and results[0].get("jaccard") is not None:
        print("\nGenerating visualization for first image...")
        visualize_token_overlap(
            results[0]["fastv_indices"],
            results[0]["prumerge_indices"],
            title=f"Token Selection: {results[0]['image_name']}",
            save_path="token_overlap_visualization.png"
        )
        plt.show()

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
