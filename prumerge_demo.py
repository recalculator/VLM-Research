"""
Simple Demo: LLaVA-PruMerge Token Selection Visualization

This script demonstrates token pruning with PruMerge only.
It's lightweight and can run without the full LLaVA model.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Setup paths
WORK_DIR = Path(__file__).parent
PRUMERGE_PATH = WORK_DIR / "LLaVA-PruMerge"
sys.path.insert(0, str(PRUMERGE_PATH))

print("=" * 80)
print("PruMerge Token Selection Demo")
print("=" * 80)
print()


def download_image(url):
    """Download image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.convert("RGB")


def visualize_tokens(indices, title="Token Selection", grid_size=(24, 24)):
    """Visualize selected tokens on a grid"""
    h, w = grid_size
    grid = np.zeros((h, w))

    for idx in indices:
        row, col = divmod(int(idx), w)
        if row < h and col < w:
            grid[row, col] = 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'{title}\n{len(indices)} tokens selected', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Add grid lines
    for i in range(h):
        ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.3)
    for j in range(w):
        ax.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.3)

    plt.colorbar(im, ax=ax, label='Selected')
    plt.tight_layout()

    return fig


def main():
    # Import after path setup
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower, kept_token_indices

    print("Loading PruMerge vision encoder...")
    print()

    # Load vision tower
    vision_tower = CLIPVisionTower(
        vision_tower="openai/clip-vit-large-patch14-336",
        args=None,
        delay_load=False
    )

    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Device: {vision_tower.device}")
    print(f"  Dtype: {vision_tower.dtype}")
    print()

    # Test images
    test_images = [
        ("Two cats on a couch", "http://images.cocodataset.org/val2017/000000039769.jpg"),
        ("Tennis player", "http://images.cocodataset.org/val2017/000000397133.jpg"),
        ("Pizza", "http://images.cocodataset.org/val2017/000000252219.jpg"),
    ]

    all_results = []

    for i, (description, url) in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] Processing: {description}")
        print("-" * 80)

        # Download image
        try:
            image = download_image(url)
            print(f"  ✓ Image loaded: {image.size}")
        except Exception as e:
            print(f"  ✗ Failed to load image: {e}")
            continue

        # Display image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(description)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"demo_image_{i}.png", dpi=100, bbox_inches='tight')
        print(f"  ✓ Saved image to demo_image_{i}.png")

        # Prepare image tensor
        image_tensor = vision_tower.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(vision_tower.device, dtype=vision_tower.dtype)

        # Run PruMerge inference
        with torch.no_grad():
            _ = vision_tower.token_prune_merge_advanced_plus(
                image_tensor,
                if_adaptive=True,
                reduction_ratio=1/8
            )

        # Extract indices
        indices = kept_token_indices["indices"]
        reduction_ratio = kept_token_indices["reduction_ratio"]

        if indices is None:
            print("  ✗ Failed to capture indices")
            continue

        # Convert to list
        if len(indices.shape) > 1:
            indices = indices[0]
        indices_list = indices.numpy().tolist()

        print(f"  ✓ PruMerge selected {len(indices_list)} tokens")
        print(f"    Reduction ratio: {reduction_ratio:.3f} ({reduction_ratio * 100:.1f}%)")
        print(f"    Tokens per row (avg): {len(indices_list) / 24:.1f}")

        # Analyze spatial distribution
        rows = [idx // 24 for idx in indices_list]
        cols = [idx % 24 for idx in indices_list]

        print(f"    Vertical spread: rows {min(rows)}-{max(rows)} (range: {max(rows) - min(rows) + 1}/24)")
        print(f"    Horizontal spread: cols {min(cols)}-{max(cols)} (range: {max(cols) - min(cols) + 1}/24)")

        # Visualize
        fig = visualize_tokens(indices_list, f"{description}\n(PruMerge)")
        plt.savefig(f"demo_tokens_{i}.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to demo_tokens_{i}.png")
        plt.close(fig)

        all_results.append({
            "description": description,
            "num_tokens": len(indices_list),
            "reduction_ratio": reduction_ratio,
            "indices": indices_list
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nProcessed {len(all_results)} images successfully\n")
    print(f"{'Image':<30} {'Tokens':>10} {'Ratio':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['description']:<30} {r['num_tokens']:>10} {r['reduction_ratio']:>10.3f}")

    if len(all_results) > 0:
        avg_tokens = np.mean([r['num_tokens'] for r in all_results])
        avg_ratio = np.mean([r['reduction_ratio'] for r in all_results])
        print("-" * 80)
        print(f"{'Average':<30} {avg_tokens:>10.1f} {avg_ratio:>10.3f}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - demo_image_*.png: Original images")
    print("  - demo_tokens_*.png: Token selection visualizations")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
