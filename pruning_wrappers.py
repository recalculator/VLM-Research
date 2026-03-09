"""
Wrapper functions to extract visual token indices from FastV and LLaVA-PruMerge.

This module provides clean interfaces to run both pruning methods and extract
the indices of kept visual tokens for Jaccard similarity analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Add repositories to path
FASTV_PATH = Path(__file__).parent / "FastV"
PRUMERGE_PATH = Path(__file__).parent / "LLaVA-PruMerge"

# Global storage for captured indices
_captured_indices = {"fastv": None, "prumerge": None}


# ============================================================================
# LLaVA-PruMerge Index Extraction
# ============================================================================

def complement_idx(idx, dim):
    """Helper function from PruMerge to compute complement indices."""
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


def outlier_detection(attn):
    """Adaptive reduction ratio based on attention outlier detection."""
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()
    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = np.where((attn_np > upper_bound))[0]
    ratio = len(outlier_indices) / len(attn_np)
    return ratio


def extract_prumerge_indices(
    vision_tower,
    images,
    reduction_ratio: float = 1/8,
    if_adaptive: bool = True,
    return_features: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract visual token indices kept by LLaVA-PruMerge.

    This is a modified version of token_prune_merge_advanced_plus that returns
    the indices of kept tokens.

    Args:
        vision_tower: CLIPVisionTower instance
        images: Input images tensor
        reduction_ratio: Fraction of tokens to keep (default 1/8)
        if_adaptive: Use adaptive reduction ratio (default True)
        return_features: Also return the pruned features (default False)

    Returns:
        idx: Tensor of shape [B, num_kept] with indices of kept tokens (0-575)
        features: (optional) Pruned image features if return_features=True
    """
    device = vision_tower.device
    dtype = vision_tower.dtype

    # Global outputs dict for hooks
    outputs = {}

    def hook_k(module, input, output):
        outputs['desired_k'] = output

    def hook_q(module, input, output):
        outputs['desired_q'] = output

    # Set hooks for extracting layer 23's k and q
    hook_handle_k = vision_tower.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
    hook_handle_q = vision_tower.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

    # Forward pass
    image_forward_outs = vision_tower.vision_tower(
        images.to(device=device, dtype=dtype),
        output_hidden_states=True
    )

    # Select features (patch tokens only, no CLS)
    image_features = image_forward_outs.hidden_states[vision_tower.select_layer][:, 1:].to(images.dtype)
    B, N, C = image_features.shape  # N should be 576

    # Extract k and q, remove hooks
    desired_layer_k = outputs["desired_k"]
    desired_layer_q = outputs["desired_q"]
    hook_handle_k.remove()
    hook_handle_q.remove()

    # Compute attention
    attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
    attn = F.softmax(attn, dim=-1)

    # CLS attention to patches (excluding CLS itself)
    cls_attn = attn[:, 0, 1:]  # Shape: [B, N]

    # Adaptive reduction ratio
    if if_adaptive:
        reduction_ratio = outlier_detection(cls_attn)

    # Select top tokens based on CLS attention
    _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)

    # Store initial indices before adaptive augmentation
    initial_idx = idx.clone()

    # Adaptive mode: add spatially sampled tokens
    if if_adaptive:
        step_length = int(1 / reduction_ratio)
        arithmetic_sequence = torch.arange(0, 575, int(step_length / 3)).to(device=device)
        original_tensor_1d = idx.flatten().to(device=device)
        filtered_sequence = torch.tensor(
            [x for x in arithmetic_sequence if x not in original_tensor_1d]
        ).to(device=device)
        concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
        idx = concatenated_tensor

    # Store captured indices
    _captured_indices["prumerge"] = idx.cpu().numpy()

    if return_features:
        # Continue with the merging process (simplified version)
        # For full implementation, include the token merging logic
        index = idx.unsqueeze(-1).expand(-1, -1, C)
        x_others = torch.gather(image_features, dim=1, index=index)
        return idx, x_others
    else:
        return idx, None


# ============================================================================
# FastV Index Extraction
# ============================================================================

def extract_fastv_indices_from_model(
    model,
    image_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sys_length: int = 35,  # Default system prompt length
    image_token_length: int = 576,
    attention_rank: int = 144,  # Default ATTENTION_RANK
    agg_layer: int = 15,  # Default aggregation layer
) -> torch.Tensor:
    """
    Extract visual token indices kept by FastV during inference.

    This requires running the model and capturing the indices computed
    during the forward pass at the aggregation layer.

    Args:
        model: LLaMA model with FastV modifications
        image_embeds: Input embeddings including system + image tokens
        attention_mask: Attention mask
        sys_length: Number of system tokens before image tokens
        image_token_length: Number of image tokens (576 for 24x24)
        attention_rank: Number of image tokens to keep
        agg_layer: Layer at which aggregation occurs

    Returns:
        indices: Tensor of shape [num_kept] with indices in range [0, 575]
    """
    # This is a placeholder - actual implementation requires
    # modifying the FastV model code to expose indices
    # See modification instructions below
    raise NotImplementedError(
        "FastV index extraction requires model code modification. "
        "See extract_fastv_indices_with_hooks() for hook-based approach."
    )


def extract_fastv_indices_with_hooks(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    sys_length: int = 35,
    image_token_length: int = 576,
) -> List[int]:
    """
    Extract FastV indices using forward hooks.

    This approach instruments the LLaMA model to capture attention scores
    and compute which tokens would be kept.

    Args:
        model: LLaVA model with FastV
        input_ids: Input token IDs
        attention_mask: Attention mask
        pixel_values: Image tensor
        sys_length: System token length
        image_token_length: Number of image tokens

    Returns:
        List of kept token indices in range [0, 575]
    """
    kept_indices = []

    def attention_hook(module, input, output):
        """Hook to capture attention at aggregation layer."""
        # output is (attn_output, attn_weights, past_key_value)
        if len(output) > 1 and output[1] is not None:
            attn_weights = output[1]  # Shape: [B, num_heads, seq_len, seq_len]

            # Average over heads
            attn_avg = torch.mean(attn_weights, dim=1)[0]  # [seq_len, seq_len]

            # Get attention from last token
            last_tok_attn = attn_avg[-1]  # [seq_len]

            # Extract attention to image tokens
            image_attn = last_tok_attn[sys_length:sys_length + image_token_length]

            # Get top-k indices
            attention_rank = model.config.fast_v_attention_rank
            top_indices = image_attn.topk(attention_rank).indices

            # Convert to 0-575 range
            kept_indices.extend(top_indices.cpu().tolist())

    # Register hook at aggregation layer
    agg_layer = model.config.fast_v_agg_layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        hook = model.model.layers[agg_layer].register_forward_hook(attention_hook)
    else:
        raise ValueError("Cannot find model layers to attach hook")

    # Run forward pass
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )

    # Remove hook
    hook.remove()

    return kept_indices


# ============================================================================
# Jaccard Similarity
# ============================================================================

def compute_jaccard_similarity(set_a: List[int], set_b: List[int]) -> float:
    """
    Compute Jaccard similarity between two sets of token indices.

    Jaccard similarity = |A ∩ B| / |A ∪ B|

    Args:
        set_a: List of token indices from method A
        set_b: List of token indices from method B

    Returns:
        Jaccard similarity score in range [0, 1]
    """
    set_a = set(set_a)
    set_b = set(set_b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def compute_jaccard_with_details(
    set_a: List[int],
    set_b: List[int]
) -> dict:
    """
    Compute Jaccard similarity with detailed statistics.

    Args:
        set_a: List of token indices from method A
        set_b: List of token indices from method B

    Returns:
        Dictionary with:
            - jaccard: Jaccard similarity score
            - intersection_size: Number of common tokens
            - union_size: Number of unique tokens
            - set_a_size: Number of tokens in set A
            - set_b_size: Number of tokens in set B
            - only_in_a: Tokens only in A
            - only_in_b: Tokens only in B
    """
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
        "only_in_a": sorted(list(only_a)),
        "only_in_b": sorted(list(only_b)),
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_token_selection(
    image_size: Tuple[int, int] = (24, 24),
    fastv_indices: Optional[List[int]] = None,
    prumerge_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize which tokens are kept by each method on a 24x24 grid.

    Args:
        image_size: Grid dimensions (default 24x24 for 576 tokens)
        fastv_indices: Indices kept by FastV
        prumerge_indices: Indices kept by PruMerge
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    h, w = image_size

    # Create grid representations
    fastv_grid = np.zeros((h, w))
    prumerge_grid = np.zeros((h, w))
    both_grid = np.zeros((h, w))

    if fastv_indices:
        for idx in fastv_indices:
            row, col = divmod(idx, w)
            fastv_grid[row, col] = 1

    if prumerge_indices:
        for idx in prumerge_indices:
            row, col = divmod(idx, w)
            prumerge_grid[row, col] = 1

    # Compute overlap
    # 0 = neither, 1 = only FastV, 2 = only PruMerge, 3 = both
    both_grid = fastv_grid + prumerge_grid * 2

    # Plot FastV
    axes[0].imshow(fastv_grid, cmap='Reds', interpolation='nearest')
    axes[0].set_title(f'FastV\n({len(fastv_indices) if fastv_indices else 0} tokens)')
    axes[0].axis('off')

    # Plot PruMerge
    axes[1].imshow(prumerge_grid, cmap='Blues', interpolation='nearest')
    axes[1].set_title(f'PruMerge\n({len(prumerge_indices) if prumerge_indices else 0} tokens)')
    axes[1].axis('off')

    # Plot overlap
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red', 'blue', 'purple']
    cmap = ListedColormap(colors)
    axes[2].imshow(both_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    axes[2].set_title('Overlap\n(Purple = both, Red = FastV only, Blue = PruMerge only)')
    axes[2].axis('off')

    # Add legend
    red_patch = mpatches.Patch(color='red', label='FastV only')
    blue_patch = mpatches.Patch(color='blue', label='PruMerge only')
    purple_patch = mpatches.Patch(color='purple', label='Both')
    axes[2].legend(handles=[red_patch, blue_patch, purple_patch],
                   loc='upper center', bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Example usage
    print("Pruning Wrappers Module")
    print("=" * 50)
    print("\nThis module provides functions to:")
    print("1. Extract token indices from FastV")
    print("2. Extract token indices from LLaVA-PruMerge")
    print("3. Compute Jaccard similarity")
    print("4. Visualize token selections")
    print("\nSee experiment_notebook.py for usage examples.")
