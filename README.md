# Visual Token Pruning Comparison Experiment

This repository contains code to compare visual token pruning strategies between **FastV** and **LLaVA-PruMerge** by computing **Jaccard similarity** on the selected token sets.

## Overview

Vision-language models process images as sequences of visual tokens (typically 576 tokens for a 24×24 patch grid from CLIP ViT-L/14@336). Token pruning methods reduce computational cost by keeping only the most important tokens.

This experiment compares two pruning methods:

1. **FastV**: Selects tokens based on average attention from the last generated token
2. **LLaVA-PruMerge**: Selects tokens based on CLS token attention from CLIP layer 23

We measure how much the two methods agree by computing **Jaccard similarity**:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Where A and B are the sets of token indices kept by each method.

---

## Repository Structure

```
Research/
├── FastV/                          # FastV repository (modified)
│   └── src/transformers/src/transformers/models/llama/
│       └── modeling_llama.py       # Modified to expose token indices
├── LLaVA-PruMerge/                 # PruMerge repository (modified)
│   └── llava/model/multimodal_encoder/
│       └── clip_encoder.py         # Modified to expose token indices
├── PRUNING_ANALYSIS.md             # Detailed analysis of pruning logic
├── pruning_wrappers.py             # Utility functions for index extraction
├── jaccard_experiment.py           # Main experiment script
├── jaccard_experiment_notebook.ipynb # Jupyter notebook version
├── prumerge_demo.py                # Simple demo (PruMerge only)
└── README.md                       # This file
```

---

## Code Modifications

### LLaVA-PruMerge

**File**: `LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py`

**Changes**:
1. Added global variable at line 29:
   ```python
   kept_token_indices = {"indices": None, "reduction_ratio": None}
   ```

2. Modified `token_prune_merge_advanced_plus()` at lines 210-213:
   ```python
   # Store the kept indices globally for analysis (before adaptive augmentation)
   global kept_token_indices
   kept_token_indices["indices"] = idx.detach().cpu().clone()
   kept_token_indices["reduction_ratio"] = reduction_ratio
   ```

**Purpose**: Captures the indices of tokens selected by PruMerge before merging operations.

**Location in pruning flow**:
- Line 201: `cls_attn = attn[:, 0, 1:]` — Get CLS attention to patches
- Line 205: `_, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)` — Select top tokens
- Line 210-213: **Store indices** ← Our modification
- Lines 216-227: Adaptive augmentation (adds spatial tokens)

### FastV

**File**: `FastV/src/transformers/src/transformers/models/llama/modeling_llama.py`

**Changes**:
1. Added global variable at line 43:
   ```python
   kept_visual_token_indices = {"indices": None, "sys_length": None, "image_token_length": None}
   ```

2. Modified pruning logic at lines 751-755 (and 795-799 for alternative code path):
   ```python
   # Store the kept visual token indices globally for analysis (0-575 range)
   global kept_visual_token_indices
   kept_visual_token_indices["indices"] = (top_attention_rank_index - SYS_LENGTH).detach().cpu().clone()
   kept_visual_token_indices["sys_length"] = SYS_LENGTH
   kept_visual_token_indices["image_token_length"] = IMAGE_TOKEN_LENGTH
   ```

**Purpose**: Captures the indices of tokens selected by FastV during inference.

**Location in pruning flow**:
- Line 743: `last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]` — Average attention over heads
- Line 745: `last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]` — Get last token's attention
- Line 747: Extract attention to image tokens
- Line 749: `top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH` — Select top tokens
- Lines 751-755: **Store indices** ← Our modification

**Important**: The stored indices are converted to 0-575 range by subtracting `SYS_LENGTH` to match PruMerge's index space.

---

## Pruning Logic Details

### PruMerge Algorithm

1. **Extract attention**: Compute attention matrix from CLIP layer 23 using key and query projections
2. **CLS attention**: Extract CLS token's attention to all patch tokens (`cls_attn = attn[:, 0, 1:]`)
3. **Adaptive ratio**: Use outlier detection to determine how many tokens to keep
4. **Select tokens**: `torch.topk(cls_attn, int(N*reduction_ratio), largest=True)`
5. **Spatial augmentation**: (if adaptive) Add uniformly distributed tokens to ensure coverage
6. **Merge**: Merge pruned tokens with their neighbors using weighted averaging

**Key parameters**:
- `reduction_ratio`: Default 1/8 (12.5% of tokens), but adaptive mode adjusts this
- `if_adaptive`: Whether to use outlier detection and spatial sampling

**Indices returned**: 0-575 (direct patch indices)

### FastV Algorithm

1. **Layer selection**: Wait until aggregation layer (default layer 15)
2. **Extract attention**: Get attention weights from that layer
3. **Average heads**: Compute mean attention across all attention heads
4. **Last token attention**: Extract attention from the last generated token
5. **Image token attention**: Focus only on attention to image tokens (indices `SYS_LENGTH` to `SYS_LENGTH+IMAGE_TOKEN_LENGTH`)
6. **Select tokens**: `topk(ATTENTION_RANK)` on image token attention
7. **Filter**: Remove low-attention image tokens from subsequent layers

**Key parameters**:
- `ATTENTION_RANK`: Number of tokens to keep (default 144 = 25% of 576)
- `AGG_LAYER`: Layer at which to perform aggregation (default 15)
- `SYS_LENGTH`: Number of tokens before image tokens (typically 35-40)
- `IMAGE_TOKEN_LENGTH`: Number of image tokens (576 for 24×24 grid)

**Indices returned**: 0-575 (after subtracting `SYS_LENGTH`)

---

## Usage

### Option 1: Simple Demo (PruMerge only)

```bash
python prumerge_demo.py
```

This runs a simple demo that:
- Loads only the PruMerge model (lightweight, ~1.7GB)
- Processes 5 sample images from COCO
- Shows which tokens are selected
- Visualizes token selection on 24×24 grid

### Option 2: Full Experiment (Both Methods)

```bash
python jaccard_experiment.py
```

This runs the full experiment:
- Loads both FastV and PruMerge models
- Processes multiple images
- Computes Jaccard similarity for each image
- Generates summary statistics
- Creates visualizations

**Requirements**:
- Full LLaVA-v1.5-7b model weights (~13GB)
- GPU with at least 16GB VRAM recommended

### Option 3: Jupyter Notebook

```bash
jupyter notebook jaccard_experiment_notebook.ipynb
```

Or upload to Google Colab:
1. Upload `jaccard_experiment_notebook.ipynb` to Colab
2. Uncomment the installation and clone cells
3. Run all cells

---

## Expected Results

### Token Counts

- **FastV**: Typically keeps 144 tokens (25% of 576)
- **PruMerge**: Adaptive mode keeps 70-100 tokens (12-17% of 576)

### Jaccard Similarity

Expected range: **0.2 - 0.5**

- **High Jaccard (>0.4)**: Both methods agree on which tokens are important
- **Medium Jaccard (0.3-0.4)**: Moderate agreement, some overlap
- **Low Jaccard (<0.3)**: Methods have different notions of token importance

### Interpretation

- **High overlap**: Both attention-based methods identify similar salient regions
- **Low overlap**: Last-token attention (FastV) vs CLS attention (PruMerge) capture different aspects
- **Variation across images**: Simple images may have higher agreement, complex images may have lower agreement

---

## Extending the Experiment

### Add More Images

Edit `jaccard_experiment.py` and add to `sample_images`:

```python
sample_images = [
    ("my_image", "path/to/image.jpg"),
    # or
    ("web_image", "https://example.com/image.jpg"),
]
```

### Adjust Pruning Parameters

**PruMerge**:
```python
prumerge_indices = run_prumerge_inference(
    vision_tower, image,
    reduction_ratio=1/8,     # Change ratio
    if_adaptive=True         # Toggle adaptive mode
)
```

**FastV**:
```python
model.config.fast_v_attention_rank = 100  # Change number of tokens to keep
model.config.fast_v_agg_layer = 10        # Change aggregation layer
```

### Visualize Individual Images

```python
from pruning_wrappers import visualize_token_selection

visualize_token_selection(
    image_size=(24, 24),
    fastv_indices=fastv_indices,
    prumerge_indices=prumerge_indices,
    save_path="visualization.png"
)
```

---

## Technical Details

### Index Space

Both methods work with **576 visual tokens** corresponding to a 24×24 patch grid from CLIP ViT-L/14@336.

Indices are in range **[0, 575]**:
- Index 0 = top-left patch
- Index 23 = top-right patch
- Index 552 = bottom-left patch
- Index 575 = bottom-right patch

**Spatial layout**:
```
Index = row * 24 + col

Example:
row=0, col=0  → index=0
row=0, col=23 → index=23
row=12, col=12 → index=300 (center)
row=23, col=23 → index=575
```

### Attention Mechanisms

**CLIP CLS attention** (PruMerge):
- CLS token learns to attend to semantically important regions
- Computed in vision encoder before LLM
- Based on visual features only

**Last token attention** (FastV):
- Generated token attends to important visual tokens for next prediction
- Computed in LLM decoder during generation
- Influenced by text prompt and generation context

These different attention sources explain why Jaccard similarity is not 1.0.

---

## Troubleshooting

### ImportError: No module named 'llava'

Make sure LLaVA-PruMerge is in your path:
```python
sys.path.insert(0, str(Path("/path/to/LLaVA-PruMerge")))
```

### CUDA out of memory

- Reduce batch size
- Use smaller model (e.g., LLaVA-v1.5-7b instead of 13b)
- Run PruMerge-only demo instead of full experiment
- Use CPU inference (slower): `model.to("cpu")`

### Indices are None

Ensure you've applied the code modifications:
- Check that global variables are defined
- Check that indices are stored during forward pass
- Verify model is actually running inference (not just loading)

---

## Citation

If you use this code, please cite the original papers:

**FastV**:
```bibtex
@article{chen2024fastv,
  title={An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models},
  author={Chen, Liang and others},
  journal={arXiv preprint arXiv:2403.06764},
  year={2024}
}
```

**LLaVA-PruMerge**:
```bibtex
@article{shi2024pruning,
  title={Pruning and Merging Tokens for Large Vision Language Models},
  author={Shi, Shawn and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

This code is provided for research purposes. Please refer to the original repositories for their respective licenses:
- FastV: https://github.com/pkunlp-icler/FastV
- LLaVA-PruMerge: https://github.com/42Shawn/LLaVA-PruMerge

---

## Contact

For questions about this experiment, please open an issue in this repository.
