# Visual Token Pruning Analysis

## FastV Pruning Logic

**Location**: `FastV/src/transformers/src/transformers/models/llama/modeling_llama.py:746`

### Key Parameters:
- `SYS_LENGTH`: System token length (before image tokens)
- `IMAGE_TOKEN_LENGTH`: Number of image tokens (576 for 24×24 grid)
- `ATTENTION_RANK`: Number of tokens to keep after pruning
- `AGG_LAYER`: Layer at which pruning occurs

### Pruning Method:
```python
# Line 740: Compute average attention over different heads
last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]

# Line 742: Get attention from last token
last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]

# Line 744: Extract attention to image tokens
last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]

# Line 746: Select top ATTENTION_RANK tokens
top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
```

### Key Insight:
- FastV selects tokens based on average attention from the **last token** to image tokens
- Indices are in the range `[SYS_LENGTH, SYS_LENGTH+IMAGE_TOKEN_LENGTH)`
- To get 0-575 range: `kept_indices = top_attention_rank_index - SYS_LENGTH`

---

## LLaVA-PruMerge Pruning Logic

**Location**: `LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py:201-214`

### Key Parameters:
- `reduction_ratio`: Fraction of tokens to keep (default 1/8 = 12.5%)
- `if_adaptive`: Whether to use adaptive mode (adds spatially sampled tokens)

### Pruning Method:
```python
# Line 198-199: Compute attention matrix
attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
attn = F.softmax(attn, dim=-1)

# Line 201: Get CLS token attention to patches (excludes CLS itself)
cls_attn = attn[:, 0, 1:]

# Line 204: Adaptive reduction ratio based on outlier detection
if if_adaptive:
    reduction_ratio = outlier_dectection(cls_attn)

# Line 205: Select top tokens
_, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)

# Line 208-214: If adaptive, add spatially sampled tokens
if if_adaptive:
    step_length = int(1/reduction_ratio)
    arithmetic_sequence = torch.arange(0, 575, int(step_length/3))
    # Filter out already selected indices and concatenate
    filtered_sequence = [x for x in arithmetic_sequence if x not in idx]
    idx = torch.cat((idx, filtered_sequence), dim=1)
```

### Key Insight:
- PruMerge selects tokens based on **CLS token attention** from layer 23
- Indices are already in 0-575 range (N=576 patches, cls_attn excludes CLS)
- In adaptive mode, additional spatially distributed tokens are added
- The final `idx` tensor contains all kept token indices

---

## Index Space Comparison

Both methods use **576 visual tokens** (24×24 patch grid from CLIP ViT-L/14@336px).

- **FastV**: Indices in `top_attention_rank_index - SYS_LENGTH` → range [0, 575]
- **PruMerge**: Indices in `idx` → range [0, 575]

Both are directly comparable for Jaccard similarity computation.

---

## Modification Strategy

### FastV Modifications:
1. Store `top_attention_rank_index - SYS_LENGTH` in a global variable or return it
2. Create a wrapper function that runs inference and returns kept indices

### PruMerge Modifications:
1. Return `idx` from `token_prune_merge_advanced_plus()`
2. Store indices before the merging operations
3. Create a wrapper function that runs inference and returns kept indices

---

## Jaccard Similarity Formula

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Where:
- A = set of indices kept by FastV
- B = set of indices kept by PruMerge
```

Example:
- FastV keeps 144 tokens: [0, 5, 10, 15, ...]
- PruMerge keeps 150 tokens: [1, 5, 11, 15, ...]
- Intersection: 73 tokens
- Union: 221 tokens
- Jaccard: 73/221 = 0.33
