# Repository Modifications Required

This document describes the exact modifications needed for the FastV and LLaVA-PruMerge repositories to enable token index extraction.

## Prerequisites

Clone both repositories:

```bash
git clone https://github.com/pkunlp-icler/FastV.git
git clone https://github.com/42Shawn/LLaVA-PruMerge.git
```

---

## LLaVA-PruMerge Modifications

### File: `llava/model/multimodal_encoder/clip_encoder.py`

**Modification 1: Add global variable (after line 27)**

**Original:**
```python
outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output
```

**Modified:**
```python
outputs = {}
# Global variable to store kept token indices for analysis
kept_token_indices = {"indices": None, "reduction_ratio": None}

def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output
```

**Modification 2: Store indices in `token_prune_merge_advanced_plus()` method (after line 208)**

**Original (around line 205-209):**
```python
        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True

        # # # print("idx: ", idx)
        if if_adaptive:
```

**Modified:**
```python
        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True

        # Store the kept indices globally for analysis (before adaptive augmentation)
        global kept_token_indices
        kept_token_indices["indices"] = idx.detach().cpu().clone()
        kept_token_indices["reduction_ratio"] = reduction_ratio

        # # # print("idx: ", idx)
        if if_adaptive:
```

---

## FastV Modifications

### File: `src/transformers/src/transformers/models/llama/modeling_llama.py`

**Modification 1: Add global variable (after line 40)**

**Original:**
```python
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
```

**Modified:**
```python
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

# Global variable to store kept visual token indices for analysis
kept_visual_token_indices = {"indices": None, "sys_length": None, "image_token_length": None}


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
```

**Modification 2: Store indices in pruning logic - Code Path 1 (around line 747-751)**

**Original:**
```python
                        # get the attention in image token
                        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                        # get the indexs of the top ATTENTION_RANK tokens
                        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
                        # keep index
                        keep_indexs = torch.cat( (torch.arange(SYS_LENGTH,device=device), top_attention_rank_index, torch.arange(SYS_LENGTH+IMAGE_TOKEN_LENGTH,seq_length_with_past,device=device)))
```

**Modified:**
```python
                        # get the attention in image token
                        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                        # get the indexs of the top ATTENTION_RANK tokens
                        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH

                        # Store the kept visual token indices globally for analysis (0-575 range)
                        global kept_visual_token_indices
                        kept_visual_token_indices["indices"] = (top_attention_rank_index - SYS_LENGTH).detach().cpu().clone()
                        kept_visual_token_indices["sys_length"] = SYS_LENGTH
                        kept_visual_token_indices["image_token_length"] = IMAGE_TOKEN_LENGTH

                        # keep index
                        keep_indexs = torch.cat( (torch.arange(SYS_LENGTH,device=device), top_attention_rank_index, torch.arange(SYS_LENGTH+IMAGE_TOKEN_LENGTH,seq_length_with_past,device=device)))
```

**Modification 3: Store indices in pruning logic - Code Path 2 (around line 791-794)**

**Original:**
```python
                            # get the attention in image token
                            last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                            # get the indexs of the top ATTENTION_RANK tokens
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
                            # generate new attention mask
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
```

**Modified:**
```python
                            # get the attention in image token
                            last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                            # get the indexs of the top ATTENTION_RANK tokens
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH

                            # Store the kept visual token indices globally for analysis (0-575 range)
                            global kept_visual_token_indices
                            kept_visual_token_indices["indices"] = (top_attention_rank_index - SYS_LENGTH).detach().cpu().clone()
                            kept_visual_token_indices["sys_length"] = SYS_LENGTH
                            kept_visual_token_indices["image_token_length"] = IMAGE_TOKEN_LENGTH

                            # generate new attention mask
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
```

---

## Applying Modifications

### Option 1: Manual Editing

1. Open the files in your editor
2. Locate the lines mentioned above
3. Add the modifications exactly as shown

### Option 2: Using Patch Files

Create patch files and apply with `git apply`:

**For PruMerge (`prumerge.patch`):**
```diff
diff --git a/llava/model/multimodal_encoder/clip_encoder.py b/llava/model/multimodal_encoder/clip_encoder.py
index xxx..xxx 100644
--- a/llava/model/multimodal_encoder/clip_encoder.py
+++ b/llava/model/multimodal_encoder/clip_encoder.py
@@ -27,6 +27,8 @@

 outputs = {}
+# Global variable to store kept token indices for analysis
+kept_token_indices = {"indices": None, "reduction_ratio": None}
+
 def hook_k(module, input, output):
     outputs['desired_k'] = output

@@ -205,6 +207,11 @@
         if if_adaptive:
             reduction_ratio = outlier_dectection(cls_attn)
         _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)
+
+        # Store the kept indices globally for analysis
+        global kept_token_indices
+        kept_token_indices["indices"] = idx.detach().cpu().clone()
+        kept_token_indices["reduction_ratio"] = reduction_ratio

         if if_adaptive:
```

Apply: `git apply prumerge.patch`

---

## Verification

After applying modifications, verify they work:

```python
# Test PruMerge
from llava.model.multimodal_encoder.clip_encoder import kept_token_indices
print(f"Global variable exists: {kept_token_indices}")

# Test FastV
from transformers.models.llama.modeling_llama import kept_visual_token_indices
print(f"Global variable exists: {kept_visual_token_indices}")
```

---

## Summary

**Total lines modified:**
- **PruMerge**: 2 additions (6 lines total)
- **FastV**: 3 additions (12 lines total)

All modifications are non-invasive and simply expose internal indices without changing the pruning algorithms.
