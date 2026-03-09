# Quick Start Guide

Get the visual token pruning comparison experiment running in 5 minutes.

---

## Prerequisites

- Python 3.8+
- pip
- Internet connection (for downloading images and models)

---

## Setup (5 steps)

### Step 1: Clone the Repository

```bash
git clone https://github.com/recalculator/VLM-Research.git
cd VLM-Research
```

### Step 2: Clone External Repositories

```bash
# Clone FastV
git clone https://github.com/pkunlp-icler/FastV.git

# Clone LLaVA-PruMerge
git clone https://github.com/42Shawn/LLaVA-PruMerge.git
```

**Note**: These repositories are already modified in this workspace. If cloning fresh, see `MODIFICATIONS.md` for required changes.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch
- Transformers
- Pillow
- Matplotlib
- NumPy
- Requests

### Step 4: Verify Setup

```bash
python verify_setup.py
```

Expected output:
```
================================================================================
✓ ALL CHECKS PASSED - Ready to run experiments!
================================================================================
```

If any checks fail, see the error messages for guidance.

### Step 5: Run the Experiment

```bash
python run_experiment.py --mode demo --images 3
```

---

## Understanding the Output

### Console Output

```
================================================================================
VISUAL TOKEN PRUNING COMPARISON EXPERIMENT
================================================================================
Mode: demo
Images: 3
Visualizations: enabled

Loading PruMerge model...
  ✓ PruMerge loaded

================================================================================
RUNNING EXPERIMENT
================================================================================

[1/3] Processing: cats
--------------------------------------------------------------------------------
  ✓ Image loaded: (640, 480)
  ✓ PruMerge: kept 106 tokens

[2/3] Processing: tennis
[3/3] Processing: pizza

================================================================================
EXPERIMENT SUMMARY
================================================================================
Image                              Tokens
--------------------------------------------------------------------------------
cats                                  106
tennis                                115
pizza                                 112
```

### Output Files

**`outputs/results.json`**: Complete results in JSON format
```json
[
  {
    "name": "cats",
    "prumerge_indices": [499, 466, 495, ...],
    "prumerge_success": true,
    ...
  }
]
```

**`outputs/visualizations/`**: Token selection visualizations (when both methods run)

---

## CLI Options

### Basic Usage

```bash
python run_experiment.py [OPTIONS]
```

### Options

| Option          | Values           | Default | Description                          |
|-----------------|------------------|---------|--------------------------------------|
| `--mode`        | `full`, `demo`   | `full`  | Experiment mode                      |
| `--images`      | `1-5`            | `3`     | Number of images to process          |
| `--no-viz`      | flag             | off     | Skip visualization generation        |

### Examples

```bash
# Run demo mode (PruMerge only) with 1 image
python run_experiment.py --mode demo --images 1

# Run demo mode with 5 images
python run_experiment.py --mode demo --images 5

# Run without visualizations (faster)
python run_experiment.py --mode demo --images 3 --no-viz

# Run full comparison (requires LLaVA weights)
python run_experiment.py --mode full --images 3
```

---

## Demo vs Full Mode

### Demo Mode (Default Recommendation)

**What it does**:
- Runs PruMerge only
- Lightweight (~2GB RAM)
- Works on CPU
- Fast execution (~10 sec/image)

**When to use**:
- Testing the pipeline
- Understanding PruMerge token selection
- No GPU available
- Quick experiments

**Command**:
```bash
python run_experiment.py --mode demo --images 3
```

### Full Mode (Requires Model Weights)

**What it does**:
- Runs both FastV and PruMerge
- Computes Jaccard similarity
- Generates comparison visualizations
- Heavy (~13GB model weights + 16GB GPU RAM)

**Requirements**:
- LLaVA-v1.5-7b model weights (~13GB download)
- GPU with 16GB+ VRAM recommended
- ~30-60 sec/image

**When to use**:
- Full comparison needed
- GPU available
- Final results for publication

**Command**:
```bash
python run_experiment.py --mode full --images 3
```

**First-time setup** (automatic):
- Script will download LLaVA weights from HuggingFace
- Takes 10-20 minutes depending on connection
- Saved locally for future runs

---

## Troubleshooting

### Import Error: No module named 'transformers'

**Solution**:
```bash
pip install -r requirements.txt
```

### Error: Repositories not found

**Solution**:
```bash
# Make sure you're in the VLM-Research directory
git clone https://github.com/pkunlp-icler/FastV.git
git clone https://github.com/42Shawn/LLaVA-PruMerge.git
```

### Error: Modifications not applied

**Solution**:
Check that the global variables exist in the code:

```bash
# Check PruMerge
grep "kept_token_indices" LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder.py

# Check FastV
grep "kept_visual_token_indices" FastV/src/transformers/src/transformers/models/llama/modeling_llama.py
```

If not found, see `MODIFICATIONS.md` for manual modification steps.

### Error: tokenizers version conflict

This is handled automatically by `direct_prumerge_loader.py`. If you still see this error:

```bash
# Temporarily remove FastV from path
mv FastV FastV.bak
python run_experiment.py --mode demo
mv FastV.bak FastV
```

### CUDA out of memory (Full mode)

**Solutions**:
1. Use `load_8bit=True` in model loading
2. Reduce number of images: `--images 1`
3. Run on CPU (slower): modify `device_map="cpu"`
4. Use demo mode instead

---

## Next Steps

### Analyze Results

```bash
# View results
cat outputs/results.json

# Pretty print
python -m json.tool outputs/results.json
```

### Add Custom Images

Edit `run_experiment.py` and add to `all_images`:

```python
all_images = [
    ("my_image", "path/to/image.jpg"),
    ("web_image", "https://example.com/image.jpg"),
    ...
]
```

### Extend the Analysis

See `EXPERIMENT_SUMMARY.md` for ideas on extending the experiment.

---

## Getting Help

1. **Verify setup**: `python verify_setup.py`
2. **Check documentation**: See `README.md` for full details
3. **Review modifications**: See `MODIFICATIONS.md`
4. **View results**: See `EXECUTION_RESULTS.md`
5. **Open issue**: [GitHub Issues](https://github.com/recalculator/VLM-Research/issues)

---

## Quick Reference

### One-Line Commands

```bash
# Verify everything is working
python verify_setup.py

# Run quick test (1 image)
python run_experiment.py --mode demo --images 1 --no-viz

# Run full demo (3 images with viz)
python run_experiment.py --mode demo --images 3

# Run full comparison (requires GPU + weights)
python run_experiment.py --mode full --images 3
```

### File Locations

| File                        | Purpose                              |
|-----------------------------|--------------------------------------|
| `run_experiment.py`         | Main experiment script               |
| `verify_setup.py`           | Validate environment                 |
| `config.py`                 | Path configuration                   |
| `direct_prumerge_loader.py` | PruMerge module loader               |
| `outputs/results.json`      | Experiment results                   |
| `outputs/visualizations/`   | Comparison visualizations            |

---

## Success Criteria

You're ready to go when:

✅ `verify_setup.py` shows all checks passed
✅ `run_experiment.py --mode demo --images 1` completes successfully
✅ `outputs/results.json` contains token indices
✅ No errors in console output

---

**Total setup time**: ~5 minutes
**First run time**: ~30 seconds (demo mode, 1 image)

Happy experimenting! 🚀
