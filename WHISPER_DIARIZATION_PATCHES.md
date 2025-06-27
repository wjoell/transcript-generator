# Whisper-Diarization Patches

This document tracks the patches made to the `whisper-diarization` library to enable Apple Silicon support and fix compatibility issues.

## Overview

The original `whisper-diarization` library has been patched to work optimally on Apple Silicon Macs with MPS (Metal Performance Shaders) support. These patches address several compatibility issues with modern Python/PyTorch versions.

## Patches Applied

### 1. Apple Silicon MPS Support

**File:** `whisper-diarization/diarize.py`

**Changes:**

-   Added `"mps": "int8"` to the `mtypes` dictionary
-   Added device handling logic to use `"cpu"` for Faster Whisper while keeping `"mps"` for PyTorch/NeMo operations

**Before:**

```python
mtypes = {"cpu": "int8", "cuda": "float16"}
whisper_model = faster_whisper.WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device]
)
```

**After:**

```python
mtypes = {"cpu": "int8", "cuda": "float16", "mps": "int8"}

# Use 'cpu' for faster_whisper if device is 'mps', but keep 'mps' for PyTorch/NeMo
fw_device = args.device if args.device != "mps" else "cpu"

whisper_model = faster_whisper.WhisperModel(
    args.model_name, device=fw_device, compute_type=mtypes[fw_device]
)
```

**Rationale:** Faster Whisper (CTranslate2) doesn't support MPS, but PyTorch/NeMo does. This patch ensures optimal device usage for each component.

### 2. PyTorch 2.x Compatibility

**File:** `/Users/winston/whisper-env/lib/python3.12/site-packages/nemo/collections/asr/modules/msdd_diarizer.py`

**Changes:**

-   Replaced `.view()` calls with `.reshape()` for PyTorch 2.x compatibility

**Before:**

```python
context_vectors = context_vectors.view(self.batch_size, self.length, -1)
lin_input_seq = conv_out.view(self.batch_size, self.length, self.cnn_output_ch * self.emb_dim)
```

**After:**

```python
context_vectors = context_vectors.reshape(self.batch_size, self.length, -1)
lin_input_seq = conv_out.reshape(self.batch_size, self.length, self.cnn_output_ch * self.emb_dim)
```

**Rationale:** PyTorch 2.x changed tensor behavior, making `.view()` fail on non-contiguous tensors. `.reshape()` handles this gracefully.

### 3. Punctuation Model Compatibility

**File:** `whisper-diarization/diarize.py`

**Changes:**

-   Removed unsupported `chunk_size` argument from punctuation model call

**Before:**

```python
labled_words = punct_model.predict(words_list, chunk_size=230)
```

**After:**

```python
labled_words = punct_model.predict(words_list)
```

**Rationale:** The installed version of `deepmultilingualpunctuation` doesn't support the `chunk_size` parameter.

### 4. NumPy Compatibility

**Environment Change:**

-   Downgraded NumPy from 2.x to 1.26.4 for NeMo compatibility

**Command:**

```bash
uv pip install "numpy<2.0"
```

**Rationale:** NeMo 2.1.0 uses deprecated NumPy APIs that were removed in NumPy 2.0.

## Maintenance

### Applying Patches to Fresh Installations

Use the `apply_patches.sh` script to automatically apply these patches:

```bash
./apply_patches.sh
```

### Updating Dependencies

When updating dependencies, be aware that:

1. **NeMo updates** may require re-applying the PyTorch 2.x patches
2. **NumPy updates** beyond 2.0 may break NeMo compatibility
3. **Faster Whisper updates** may change device handling requirements

### Recommended Workflow

1. **Fork the original whisper-diarization repo**
2. **Apply these patches to your fork**
3. **Update your transcript-generator to use your forked version**
4. **Submit PRs to the original repo** with your patches

## Testing

After applying patches, test with:

```bash
python whisper-diarization/diarize.py -a "test_audio.mp3" --whisper-model tiny --device mps
```

The pipeline should complete successfully with Apple Silicon GPU acceleration.

## Dependencies

These patches require:

-   Python 3.10+
-   PyTorch with MPS support
-   NumPy < 2.0
-   NeMo 2.1.0
-   Faster Whisper
-   All other whisper-diarization dependencies

## Notes

-   The NeMo patches are applied to the installed package in your virtual environment
-   These patches may need to be re-applied after environment updates
-   Consider contributing these patches upstream to benefit the broader community
