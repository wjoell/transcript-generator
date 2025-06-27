#!/bin/bash
# apply_patches.sh - Apply Apple Silicon and compatibility patches to whisper-diarization

set -e

echo "Applying patches to whisper-diarization..."

# Check if we're in the whisper-diarization directory
if [ ! -f "diarize.py" ]; then
    echo "Error: diarize.py not found. Make sure you're in the whisper-diarization directory"
    exit 1
fi

# Patch 1: Add MPS support to mtypes
echo "Patching MPS support..."
sed -i.bak 's/mtypes = {"cpu": "int8", "cuda": "float16"}/mtypes = {"cpu": "int8", "cuda": "float16", "mps": "int8"}/' diarize.py

# Patch 2: Fix device handling for Faster Whisper
echo "Patching device handling..."
sed -i.bak2 '/# Transcribe the audio file/a\
\
# Use '\''cpu'\'' for faster_whisper if device is '\''mps'\'', but keep '\''mps'\'' for PyTorch/NeMo\
fw_device = args.device if args.device != "mps" else "cpu"\
' diarize.py

sed -i.bak3 's/args.model_name, device=args.device, compute_type=mtypes\[args.device\]/args.model_name, device=fw_device, compute_type=mtypes[fw_device]/' diarize.py

# Patch 3: Fix punctuation model call
echo "Patching punctuation model..."
sed -i.bak4 's/labled_words = punct_model.predict(words_list, chunk_size=230)/labled_words = punct_model.predict(words_list)/' diarize.py

# Patch 4: Apply NeMo PyTorch 2.x compatibility (if nemo is installed)
if [ -f "/Users/winston/whisper-env/lib/python3.12/site-packages/nemo/collections/asr/modules/msdd_diarizer.py" ]; then
    echo "Patching NeMo for PyTorch 2.x compatibility..."
    sed -i.bak5 's/\.view(/\.reshape(/g' /Users/winston/whisper-env/lib/python3.12/site-packages/nemo/collections/asr/modules/msdd_diarizer.py
fi

echo "Patches applied successfully!"
echo "Backup files created with .bak extensions" 