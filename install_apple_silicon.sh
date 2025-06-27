#!/bin/bash

# install_apple_silicon.sh - Installation script optimized for Apple Silicon
# This script sets up the transcript generator with Apple Silicon GPU acceleration

set -e  # Exit on any error

echo "=========================================="
echo "  Apple Silicon Transcript Generator Setup"
echo "=========================================="

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠ Warning: This script is optimized for Apple Silicon (arm64)"
    echo "   You're running on: $(uname -m)"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if we're on macOS
if [[ $(uname) != "Darwin" ]]; then
    echo "✗ Error: This script is designed for macOS"
    exit 1
fi

echo "✓ Running on Apple Silicon macOS"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

echo "✓ Homebrew is installed"

# Install FFmpeg
echo "Installing FFmpeg..."
brew install ffmpeg

# Check if Python 3.10+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Installing Python 3.10+..."
    brew install python@3.10
    echo "✓ Python 3.10+ installed"
else
    echo "✓ Python $python_version is already installed"
fi

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
    echo "✓ UV installed"
else
    echo "✓ UV is already installed"
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python=3.10 .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with Apple Silicon optimization
echo "Installing PyTorch with Apple Silicon support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install additional Apple Silicon optimizations
echo "Installing Apple Silicon optimizations..."
pip install accelerate transformers

# Make scripts executable
echo "Making scripts executable..."
chmod +x whisper_transcribe.py
chmod +x test_apple_silicon.py

# Test the installation
echo "Testing installation..."
python test_apple_silicon.py

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "✓ Apple Silicon optimized transcript generator is ready!"
echo ""
echo "To use GPU acceleration:"
echo "  source .venv/bin/activate"
echo "  ./whisper_transcribe.py your_audio_file.mp3"
echo ""
echo "For speaker diarization with GPU:"
echo "  ./whisper_transcribe.py your_audio_file.mp3 --diarize"
echo ""
echo "To test GPU compatibility:"
echo "  python test_apple_silicon.py"
echo ""
echo "Happy transcribing! 🎤" 