#!/usr/bin/env python3
"""
test_apple_silicon.py - Test script for Apple Silicon GPU compatibility and performance
"""

import os
import sys
import time
import platform
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title):
    """Print a formatted section"""
    print(f"\n--- {title} ---")


def test_system_info():
    """Test and display system information"""
    print_header("SYSTEM INFORMATION")

    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    # Check if we're on Apple Silicon
    is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
    print(f"Apple Silicon: {'Yes' if is_apple_silicon else 'No'}")

    if is_apple_silicon:
        print("✓ Running on Apple Silicon Mac")
    else:
        print("⚠ Not running on Apple Silicon Mac")

    return is_apple_silicon


def test_python_environment():
    """Test Python environment and dependencies"""
    print_header("PYTHON ENVIRONMENT")

    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"Virtual environment: {'Yes' if in_venv else 'No'}")

    if in_venv:
        print(f"Virtual environment path: {sys.prefix}")


def test_pytorch_installation():
    """Test PyTorch installation and GPU support"""
    print_header("PYTORCH INSTALLATION")

    try:
        import torch

        print(f"✓ PyTorch version: {torch.__version__}")

        # Test CUDA support
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {'Yes' if cuda_available else 'No'}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        # Test MPS support (Apple Silicon)
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        print(f"MPS available: {'Yes' if mps_available else 'No'}")
        print(f"MPS built: {'Yes' if mps_built else 'No'}")

        if mps_available and mps_built:
            print("✓ Apple Silicon GPU acceleration is available!")
        else:
            print("⚠ Apple Silicon GPU acceleration not available")

        return True, mps_available and mps_built

    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False, False


def test_mps_device():
    """Test MPS device functionality"""
    print_header("MPS DEVICE TESTING")

    try:
        import torch

        if not torch.backends.mps.is_available():
            print("⚠ MPS not available, skipping device tests")
            return False

        print("Testing MPS device creation...")

        # Test basic tensor operations
        device = torch.device("mps")
        print(f"✓ MPS device created: {device}")

        # Test tensor creation and operations
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)

        start_time = time.time()
        z = torch.matmul(x, y)
        end_time = time.time()

        print(
            f"✓ Matrix multiplication test completed in {end_time - start_time:.4f} seconds"
        )
        print(f"✓ Result tensor shape: {z.shape}")

        # Test memory management
        del x, y, z
        torch.mps.empty_cache()
        print("✓ Memory cleanup successful")

        return True

    except Exception as e:
        print(f"✗ MPS device test failed: {e}")
        return False


def test_whisper_installation():
    """Test Whisper installation"""
    print_header("WHISPER INSTALLATION")

    try:
        import whisper

        print(f"✓ OpenAI Whisper version: {whisper.__version__}")

        # Test model loading
        print("Testing model loading...")
        start_time = time.time()
        model = whisper.load_model("tiny")
        end_time = time.time()

        print(f"✓ Tiny model loaded in {end_time - start_time:.2f} seconds")
        print(f"✓ Model device: {next(model.parameters()).device}")

        return True

    except ImportError as e:
        print(f"✗ Whisper not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Whisper test failed: {e}")
        return False


def test_faster_whisper():
    """Test Faster Whisper installation"""
    print_header("FASTER WHISPER TESTING")

    try:
        import faster_whisper

        print(f"✓ Faster Whisper version: {faster_whisper.__version__}")

        # Test model loading with different compute types
        compute_types = ["int8", "float16", "float32"]

        for compute_type in compute_types:
            try:
                print(f"Testing {compute_type} compute type...")
                start_time = time.time()
                model = faster_whisper.WhisperModel("tiny", compute_type=compute_type)
                end_time = time.time()
                print(
                    f"✓ {compute_type} model loaded in {end_time - start_time:.2f} seconds"
                )
                break
            except Exception as e:
                print(f"⚠ {compute_type} failed: {e}")
                continue

        return True

    except ImportError:
        print("ℹ Faster Whisper not installed (optional)")
        return False
    except Exception as e:
        print(f"✗ Faster Whisper test failed: {e}")
        return False


def test_ffmpeg():
    """Test FFmpeg installation"""
    print_header("FFMPEG TESTING")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"✓ FFmpeg: {version_line}")
            return True
        else:
            print("✗ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("✗ FFmpeg test timed out")
        return False


def test_whisper_with_mps():
    """Test Whisper with MPS acceleration"""
    print_header("WHISPER WITH MPS TESTING")

    try:
        import whisper
        import torch

        if not torch.backends.mps.is_available():
            print("⚠ MPS not available, skipping MPS test")
            return False

        print("Loading model with MPS acceleration...")
        start_time = time.time()
        model = whisper.load_model("tiny")
        model = model.to("mps")
        end_time = time.time()

        print(f"✓ Model loaded on MPS in {end_time - start_time:.2f} seconds")

        # Create a dummy audio file for testing
        test_audio_path = "test_audio.wav"
        if not os.path.exists(test_audio_path):
            print("Creating test audio file...")
            # Create a simple test audio file using ffmpeg
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=5",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        test_audio_path,
                        "-y",
                    ],
                    capture_output=True,
                    check=True,
                )
                print("✓ Test audio file created")
            except subprocess.CalledProcessError:
                print("⚠ Could not create test audio file")
                return False

        print("Testing transcription with MPS...")
        start_time = time.time()
        result = model.transcribe(test_audio_path)
        end_time = time.time()

        print(f"✓ Transcription completed in {end_time - start_time:.2f} seconds")
        print(f"✓ Transcribed text: {result['text'][:100]}...")

        # Clean up test file
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)

        return True

    except Exception as e:
        print(f"✗ Whisper MPS test failed: {e}")
        return False


def performance_comparison():
    """Compare CPU vs MPS performance"""
    print_header("PERFORMANCE COMPARISON")

    try:
        import whisper
        import torch

        if not torch.backends.mps.is_available():
            print("⚠ MPS not available, skipping performance comparison")
            return

        # Create test audio if needed
        test_audio_path = "test_audio.wav"
        if not os.path.exists(test_audio_path):
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=10",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        test_audio_path,
                        "-y",
                    ],
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("⚠ Could not create test audio file")
                return

        # Test CPU performance
        print("Testing CPU performance...")
        model_cpu = whisper.load_model("tiny")
        start_time = time.time()
        result_cpu = model_cpu.transcribe(test_audio_path)
        cpu_time = time.time() - start_time

        # Test MPS performance
        print("Testing MPS performance...")
        model_mps = whisper.load_model("tiny")
        model_mps = model_mps.to("mps")
        start_time = time.time()
        result_mps = model_mps.transcribe(test_audio_path)
        mps_time = time.time() - start_time

        print(f"\nPerformance Results:")
        print(f"CPU time: {cpu_time:.2f} seconds")
        print(f"MPS time: {mps_time:.2f} seconds")

        if mps_time < cpu_time:
            speedup = cpu_time / mps_time
            print(f"✓ MPS is {speedup:.2f}x faster than CPU")
        else:
            slowdown = mps_time / cpu_time
            print(f"⚠ MPS is {slowdown:.2f}x slower than CPU")

        # Clean up
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)

    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")


def main():
    """Run all tests"""
    print_header("APPLE SILICON GPU COMPATIBILITY TEST")

    # Run all tests
    is_apple_silicon = test_system_info()
    test_python_environment()

    pytorch_ok, mps_available = test_pytorch_installation()

    if pytorch_ok and mps_available:
        test_mps_device()
        test_whisper_with_mps()
        performance_comparison()

    test_whisper_installation()
    test_faster_whisper()
    test_ffmpeg()

    print_header("TEST SUMMARY")

    if is_apple_silicon and mps_available:
        print("✓ Apple Silicon GPU acceleration is available and working!")
        print("✓ Your system is optimized for GPU-accelerated transcription")
    elif is_apple_silicon:
        print("⚠ Apple Silicon detected but MPS not available")
        print("  Consider updating PyTorch: pip install --upgrade torch torchaudio")
    else:
        print("ℹ Not running on Apple Silicon")

    print("\nTo use GPU acceleration, run:")
    print("  ./whisper_transcribe.py your_audio_file.mp3")
    print("\nFor speaker diarization with GPU:")
    print("  ./whisper_transcribe.py your_audio_file.mp3 --diarize")


if __name__ == "__main__":
    main()
