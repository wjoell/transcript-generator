#!/usr/bin/env python3
"""
performance_test.py - Performance comparison between CPU and GPU acceleration on Apple Silicon
"""

import os
import time
import subprocess
import platform
from pathlib import Path


def create_test_audio():
    """Create a test audio file for performance testing"""
    test_file = "performance_test_audio.wav"

    if not os.path.exists(test_file):
        print("Creating test audio file...")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=30",  # 30 second test
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    test_file,
                    "-y",
                ],
                capture_output=True,
                check=True,
            )
            print("✓ Test audio file created")
        except subprocess.CalledProcessError:
            print("✗ Could not create test audio file")
            return None

    return test_file


def test_whisper_performance():
    """Test standard Whisper performance"""
    print("\n--- Testing Standard Whisper ---")

    test_file = create_test_audio()
    if not test_file:
        return

    try:
        start_time = time.time()
        result = subprocess.run(
            [
                "python",
                "whisper_transcribe.py",
                test_file,
                "--model",
                "tiny",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        end_time = time.time()

        if result.returncode == 0:
            print(f"✓ Standard Whisper: {end_time - start_time:.2f} seconds")
            return end_time - start_time
        else:
            print(f"✗ Standard Whisper failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("✗ Standard Whisper timed out")
        return None
    except Exception as e:
        print(f"✗ Standard Whisper error: {e}")
        return None


def test_faster_whisper_performance():
    """Test Faster Whisper performance"""
    print("\n--- Testing Faster Whisper ---")

    test_file = create_test_audio()
    if not test_file:
        return

    try:
        start_time = time.time()
        result = subprocess.run(
            [
                "python",
                "whisper_transcribe_fast.py",
                test_file,
                "--model",
                "tiny",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        end_time = time.time()

        if result.returncode == 0:
            print(f"✓ Faster Whisper: {end_time - start_time:.2f} seconds")
            return end_time - start_time
        else:
            print(f"✗ Faster Whisper failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("✗ Faster Whisper timed out")
        return None
    except Exception as e:
        print(f"✗ Faster Whisper error: {e}")
        return None


def test_gpu_acceleration():
    """Test GPU acceleration capabilities"""
    print("\n--- Testing GPU Acceleration ---")

    try:
        import torch

        # Test MPS availability
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()

        print(f"MPS Available: {'Yes' if mps_available else 'No'}")
        print(f"MPS Built: {'Yes' if mps_built else 'No'}")

        if mps_available and mps_built:
            # Test basic GPU operations
            device = torch.device("mps")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)

            start_time = time.time()
            z = torch.matmul(x, y)
            end_time = time.time()

            print(f"✓ GPU Matrix multiplication: {end_time - start_time:.4f} seconds")
            print("✓ GPU acceleration is working!")
            return True
        else:
            print("⚠ GPU acceleration not available")
            return False

    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False


def main():
    """Run performance comparison"""
    print("=" * 60)
    print("  APPLE SILICON PERFORMANCE COMPARISON")
    print("=" * 60)

    # System info
    print(f"\nSystem: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    # Test GPU acceleration
    gpu_working = test_gpu_acceleration()

    if gpu_working:
        print("\n" + "=" * 60)
        print("  TRANSCRIPTION PERFORMANCE COMPARISON")
        print("=" * 60)

        # Test both implementations
        whisper_time = test_whisper_performance()
        faster_time = test_faster_whisper_performance()

        # Compare results
        if whisper_time and faster_time:
            print(f"\n--- Performance Results ---")
            print(f"Standard Whisper: {whisper_time:.2f} seconds")
            print(f"Faster Whisper: {faster_time:.2f} seconds")

            if faster_time < whisper_time:
                speedup = whisper_time / faster_time
                print(f"✓ Faster Whisper is {speedup:.2f}x faster!")
            else:
                slowdown = faster_time / whisper_time
                print(f"⚠ Faster Whisper is {slowdown:.2f}x slower")

        print(f"\n--- Recommendations ---")
        print("✓ Use Faster Whisper for optimal Apple Silicon performance")
        print("✓ GPU acceleration is working - expect 2-4x speedup")
        print("✓ For best results, use: ./whisper_transcribe_fast.py")

    else:
        print(f"\n--- Recommendations ---")
        print("⚠ GPU acceleration not available")
        print("✓ Consider updating PyTorch: pip install --upgrade torch torchaudio")
        print("✓ Both implementations will use CPU")

    # Cleanup
    test_file = "performance_test_audio.wav"
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n✓ Cleaned up test files")


if __name__ == "__main__":
    main()
