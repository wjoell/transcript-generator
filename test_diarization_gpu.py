#!/usr/bin/env python3
"""
test_diarization_gpu.py - Test diarization with GPU acceleration
"""

import os
import time
import subprocess
from pathlib import Path


def create_test_audio():
    """Create a test audio file with multiple speakers"""
    test_file = "test_diarization.wav"

    if not os.path.exists(test_file):
        print("Creating test audio file with multiple speakers...")
        try:
            # Create a simple test with different frequencies to simulate speakers
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=800:duration=5",  # Speaker 1
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1200:duration=5",  # Speaker 2
                    "-filter_complex",
                    "[0:0][1:0]concat=n=2:v=0:a=1[out]",
                    "-map",
                    "[out]",
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


def test_diarization_gpu():
    """Test diarization with GPU acceleration"""
    print("=" * 60)
    print("  DIARIZATION GPU ACCELERATION TEST")
    print("=" * 60)

    # Test GPU availability
    try:
        import torch

        mps_available = torch.backends.mps.is_available()
        print(f"✓ MPS (Apple Silicon GPU) available: {mps_available}")

        if not mps_available:
            print("⚠ GPU acceleration not available")
            return False
    except ImportError:
        print("✗ PyTorch not available")
        return False

    # Test diarization setup
    try:
        from whisper_diarize import WhisperDiarizer

        diarizer = WhisperDiarizer(verbose=True)
        print(f"✓ Diarizer initialized with device: {diarizer.device}")
    except Exception as e:
        print(f"✗ Diarizer initialization failed: {e}")
        return False

    # Create test audio
    test_file = create_test_audio()
    if not test_file:
        return False

    # Test diarization with GPU
    print(f"\n--- Testing Diarization with GPU ---")
    print(f"Input file: {test_file}")
    print(f"Device: {diarizer.device}")

    try:
        start_time = time.time()

        # Run diarization (this will use the GPU-optimized pipeline)
        result = diarizer.diarize(
            file_path=test_file,
            model_name="tiny",  # Use tiny model for quick test
            output_dir="./test_output",
            output_formats=["txt", "json"],
            num_speakers=2,  # We created 2 "speakers" in the test audio
            verbose=True,
        )

        end_time = time.time()

        print(f"✓ Diarization completed in {end_time - start_time:.2f} seconds")
        print(f"✓ GPU acceleration was used: {diarizer.device != 'cpu'}")

        # Check output files
        output_files = list(Path("./test_output").glob("*.txt"))
        if output_files:
            print(f"✓ Output files created: {[f.name for f in output_files]}")

        return True

    except Exception as e:
        print(f"✗ Diarization failed: {e}")
        return False


def test_faster_whisper_integration():
    """Test Faster Whisper integration with diarization"""
    print(f"\n--- Testing Faster Whisper Integration ---")

    try:
        from faster_whisper import WhisperModel
        import torch

        # Test device selection
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        compute_type = "int8" if device == "cpu" or device == "mps" else "float16"

        print(f"Device: {device}")
        print(f"Compute type: {compute_type}")

        # Test model loading
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        print("✓ Faster Whisper model loaded successfully")

        # Test transcription
        test_file = "test_diarization.wav"
        if os.path.exists(test_file):
            segments, info = model.transcribe(test_file)
            print(f"✓ Transcription test successful")
            print(f"  Language: {info.language}")
            print(f"  Segments: {len(list(segments))}")

        return True

    except Exception as e:
        print(f"✗ Faster Whisper test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing Diarization with GPU Acceleration")
    print(
        "This demonstrates how the diarization library integrates with Faster Whisper"
    )
    print("and leverages Apple Silicon GPU acceleration.\n")

    # Test 1: GPU availability
    gpu_test = test_diarization_gpu()

    # Test 2: Faster Whisper integration
    whisper_test = test_faster_whisper_integration()

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    if gpu_test and whisper_test:
        print("✓ All tests passed!")
        print("✓ Diarization with GPU acceleration is working")
        print("✓ Faster Whisper integration is functional")
        print("\nYou can now use diarization with GPU acceleration:")
        print("  ./whisper_transcribe.py audio.mp3 --diarize")
    else:
        print("⚠ Some tests failed")
        print("Check the output above for specific issues")

    # Cleanup
    test_files = ["test_diarization.wav", "test_output"]
    for file in test_files:
        if os.path.exists(file):
            if os.path.isdir(file):
                import shutil

                shutil.rmtree(file)
            else:
                os.remove(file)

    print("\n✓ Test files cleaned up")


if __name__ == "__main__":
    main()
