#!/usr/bin/env python3
"""
whisper_transcribe.py - Local audio transcription using OpenAI's Whisper model
"""

import os
import sys
import argparse
import whisper
import time
from pathlib import Path


def validate_environment(verbose=True):
    """
    Validate that the environment is set up correctly

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        import torch

        if verbose:
            print(f"✓ PyTorch version: {torch.__version__}")
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ Error: PyTorch is not installed")
        return False

    # Validate ffmpeg is installed
    import shutil

    if not shutil.which("ffmpeg"):
        print("✗ Error: ffmpeg not found in PATH. Please install ffmpeg")
        return False

    if verbose:
        print("✓ ffmpeg is installed")

    return True


def validate_file(file_path, verbose=True):
    """
    Validate that the audio file exists and is accessible

    Args:
        file_path (str): Path to the audio file
        verbose (bool): Whether to print progress information

    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return False

    if not os.path.isfile(file_path):
        print(f"✗ Error: Not a file: {file_path}")
        return False

    # Check if file is readable
    try:
        with open(file_path, "rb") as f:
            f.read(1)
        if verbose:
            print(f"✓ File is readable: {file_path}")
    except PermissionError:
        print(f"✗ Error: Permission denied when reading file: {file_path}")
        return False
    except Exception as e:
        print(f"✗ Error accessing file: {file_path}, Error: {str(e)}")
        return False

    # Basic file extension check
    valid_extensions = [".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"⚠ Warning: File extension not recognized as audio: {file_path}")
        print(f"⚠ Supported extensions: {', '.join(valid_extensions)}")
        # Not returning False here as some files might have non-standard extensions

    return True


def validate_output_dir(output_dir, verbose=True):
    """
    Validate that the output directory exists or can be created

    Args:
        output_dir (str): Path to the output directory
        verbose (bool): Whether to print progress information

    Returns:
        bool: True if valid, False otherwise
    """
    # Convert to Path object for easier handling
    output_path = Path(output_dir)

    # If it exists, check if it's a directory and writable
    if output_path.exists():
        if not output_path.is_dir():
            print(f"✗ Error: Output path exists but is not a directory: {output_dir}")
            return False

        # Check if directory is writable
        try:
            test_file = output_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            if verbose:
                print(f"✓ Output directory is writable: {output_dir}")
        except PermissionError:
            print(f"✗ Error: Permission denied when writing to directory: {output_dir}")
            return False
        except Exception as e:
            print(
                f"✗ Error checking directory write access: {output_dir}, Error: {str(e)}"
            )
            return False
    else:
        # Try to create the directory
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"✓ Created output directory: {output_dir}")
        except PermissionError:
            print(f"✗ Error: Permission denied when creating directory: {output_dir}")
            return False
        except Exception as e:
            print(f"✗ Error creating directory: {output_dir}, Error: {str(e)}")
            return False

    return True


def validate_model(model_name, verbose=True):
    """
    Validate that the specified model is valid

    Args:
        model_name (str): Name of the model to use
        verbose (bool): Whether to print progress information

    Returns:
        bool: True if valid, False otherwise
    """
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2"]

    if model_name not in valid_models:
        print(f"✗ Error: Invalid model name: {model_name}")
        print(f"✓ Valid models: {', '.join(valid_models)}")
        return False

    # Check for disk space if loading larger models
    if model_name in ["medium", "large", "large-v1", "large-v2"]:
        model_sizes = {
            "medium": "1.5 GB",
            "large": "2.9 GB",
            "large-v1": "2.9 GB",
            "large-v2": "2.9 GB",
        }
        if verbose:
            print(
                f"⚠ Note: The '{model_name}' model requires approximately {model_sizes.get(model_name)} of disk space."
            )
            print("⚠ It will be downloaded on first use if not already present.")

    return True


def transcribe_audio(
    file_path,
    model_name="medium",
    language=None,
    output_dir="./",
    output_formats=["txt"],
    word_timestamps=False,
    verbose=True,
):
    """
    Transcribe an audio file using Whisper

    Args:
        file_path (str): Path to the audio file
        model_name (str): Model size (tiny, base, small, medium, large)
        language (str): Language code (e.g., 'en' for English)
        output_dir (str): Directory to save the output files
        output_formats (list): List of output formats (txt, srt, vtt, json)
        word_timestamps (bool): Whether to include timestamps for each word
        verbose (bool): Whether to print progress information

    Returns:
        dict: Transcription result
    """
    if verbose:
        print(f"Loading model: {model_name}")

    # Load the model
    model = whisper.load_model(model_name)

    if verbose:
        print(f"Transcribing: {file_path}")
        start_time = time.time()

    # Prepare transcription options
    options = {}
    if language:
        options["language"] = language
    if word_timestamps:
        options["word_timestamps"] = True

    # Perform transcription
    result = model.transcribe(file_path, **options)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the output in requested formats
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    if "txt" in output_formats:
        txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        if verbose:
            print(f"Text transcript saved to: {txt_path}")

    if "json" in output_formats:
        import json

        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"JSON data saved to: {json_path}")

    if "srt" in output_formats or "vtt" in output_formats:
        from whisper.utils import get_writer

        if "srt" in output_formats:
            srt_writer = get_writer("srt", output_dir)
            srt_writer(result, file_path)
            if verbose:
                print(
                    f"SRT subtitles saved to: {os.path.join(output_dir, base_filename)}.srt"
                )

        if "vtt" in output_formats:
            vtt_writer = get_writer("vtt", output_dir)
            vtt_writer(result, file_path)
            if verbose:
                print(
                    f"VTT subtitles saved to: {os.path.join(output_dir, base_filename)}.vtt"
                )

    if verbose:
        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed:.2f} seconds")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI's Whisper"
    )
    parser.add_argument("file", help="Audio file to transcribe")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2"],
        help="Model size to use (default: medium)",
    )
    parser.add_argument("--language", help="Language code (e.g., 'en' for English)")
    parser.add_argument(
        "--output-dir", default="./", help="Directory to save output files"
    )
    parser.add_argument(
        "--formats",
        default="txt",
        help="Comma-separated list of output formats (txt,json,srt,vtt)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include timestamps for each word",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress information"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without performing transcription",
    )

    args = parser.parse_args()

    # Parse output formats
    formats = args.formats.split(",")

    # Set verbose flag
    verbose = not args.quiet

    # Validate environment, input file, output directory, and model
    all_valid = (
        validate_environment(verbose)
        and validate_file(args.file, verbose)
        and validate_output_dir(args.output_dir, verbose)
        and validate_model(args.model, verbose)
    )

    if not all_valid:
        print("\n✗ Validation failed. Please fix the issues above and try again.")
        sys.exit(1)

    if args.dry_run:
        print("\n✓ Dry run completed successfully. All inputs are valid.")
        print(
            "✓ Run the command without --dry-run to perform the actual transcription."
        )
        sys.exit(0)

    # Perform transcription
    transcribe_audio(
        args.file,
        model_name=args.model,
        language=args.language,
        output_dir=args.output_dir,
        output_formats=formats,
        word_timestamps=args.word_timestamps,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
