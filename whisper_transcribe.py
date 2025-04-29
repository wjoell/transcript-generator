#!/usr/bin/env python3
"""
whisper_transcribe.py - Local audio transcription using OpenAI's Whisper model
with optional speaker diarization
"""

import os
import sys
import argparse
import whisper
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import diarization module if available
try:
    from whisper_diarize import WhisperDiarizer, validate_diarization_dependencies

    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False


def validate_environment(verbose=True):
    """
    Validate that the environment is set up correctly

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        import torch

        if verbose:
            logger.info(f"✓ PyTorch version: {torch.__version__}")
            logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("✗ Error: PyTorch is not installed")
        return False

    # Validate ffmpeg is installed
    import shutil

    if not shutil.which("ffmpeg"):
        logger.error("✗ Error: ffmpeg not found in PATH. Please install ffmpeg")
        return False

    if verbose:
        logger.info("✓ ffmpeg is installed")

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
        logger.error(f"✗ Error: File not found: {file_path}")
        return False

    if not os.path.isfile(file_path):
        logger.error(f"✗ Error: Not a file: {file_path}")
        return False

    # Check if file is readable
    try:
        with open(file_path, "rb") as f:
            f.read(1)
        if verbose:
            logger.info(f"✓ File is readable: {file_path}")
    except PermissionError:
        logger.error(f"✗ Error: Permission denied when reading file: {file_path}")
        return False
    except Exception as e:
        logger.error(f"✗ Error accessing file: {file_path}, Error: {str(e)}")
        return False

    # Basic file extension check
    valid_extensions = [".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        logger.warning(
            f"⚠ Warning: File extension not recognized as audio: {file_path}"
        )
        logger.warning(f"⚠ Supported extensions: {', '.join(valid_extensions)}")
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
            logger.error(
                f"✗ Error: Output path exists but is not a directory: {output_dir}"
            )
            return False

        # Check if directory is writable
        try:
            test_file = output_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            if verbose:
                logger.info(f"✓ Output directory is writable: {output_dir}")
        except PermissionError:
            logger.error(
                f"✗ Error: Permission denied when writing to directory: {output_dir}"
            )
            return False
        except Exception as e:
            logger.error(
                f"✗ Error checking directory write access: {output_dir}, Error: {str(e)}"
            )
            return False
    else:
        # Try to create the directory
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                logger.info(f"✓ Created output directory: {output_dir}")
        except PermissionError:
            logger.error(
                f"✗ Error: Permission denied when creating directory: {output_dir}"
            )
            return False
        except Exception as e:
            logger.error(f"✗ Error creating directory: {output_dir}, Error: {str(e)}")
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
    valid_models = [
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
    ]

    if model_name not in valid_models:
        logger.error(f"✗ Error: Invalid model name: {model_name}")
        logger.info(f"✓ Valid models: {', '.join(valid_models)}")
        return False

    # Check for disk space if loading larger models
    if model_name in ["medium", "large", "large-v1", "large-v2", "large-v3"]:
        model_sizes = {
            "medium": "1.5 GB",
            "large": "2.9 GB",
            "large-v1": "2.9 GB",
            "large-v2": "2.9 GB",
            "large-v3": "3.1 GB",
        }
        if verbose:
            logger.warning(
                f"⚠ Note: The '{model_name}' model requires approximately {model_sizes.get(model_name, '3 GB')} of disk space."
            )
            logger.warning(
                "⚠ It will be downloaded on first use if not already present."
            )

    return True


def validate_diarization_options(
    num_speakers, min_speakers, max_speakers, verbose=True
):
    """
    Validate speaker diarization options

    Args:
        num_speakers (int): Exact number of speakers
        min_speakers (int): Minimum number of speakers
        max_speakers (int): Maximum number of speakers
        verbose (bool): Whether to print progress information

    Returns:
        bool: True if valid, False otherwise
    """
    # Check if diarization is available
    if not DIARIZATION_AVAILABLE:
        logger.error("✗ Error: Diarization module is not available")
        logger.error("✗ Please install the required dependencies with:")
        logger.error("  pip install -r requirements.txt")
        return False

    # Validate diarization dependencies
    if not validate_diarization_dependencies(verbose):
        return False

    # Check for conflicts in speaker count options
    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        logger.error(
            "✗ Error: Cannot specify both exact number of speakers and min/max speakers"
        )
        return False

    # Validate number ranges
    if num_speakers is not None and num_speakers < 1:
        logger.error("✗ Error: Number of speakers must be at least 1")
        return False

    if min_speakers is not None and min_speakers < 1:
        logger.error("✗ Error: Minimum number of speakers must be at least 1")
        return False

    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        logger.error(
            "✗ Error: Minimum number of speakers cannot be greater than maximum"
        )
        return False

    return True


def transcribe_audio(
    file_path,
    model_name="medium",
    language=None,
    output_dir="./",
    output_formats=["txt"],
    word_timestamps=False,
    verbose=True,
    diarize=False,
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
    parallel_diarize=False,
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
        diarize (bool): Whether to use speaker diarization
        num_speakers (int): Exact number of speakers
        min_speakers (int): Minimum number of speakers
        max_speakers (int): Maximum number of speakers
        parallel_diarize (bool): Whether to use parallel processing for diarization

    Returns:
        dict: Transcription result
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # If diarization is requested and available, use it
    if diarize and DIARIZATION_AVAILABLE:
        if verbose:
            logger.info(f"Using speaker diarization with model: {model_name}")

        diarizer = WhisperDiarizer(verbose=verbose)
        result = diarizer.diarize(
            file_path=file_path,
            model_name=model_name,
            language=language,
            output_dir=output_dir,
            output_formats=output_formats,
            word_timestamps=word_timestamps,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            use_parallel=parallel_diarize,
        )

        return result

    # Otherwise, use standard Whisper transcription
    if verbose:
        logger.info(f"Loading model: {model_name}")

    # Load the model
    model = whisper.load_model(model_name)

    if verbose:
        logger.info(f"Transcribing: {file_path}")
        start_time = time.time()

    # Prepare transcription options
    options = {}
    if language:
        options["language"] = language
    if word_timestamps:
        options["word_timestamps"] = True

    # Perform transcription
    result = model.transcribe(file_path, **options)

    # Save the output in requested formats
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    if "txt" in output_formats:
        txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        if verbose:
            logger.info(f"Text transcript saved to: {txt_path}")

    if "json" in output_formats:
        import json

        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            logger.info(f"JSON data saved to: {json_path}")

    if "srt" in output_formats or "vtt" in output_formats:
        from whisper.utils import get_writer

        if "srt" in output_formats:
            srt_writer = get_writer("srt", output_dir)
            srt_writer(result, file_path)
            if verbose:
                logger.info(
                    f"SRT subtitles saved to: {os.path.join(output_dir, base_filename)}.srt"
                )

        if "vtt" in output_formats:
            vtt_writer = get_writer("vtt", output_dir)
            vtt_writer(result, file_path)
            if verbose:
                logger.info(
                    f"VTT subtitles saved to: {os.path.join(output_dir, base_filename)}.vtt"
                )

    if verbose:
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f} seconds")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI's Whisper with optional speaker diarization"
    )
    parser.add_argument("file", help="Audio file to transcribe")
    parser.add_argument(
        "--model",
        default="medium",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
        ],
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

    # Add diarization options
    diarization_group = parser.add_argument_group("Speaker Diarization Options")
    diarization_group.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization to identify different speakers",
    )

    speaker_count_group = diarization_group.add_mutually_exclusive_group()
    speaker_count_group.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers in the audio (if known)",
    )

    speaker_count_group.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers to detect",
    )

    diarization_group.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to detect (use with --min-speakers)",
    )

    diarization_group.add_argument(
        "--parallel-diarize",
        action="store_true",
        help="Use parallel processing for diarization (requires more VRAM)",
    )

    args = parser.parse_args()

    # Parse output formats
    formats = args.formats.split(",")

    # Set verbose flag
    verbose = not args.quiet

    # Validate environment, input file, output directory, and model
    validation_checks = [
        validate_environment(verbose),
        validate_file(args.file, verbose),
        validate_output_dir(args.output_dir, verbose),
        validate_model(args.model, verbose),
    ]

    # Validate diarization options if enabled
    if args.diarize:
        validation_checks.append(
            validate_diarization_options(
                args.num_speakers, args.min_speakers, args.max_speakers, verbose
            )
        )

    all_valid = all(validation_checks)

    if not all_valid:
        logger.error(
            "\n✗ Validation failed. Please fix the issues above and try again."
        )
        sys.exit(1)

    if args.dry_run:
        logger.info("\n✓ Dry run completed successfully. All inputs are valid.")
        logger.info(
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
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        parallel_diarize=args.parallel_diarize,
    )


if __name__ == "__main__":
    main()
