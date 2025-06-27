#!/usr/bin/env python3
"""
whisper_transcribe_fast.py - Fast transcription using Faster Whisper with Apple Silicon optimization
"""

import os
import sys
import argparse
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
        import faster_whisper

        if verbose:
            logger.info(f"✓ PyTorch version: {torch.__version__}")
            logger.info(f"✓ Faster Whisper version: {faster_whisper.__version__}")

            # Test CUDA support
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {'Yes' if cuda_available else 'No'}")

            if cuda_available:
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

            # Test MPS support (Apple Silicon)
            mps_available = torch.backends.mps.is_available()
            mps_built = torch.backends.mps.is_built()
            logger.info(f"MPS available: {'Yes' if mps_available else 'No'}")
            logger.info(f"MPS built: {'Yes' if mps_built else 'No'}")

            if mps_available and mps_built:
                logger.info("✓ Apple Silicon GPU acceleration is available!")
            else:
                logger.info("ℹ Apple Silicon GPU acceleration not available")

    except ImportError as e:
        logger.error(f"✗ Error: Required packages not installed: {e}")
        return False

    # Validate ffmpeg is installed
    import shutil

    if not shutil.which("ffmpeg"):
        logger.error("✗ Error: ffmpeg not found in PATH. Please install ffmpeg")
        return False

    if verbose:
        logger.info("✓ ffmpeg is installed")

    return True


def get_optimal_compute_type():
    """
    Get the optimal compute type for Faster Whisper on Apple Silicon

    Returns:
        str: Compute type ('int8', 'float16', or 'float32')
    """
    try:
        import torch

        # Check for CUDA
        if torch.cuda.is_available():
            return "float16"  # Best performance on CUDA

        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "int8"  # Best performance on Apple Silicon

        # Fall back to CPU
        return "int8"  # Best performance on CPU

    except ImportError:
        return "int8"


def transcribe_audio_fast(
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
    Transcribe an audio file using Faster Whisper

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

    # Otherwise, use Faster Whisper transcription
    if verbose:
        logger.info(f"Loading Faster Whisper model: {model_name}")

    # Get optimal compute type for Apple Silicon
    compute_type = get_optimal_compute_type()
    if verbose:
        logger.info(f"Using compute type: {compute_type}")

    # Load the model
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, compute_type=compute_type)

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
    segments, info = model.transcribe(file_path, **options)

    # Process results
    result = {
        "text": "",
        "segments": [],
        "language": info.language,
        "language_probability": info.language_probability,
    }

    for segment in segments:
        segment_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [],
        }

        if word_timestamps and segment.words:
            for word in segment.words:
                segment_data["words"].append(
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                )

        result["segments"].append(segment_data)
        result["text"] += segment.text + " "

    result["text"] = result["text"].strip()

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

    if "srt" in output_formats:
        srt_path = os.path.join(output_dir, f"{base_filename}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start_time_str = format_timestamp(segment["start"])
                end_time_str = format_timestamp(segment["end"])
                f.write(f"{i}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment['text']}\n\n")
        if verbose:
            logger.info(f"SRT subtitles saved to: {srt_path}")

    if "vtt" in output_formats:
        vtt_path = os.path.join(output_dir, f"{base_filename}.vtt")
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, segment in enumerate(result["segments"], 1):
                start_time_str = format_timestamp(segment["start"])
                end_time_str = format_timestamp(segment["end"])
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment['text']}\n\n")
        if verbose:
            logger.info(f"VTT subtitles saved to: {vtt_path}")

    if verbose:
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f} seconds")

    return result


def format_timestamp(seconds):
    """Format seconds to SRT/VTT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Fast transcription using Faster Whisper with Apple Silicon optimization"
    )
    parser.add_argument("file", help="Audio file to transcribe")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
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

    # Speaker diarization options
    parser.add_argument(
        "--diarize", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers in the audio (if known)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers to detect",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to detect (use with --min-speakers)",
    )
    parser.add_argument(
        "--parallel-diarize",
        action="store_true",
        help="Use parallel processing for diarization (requires more VRAM)",
    )

    args = parser.parse_args()

    # Validate environment
    if not validate_environment(verbose=not args.quiet):
        sys.exit(1)

    # Validate file
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    # Parse output formats
    output_formats = [fmt.strip() for fmt in args.formats.split(",")]

    # Validate output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        logger.info("Dry run completed successfully")
        return

    # Perform transcription
    try:
        result = transcribe_audio_fast(
            file_path=args.file,
            model_name=args.model,
            language=args.language,
            output_dir=args.output_dir,
            output_formats=output_formats,
            word_timestamps=args.word_timestamps,
            verbose=not args.quiet,
            diarize=args.diarize,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            parallel_diarize=args.parallel_diarize,
        )

        if not args.quiet:
            logger.info("Transcription completed successfully!")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
