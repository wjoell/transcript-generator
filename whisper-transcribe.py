#!/usr/bin/env python3
"""
whisper_transcribe.py - Local audio transcription using OpenAI's Whisper model
"""

import os
import argparse
import whisper
import time


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
        choices=["tiny", "base", "small", "medium", "large"],
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

    args = parser.parse_args()

    # Parse output formats
    formats = args.formats.split(",")

    # Perform transcription
    transcribe_audio(
        args.file,
        model_name=args.model,
        language=args.language,
        output_dir=args.output_dir,
        output_formats=formats,
        word_timestamps=args.word_timestamps,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
