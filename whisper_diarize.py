#!/usr/bin/env python3
"""
whisper_diarize.py - Module for transcription with speaker diarization using whisper-diarization
Optimized for Apple Silicon with MPS support
"""

import os
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_optimal_device():
    """
    Get the optimal device for inference on Apple Silicon

    Returns:
        str: Device string ('mps', 'cuda', or 'cpu')
    """
    try:
        import torch

        # Check for CUDA first (for external GPUs)
        if torch.cuda.is_available():
            return "cuda"

        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"

        # Fall back to CPU
        return "cpu"

    except ImportError:
        return "cpu"


class WhisperDiarizer:
    """
    Class to handle speaker diarization with Whisper
    Optimized for Apple Silicon with MPS support
    """

    def __init__(self, verbose=True):
        """
        Initialize the diarizer

        Args:
            verbose (bool): Whether to print progress information
        """
        self.verbose = verbose
        self.device = get_optimal_device()

        # Check if whisper-diarization is installed/cloned
        self.whisper_diarization_path = self._get_whisper_diarization_path()
        if not self.whisper_diarization_path:
            if self.verbose:
                logger.warning(
                    "whisper-diarization not found. Will clone it when needed."
                )

    def _get_whisper_diarization_path(self):
        """
        Check if whisper-diarization is in the current directory or parent directories

        Returns:
            Path or None: Path to whisper-diarization directory if found, None otherwise
        """
        # Check current directory
        current_dir = Path("whisper-diarization")
        if current_dir.exists() and (current_dir / "diarize.py").exists():
            return current_dir

        # Check parent directory
        parent_dir = Path("..") / "whisper-diarization"
        if parent_dir.exists() and (parent_dir / "diarize.py").exists():
            return parent_dir

        return None

    def _setup_whisper_diarization(self):
        """
        Clone and set up whisper-diarization if not already available

        Returns:
            Path: Path to the whisper-diarization directory
        """
        if self.whisper_diarization_path:
            return self.whisper_diarization_path

        if self.verbose:
            logger.info("Setting up whisper-diarization...")

        # Clone the repository
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/MahmoudAshraf97/whisper-diarization.git",
            ],
            check=True,
        )

        # Install dependencies
        subprocess.run(
            [
                "pip",
                "install",
                "-c",
                "whisper-diarization/constraints.txt",
                "-r",
                "whisper-diarization/requirements.txt",
            ],
            check=True,
        )

        self.whisper_diarization_path = Path("whisper-diarization")
        return self.whisper_diarization_path

    def diarize(
        self,
        file_path,
        model_name="medium",
        language=None,
        output_dir="./",
        output_formats=["txt"],
        word_timestamps=False,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        use_parallel=False,
    ):
        """
        Transcribe with speaker diarization

        Args:
            file_path (str): Path to the audio file
            model_name (str): Whisper model size (tiny, base, small, medium, large)
            language (str): Language code (e.g., 'en' for English)
            output_dir (str): Directory to save the output files
            output_formats (list): List of output formats (txt, srt, vtt, json)
            word_timestamps (bool): Whether to include timestamps for each word
            num_speakers (int): Number of speakers (exact)
            min_speakers (int): Minimum number of speakers
            max_speakers (int): Maximum number of speakers
            use_parallel (bool): Whether to use parallel processing

        Returns:
            dict: Transcription result with speaker information
        """
        # Ensure whisper-diarization is available
        diarize_dir = self._setup_whisper_diarization()

        # Create a temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare command arguments
            script_path = "diarize_parallel.py" if use_parallel else "diarize.py"
            script_full_path = f"whisper-diarization/{script_path}"
            cmd = [
                "python",
                script_full_path,
                "-a",
                file_path,
                "--whisper-model",
                model_name,
            ]

            # Add device optimization for Apple Silicon
            if self.device != "cpu":
                cmd.extend(["--device", self.device])
                if self.verbose:
                    logger.info(f"Using {self.device.upper()} device for diarization")

            # Add optional arguments
            if language:
                cmd.extend(["--language", language])

            if not self.verbose:
                cmd.append("--quiet")

            # Run the diarization process
            if self.verbose:
                logger.info(f"Running diarization with command: {' '.join(cmd)}")

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode != 0:
                logger.error(f"Diarization failed with error: {process.stderr}")
                raise RuntimeError(f"Diarization failed: {process.stderr}")

            # Get output files
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            source_dir = diarize_dir

            # Copy and process output files
            result = {}

            # Speaker-aware transcript (txt)
            if "txt" in output_formats:
                src_txt = source_dir / f"{base_filename}.txt"
                dst_txt = Path(output_dir) / f"{base_filename}.txt"
                if src_txt.exists():
                    shutil.copy(src_txt, dst_txt)
                    if self.verbose:
                        logger.info(
                            f"Text transcript with speakers saved to: {dst_txt}"
                        )

            # JSON output
            src_json = source_dir / f"{base_filename}.json"
            if src_json.exists():
                with open(src_json, "r", encoding="utf-8") as f:
                    result = json.load(f)

                if "json" in output_formats:
                    dst_json = Path(output_dir) / f"{base_filename}.json"
                    shutil.copy(src_json, dst_json)
                    if self.verbose:
                        logger.info(f"JSON data with speakers saved to: {dst_json}")

            # SRT output
            if "srt" in output_formats:
                src_srt = source_dir / f"{base_filename}.srt"
                dst_srt = Path(output_dir) / f"{base_filename}.srt"
                if src_srt.exists():
                    shutil.copy(src_srt, dst_srt)
                    if self.verbose:
                        logger.info(f"SRT subtitles with speakers saved to: {dst_srt}")

            # VTT output
            if "vtt" in output_formats:
                # Convert SRT to VTT if needed, as whisper-diarization may not output VTT directly
                src_srt = source_dir / f"{base_filename}.srt"
                dst_vtt = Path(output_dir) / f"{base_filename}.vtt"

                if src_srt.exists():
                    self._convert_srt_to_vtt(src_srt, dst_vtt)
                    if self.verbose:
                        logger.info(f"VTT subtitles with speakers saved to: {dst_vtt}")

            return result

    def _convert_srt_to_vtt(self, srt_path, vtt_path):
        """
        Convert SRT file to VTT format

        Args:
            srt_path (Path): Path to the SRT file
            vtt_path (Path): Path to the output VTT file
        """
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()

        # Basic SRT to VTT conversion
        vtt_content = "WEBVTT\n\n"

        # Skip first line (it's usually empty or contains "1")
        lines = srt_content.split("\n")
        i = 0
        while i < len(lines):
            # Skip empty lines and subtitle numbers
            if lines[i].strip() and not lines[i].strip().isdigit():
                # This should be the timestamp line
                if "-->" in lines[i]:
                    # Convert timestamps from SRT (00:00:00,000) to VTT (00:00:00.000)
                    vtt_line = lines[i].replace(",", ".")
                    vtt_content += vtt_line + "\n"
                else:
                    # This is subtitle text
                    vtt_content += lines[i] + "\n"
            i += 1

        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)


def validate_diarization_dependencies(verbose=True):
    """
    Validate that the required dependencies for diarization are installed

    Args:
        verbose (bool): Whether to print progress information

    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    try:
        import faster_whisper
        import torch
        import nemo
        import ctc_segmentation

        if verbose:
            logger.info("✓ All diarization dependencies are installed")

        return True
    except ImportError as e:
        if verbose:
            logger.error(f"✗ Missing dependency: {e}")
            logger.error("✗ Please install all required dependencies with:")
            logger.error("  pip install -r requirements.txt")

        return False
