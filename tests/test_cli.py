#!/usr/bin/env python3
"""
Tests for the command-line interface in whisper_transcribe.py
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import argparse

# Add the parent directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import whisper_transcribe


class TestCommandLineInterface(unittest.TestCase):
    """Test cases for the command-line interface"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create a test audio file
        self.test_audio_file = os.path.join(self.test_dir, "test_audio.mp3")
        with open(self.test_audio_file, "wb") as f:
            f.write(b"dummy audio content")

        # Create an output directory
        self.test_output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.test_output_dir)

    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("whisper_transcribe.validate_environment")
    @patch("whisper_transcribe.validate_file")
    @patch("whisper_transcribe.validate_output_dir")
    @patch("whisper_transcribe.validate_model")
    @patch("whisper_transcribe.transcribe_audio")
    def test_main_function(
        self,
        mock_transcribe,
        mock_validate_model,
        mock_validate_output,
        mock_validate_file,
        mock_validate_env,
        mock_parse_args,
    ):
        """Test the main function with default arguments"""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.file = self.test_audio_file
        mock_args.model = "medium"
        mock_args.language = None
        mock_args.output_dir = "./"
        mock_args.formats = "txt"
        mock_args.word_timestamps = False
        mock_args.quiet = False
        mock_args.dry_run = False
        mock_parse_args.return_value = mock_args

        # All validation functions return True
        mock_validate_env.return_value = True
        mock_validate_file.return_value = True
        mock_validate_output.return_value = True
        mock_validate_model.return_value = True

        # Call the main function
        with patch.object(sys, "argv", ["whisper_transcribe.py", self.test_audio_file]):
            whisper_transcribe.main()

        # Check that validation functions were called
        mock_validate_env.assert_called_once_with(True)  # verbose=True
        mock_validate_file.assert_called_once_with(self.test_audio_file, True)
        mock_validate_output.assert_called_once_with("./", True)
        mock_validate_model.assert_called_once_with("medium", True)

        # Check that transcribe_audio was called with the right arguments
        mock_transcribe.assert_called_once_with(
            self.test_audio_file,
            model_name="medium",
            language=None,
            output_dir="./",
            output_formats=["txt"],
            word_timestamps=False,
            verbose=True,
        )

    @patch("argparse.ArgumentParser.parse_args")
    @patch("whisper_transcribe.validate_environment")
    @patch("whisper_transcribe.validate_file")
    @patch("whisper_transcribe.validate_output_dir")
    @patch("whisper_transcribe.validate_model")
    @patch("whisper_transcribe.transcribe_audio")
    def test_main_with_options(
        self,
        mock_transcribe,
        mock_validate_model,
        mock_validate_output,
        mock_validate_file,
        mock_validate_env,
        mock_parse_args,
    ):
        """Test the main function with custom options"""
        # Set up mocks with custom options
        mock_args = MagicMock()
        mock_args.file = self.test_audio_file
        mock_args.model = "large-v2"
        mock_args.language = "en"
        mock_args.output_dir = self.test_output_dir
        mock_args.formats = "txt,srt,json"
        mock_args.word_timestamps = True
        mock_args.quiet = True
        mock_args.dry_run = False
        mock_parse_args.return_value = mock_args

        # All validation functions return True
        mock_validate_env.return_value = True
        mock_validate_file.return_value = True
        mock_validate_output.return_value = True
        mock_validate_model.return_value = True

        # Call the main function
        with patch.object(
            sys,
            "argv",
            [
                "whisper_transcribe.py",
                self.test_audio_file,
                "--model",
                "large-v2",
                "--language",
                "en",
                "--output-dir",
                self.test_output_dir,
                "--formats",
                "txt,srt,json",
                "--word-timestamps",
                "--quiet",
            ],
        ):
            whisper_transcribe.main()

        # Check that validation functions were called with quiet mode
        mock_validate_env.assert_called_once_with(
            False
        )  # verbose=False due to quiet mode
        mock_validate_file.assert_called_once_with(self.test_audio_file, False)
        mock_validate_output.assert_called_once_with(self.test_output_dir, False)
        mock_validate_model.assert_called_once_with("large-v2", False)

        # Check that transcribe_audio was called with the right arguments
        mock_transcribe.assert_called_once_with(
            self.test_audio_file,
            model_name="large-v2",
            language="en",
            output_dir=self.test_output_dir,
            output_formats=["txt", "srt", "json"],
            word_timestamps=True,
            verbose=False,
        )

    @patch("argparse.ArgumentParser.parse_args")
    @patch("whisper_transcribe.validate_environment")
    @patch("whisper_transcribe.transcribe_audio")
    @patch("sys.exit")
    def test_main_validation_failure(
        self, mock_exit, mock_transcribe, mock_validate_env, mock_parse_args
    ):
        """Test main function when validation fails"""
        # Set up mock arguments
        mock_args = MagicMock()
        mock_args.file = self.test_audio_file
        mock_args.model = "medium"
        mock_args.language = None
        mock_args.output_dir = "./"
        mock_args.formats = "txt"
        mock_args.word_timestamps = False
        mock_args.quiet = False
        mock_args.dry_run = False
        mock_parse_args.return_value = mock_args

        # Make validation fail
        mock_validate_env.return_value = False

        # Set up mock_exit to simulate program exit by raising an exception
        mock_exit.side_effect = SystemExit(1)

        # Call the main function
        with patch.object(sys, "argv", ["whisper_transcribe.py", self.test_audio_file]):
            with self.assertRaises(SystemExit):
                whisper_transcribe.main()

        # Check that sys.exit was called with code 1 (error)
        mock_exit.assert_called_once_with(1)
        
        # Verify that transcribe_audio was not called
        mock_transcribe.assert_not_called()

    @patch("argparse.ArgumentParser.parse_args")
    @patch("whisper_transcribe.validate_environment")
    @patch("whisper_transcribe.validate_file")
    @patch("whisper_transcribe.validate_output_dir")
    @patch("whisper_transcribe.validate_model")
    @patch("whisper_transcribe.transcribe_audio")
    @patch("sys.exit")
    def test_main_dry_run(
        self,
        mock_exit,
        mock_transcribe,
        mock_validate_model,
        mock_validate_output,
        mock_validate_file,
        mock_validate_env,
        mock_parse_args,
    ):
        """Test the main function with dry-run option"""
        # Set up mocks with dry-run option
        mock_args = MagicMock()
        mock_args.file = self.test_audio_file
        mock_args.model = "medium"
        mock_args.language = None
        mock_args.output_dir = "./"
        mock_args.formats = "txt"
        mock_args.word_timestamps = False
        mock_args.quiet = False
        mock_args.dry_run = True
        mock_parse_args.return_value = mock_args

        # All validation functions return True
        mock_validate_env.return_value = True
        mock_validate_file.return_value = True
        mock_validate_output.return_value = True
        mock_validate_model.return_value = True
        
        # Set up mock_exit to simulate program exit by raising an exception
        mock_exit.side_effect = SystemExit(0)

        # Call the main function
        with patch.object(
            sys, "argv", ["whisper_transcribe.py", self.test_audio_file, "--dry-run"]
        ):
            with self.assertRaises(SystemExit):
                whisper_transcribe.main()

        # Check that validation functions were called
        mock_validate_env.assert_called_once()
        mock_validate_file.assert_called_once()
        mock_validate_output.assert_called_once()
        mock_validate_model.assert_called_once()

        # Check that transcribe_audio was NOT called due to dry-run
        mock_transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
