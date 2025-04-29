#!/usr/bin/env python3
"""
Tests for the transcription functionality in whisper_transcribe.py
"""

import os
import unittest
import tempfile
import shutil
import sys
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whisper_transcribe import transcribe_audio


class TestTranscribe(unittest.TestCase):
    """Test cases for transcription functionality"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir)

        # Create a mock audio file
        self.test_audio_file = os.path.join(self.test_dir, "test_audio.mp3")
        with open(self.test_audio_file, "wb") as f:
            f.write(b"mock audio content")

    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)

    @patch("whisper.load_model")
    def test_transcribe_audio_txt(self, mock_load_model):
        """Test basic transcription with txt output"""
        # Create mock model and result
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "This is a test transcription.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.0,
                    "text": "This is a test transcription.",
                }
            ],
        }
        mock_load_model.return_value = mock_model

        # Call the function
        result = transcribe_audio(
            file_path=self.test_audio_file,
            model_name="tiny",
            output_dir=self.output_dir,
            output_formats=["txt"],
            verbose=False,
        )

        # Check that the model was loaded correctly
        mock_load_model.assert_called_once_with("tiny")

        # Check that the transcription was called with the right file
        mock_model.transcribe.assert_called_once()
        args, kwargs = mock_model.transcribe.call_args
        self.assertEqual(args[0], self.test_audio_file)

        # Check the output file was created
        output_txt = os.path.join(self.output_dir, "test_audio.txt")
        self.assertTrue(os.path.exists(output_txt))

        # Check the content of the output file
        with open(output_txt, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, "This is a test transcription.")

        # Check the returned result
        self.assertEqual(result["text"], "This is a test transcription.")

    @patch("whisper.load_model")
    def test_transcribe_audio_json(self, mock_load_model):
        """Test transcription with JSON output"""
        # Create mock model and result
        mock_model = MagicMock()
        mock_result = {
            "text": "This is a test transcription.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.0,
                    "text": "This is a test transcription.",
                }
            ],
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Call the function
        transcribe_audio(
            file_path=self.test_audio_file,
            model_name="tiny",
            output_dir=self.output_dir,
            output_formats=["json"],
            verbose=False,
        )

        # Check the output file was created
        output_json = os.path.join(self.output_dir, "test_audio.json")
        self.assertTrue(os.path.exists(output_json))

        # Check the content of the output file
        with open(output_json, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertEqual(content["text"], "This is a test transcription.")
            self.assertEqual(len(content["segments"]), 1)

    @patch("whisper.load_model")
    @patch("whisper.utils.get_writer")
    def test_transcribe_audio_srt(self, mock_get_writer, mock_load_model):
        """Test transcription with SRT output"""
        # Create mock model and writer
        mock_model = MagicMock()
        mock_result = {
            "text": "This is a test transcription.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.0,
                    "text": "This is a test transcription.",
                }
            ],
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        mock_srt_writer = MagicMock()
        mock_get_writer.return_value = mock_srt_writer

        # Call the function
        transcribe_audio(
            file_path=self.test_audio_file,
            model_name="tiny",
            output_dir=self.output_dir,
            output_formats=["srt"],
            verbose=False,
        )

        # Check that the writer was created and called correctly
        mock_get_writer.assert_called_with("srt", self.output_dir)
        mock_srt_writer.assert_called_once_with(mock_result, self.test_audio_file)

    @patch("whisper.load_model")
    def test_transcribe_with_language(self, mock_load_model):
        """Test transcription with specific language setting"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Test with language"}
        mock_load_model.return_value = mock_model

        # Call the function with language option
        transcribe_audio(
            file_path=self.test_audio_file,
            model_name="tiny",
            language="en",
            output_dir=self.output_dir,
            output_formats=["txt"],
            verbose=False,
        )

        # Check that the model was called with language parameter
        mock_model.transcribe.assert_called_once()
        args, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("language"), "en")

    @patch("whisper.load_model")
    def test_transcribe_with_word_timestamps(self, mock_load_model):
        """Test transcription with word timestamps option"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Test with word timestamps"}
        mock_load_model.return_value = mock_model

        # Call the function with word_timestamps option
        transcribe_audio(
            file_path=self.test_audio_file,
            model_name="tiny",
            output_dir=self.output_dir,
            output_formats=["txt"],
            word_timestamps=True,
            verbose=False,
        )

        # Check that the model was called with word_timestamps parameter
        mock_model.transcribe.assert_called_once()
        args, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("word_timestamps"), True)


if __name__ == "__main__":
    unittest.main()
