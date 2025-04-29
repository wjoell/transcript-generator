#!/usr/bin/env python3
"""
Tests for the validation functions in whisper_transcribe.py
"""

import os
import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whisper_transcribe import validate_file, validate_output_dir, validate_model


class TestValidators(unittest.TestCase):
    """Test cases for validation functions"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create a test audio file
        self.test_audio_file = os.path.join(self.test_dir, "test_audio.mp3")
        with open(self.test_audio_file, "wb") as f:
            f.write(b"dummy audio content")

        # Create a test non-audio file
        self.test_text_file = os.path.join(self.test_dir, "test_file.txt")
        with open(self.test_text_file, "w") as f:
            f.write("dummy text content")

        # Create a directory that we'll use for permission tests
        self.test_output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.test_output_dir)

    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)

    def test_validate_file_exists(self):
        """Test validation of an existing file"""
        self.assertTrue(validate_file(self.test_audio_file, verbose=False))

    def test_validate_file_not_exists(self):
        """Test validation of a non-existent file"""
        non_existent_file = os.path.join(self.test_dir, "non_existent.mp3")
        self.assertFalse(validate_file(non_existent_file, verbose=False))

    def test_validate_file_is_directory(self):
        """Test validation when file path is a directory"""
        self.assertFalse(validate_file(self.test_dir, verbose=False))

    def test_validate_file_extension(self):
        """Test file extension validation warnings (should still return True)"""
        # This should return True but print a warning (not tested here)
        self.assertTrue(validate_file(self.test_text_file, verbose=False))

    def test_validate_output_dir_exists(self):
        """Test validation of an existing output directory"""
        self.assertTrue(validate_output_dir(self.test_output_dir, verbose=False))

    def test_validate_output_dir_creation(self):
        """Test creation of output directory if it doesn't exist"""
        new_dir = os.path.join(self.test_dir, "new_output")
        self.assertTrue(validate_output_dir(new_dir, verbose=False))
        # Verify directory was created
        self.assertTrue(os.path.exists(new_dir))

    def test_validate_output_dir_is_file(self):
        """Test validation when output dir is a file"""
        self.assertFalse(validate_output_dir(self.test_audio_file, verbose=False))

    def test_validate_model(self):
        """Test model validation"""
        # Valid models
        self.assertTrue(validate_model("tiny", verbose=False))
        self.assertTrue(validate_model("base", verbose=False))
        self.assertTrue(validate_model("small", verbose=False))
        self.assertTrue(validate_model("medium", verbose=False))
        self.assertTrue(validate_model("large", verbose=False))
        self.assertTrue(validate_model("large-v1", verbose=False))
        self.assertTrue(validate_model("large-v2", verbose=False))

        # Invalid model
        self.assertFalse(validate_model("nonexistent-model", verbose=False))


if __name__ == "__main__":
    unittest.main()
