#!/usr/bin/env python3
"""
Tests for the environment validation in whisper_transcribe.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whisper_transcribe import validate_environment


class TestEnvironmentValidation(unittest.TestCase):
    """Test cases for environment validation"""

    @patch("torch.__version__", "2.0.0")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("shutil.which")
    def test_environment_valid(
        self, mock_which, mock_get_device_name, mock_cuda_available
    ):
        """Test when environment is valid"""
        # Set up mocks
        mock_cuda_available.return_value = True
        mock_get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_which.return_value = "/usr/bin/ffmpeg"  # ffmpeg is found

        # Call the function
        result = validate_environment(verbose=False)

        # Check result
        self.assertTrue(result)
        mock_which.assert_called_once_with("ffmpeg")

    @patch("torch.__version__", "2.0.0")
    @patch("torch.cuda.is_available")
    @patch("shutil.which")
    def test_environment_no_cuda(self, mock_which, mock_cuda_available):
        """Test when CUDA is not available"""
        # Set up mocks
        mock_cuda_available.return_value = False
        mock_which.return_value = "/usr/bin/ffmpeg"  # ffmpeg is found

        # Call the function
        result = validate_environment(verbose=False)

        # Check result - should still be valid even without CUDA
        self.assertTrue(result)

    @patch("torch.__version__", "2.0.0")
    @patch("torch.cuda.is_available")
    @patch("shutil.which")
    def test_environment_no_ffmpeg(self, mock_which, mock_cuda_available):
        """Test when ffmpeg is not installed"""
        # Set up mocks
        mock_cuda_available.return_value = True
        mock_which.return_value = None  # ffmpeg not found

        # Call the function
        result = validate_environment(verbose=False)

        # Check result - should be invalid without ffmpeg
        self.assertFalse(result)

    @patch("builtins.__import__")
    def test_environment_no_torch(self, mock_import):
        """Test when torch is not installed"""

        # Set up mock to raise ImportError when torch is imported
        def import_mock(name, *args):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return MagicMock()

        mock_import.side_effect = import_mock

        # Call the function
        with patch.dict("sys.modules", {"torch": None}):
            result = validate_environment(verbose=False)

        # Check result - should be invalid without torch
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
