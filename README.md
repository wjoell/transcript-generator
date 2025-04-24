# Whisper Transcriber

A local transcription tool powered by OpenAI's Whisper that allows you to transcribe audio recordings without sending your data to third-party services. Perfect for confidential meetings, interviews, and any audio content where privacy is a concern.

## Features

-   **Fully Local Processing**: All transcription happens on your machine
-   **Multiple Output Formats**: Export as TXT, SRT, VTT, or JSON
-   **Word-Level Timestamps**: Optional timestamps for each word
-   **Validation Tools**: Dry-run feature to validate your setup before processing
-   **Clear Error Reporting**: Comprehensive error checking and reporting

## Prerequisites

-   Python 3.8 or newer
-   FFmpeg (required for audio processing)
-   Sufficient computational resources (GPU recommended for faster processing)

## Installation

### 1. Install FFmpeg

**macOS**:

```bash
brew install ffmpeg
```

**Linux**:

```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows**:

```bash
# Using Chocolatey
choco install ffmpeg

# Or download from the official website
# https://ffmpeg.org/download.html
```

### 2. Set Up Environment with UV

```bash
# Create a virtual environment with Python 3.10
uv venv --python=3.10 .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Make the Script Executable (Unix/macOS)

```bash
chmod +x whisper_transcribe.py
```

## Usage

### Basic Transcription

```bash
./whisper_transcribe.py path/to/audio_file.mp3
```

### Options

```
usage: whisper_transcribe.py [-h] [--model {tiny,base,small,medium,large,large-v1,large-v2}]
                            [--language LANGUAGE] [--output-dir OUTPUT_DIR]
                            [--formats FORMATS] [--word-timestamps] [--quiet]
                            [--dry-run]
                            file

Transcribe audio files using OpenAI's Whisper

positional arguments:
  file                  Audio file to transcribe

options:
  -h, --help            show this help message and exit
  --model {tiny,base,small,medium,large,large-v1,large-v2}
                        Model size to use (default: medium)
  --language LANGUAGE   Language code (e.g., 'en' for English)
  --output-dir OUTPUT_DIR
                        Directory to save output files
  --formats FORMATS     Comma-separated list of output formats (txt,json,srt,vtt)
  --word-timestamps     Include timestamps for each word
  --quiet               Suppress progress information
  --dry-run             Validate inputs without performing transcription
```

### Examples

**Basic usage with default settings**:

```bash
./whisper_transcribe.py meeting_recording.mp3
```

**Specify model size and language**:

```bash
./whisper_transcribe.py interview.wav --model large --language en
```

**Generate multiple output formats**:

```bash
./whisper_transcribe.py podcast.mp3 --formats txt,srt,json
```

**Save outputs to specific directory**:

```bash
./whisper_transcribe.py lecture.m4a --output-dir transcripts/
```

**Validate setup without transcribing**:

```bash
./whisper_transcribe.py recording.mp3 --dry-run
```

## Model Sizes

Whisper offers several model sizes with different accuracy levels and resource requirements:

| Model  | Parameters | Disk Space | Accuracy | Speed   |
| ------ | ---------- | ---------- | -------- | ------- |
| tiny   | 39M        | ~75MB      | Lowest   | Fastest |
| base   | 74M        | ~142MB     | Low      | Fast    |
| small  | 244M       | ~466MB     | Medium   | Medium  |
| medium | 769M       | ~1.5GB     | High     | Slow    |
| large  | 1550M      | ~2.9GB     | Highest  | Slowest |

Choose the model size based on your accuracy needs and computational resources.

## Supported Audio Formats

-   MP3
-   MP4
-   WAV
-   M4A
-   FLAC
-   OGG
-   AAC
-   And most other common audio formats supported by FFmpeg

## Performance Tips

-   **GPU Acceleration**: Using a CUDA-compatible GPU can significantly speed up transcription
-   **Shorter Files**: Breaking longer recordings into smaller files can improve processing time
-   **Model Selection**: The "tiny" or "base" models are much faster but less accurate
-   **Audio Quality**: Clear audio with minimal background noise produces better results

## Troubleshooting

-   **FFmpeg Missing**: Ensure FFmpeg is installed and in your PATH
-   **CUDA Not Working**: Check PyTorch installation with proper CUDA support
-   **Memory Issues**: Try a smaller model size or processing shorter audio segments
-   **File Not Found**: Verify the file path and use quotes if the path contains spaces

## Credits

This tool was created as a collaboration between a human developer and Claude AI (Anthropic). The code was generated with Claude's assistance, including comprehensive error handling and validation features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
