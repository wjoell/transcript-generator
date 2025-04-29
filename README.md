# Whisper Transcriber

A local transcription tool powered by OpenAI's Whisper that allows you to transcribe audio recordings without sending your data to third-party services. Perfect for confidential meetings, interviews, and any audio content where privacy is a concern.

## Features

-   **Fully Local Processing**: All transcription happens on your machine
-   **Multiple Output Formats**: Export as TXT, SRT, VTT, or JSON
-   **Word-Level Timestamps**: Optional timestamps for each word
-   **Speaker Diarization**: Identify different speakers in conversations
-   **Validation Tools**: Dry-run feature to validate your setup before processing
-   **Clear Error Reporting**: Comprehensive error checking and reporting

## Prerequisites

-   Python 3.10 or newer (3.9 may work with manual dependency installation)
-   FFmpeg (required for audio processing)
-   Sufficient computational resources (GPU with 8GB+ VRAM recommended for diarization)

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

### With Speaker Diarization

```bash
./whisper_transcribe.py path/to/audio_file.mp3 --diarize
```

### Options

```
usage: whisper_transcribe.py [-h] [--model {tiny,base,small,medium,large,large-v1,large-v2,large-v3}]
                            [--language LANGUAGE] [--output-dir OUTPUT_DIR]
                            [--formats FORMATS] [--word-timestamps] [--quiet]
                            [--dry-run] [--diarize] [--num-speakers NUM_SPEAKERS]
                            [--min-speakers MIN_SPEAKERS] [--max-speakers MAX_SPEAKERS]
                            [--parallel-diarize]
                            file

Transcribe audio files using OpenAI's Whisper with optional speaker diarization

positional arguments:
  file                  Audio file to transcribe

options:
  -h, --help            show this help message and exit
  --model {tiny,base,small,medium,large,large-v1,large-v2,large-v3}
                        Model size to use (default: medium)
  --language LANGUAGE   Language code (e.g., 'en' for English)
  --output-dir OUTPUT_DIR
                        Directory to save output files
  --formats FORMATS     Comma-separated list of output formats (txt,json,srt,vtt)
  --word-timestamps     Include timestamps for each word
  --quiet               Suppress progress information
  --dry-run             Validate inputs without performing transcription

Speaker Diarization Options:
  --diarize             Enable speaker diarization to identify different speakers
  --num-speakers NUM_SPEAKERS
                        Exact number of speakers in the audio (if known)
  --min-speakers MIN_SPEAKERS
                        Minimum number of speakers to detect
  --max-speakers MAX_SPEAKERS
                        Maximum number of speakers to detect (use with --min-speakers)
  --parallel-diarize    Use parallel processing for diarization (requires more VRAM)
```

### Examples

**Basic usage with default settings**:

```bash
./whisper_transcribe.py meeting_recording.mp3
```

**Transcription with speaker diarization**:

```bash
./whisper_transcribe.py interview.wav --diarize
```

**Specifying the number of speakers for diarization**:

```bash
./whisper_transcribe.py podcast.mp3 --diarize --num-speakers 3
```

**Using a range for speaker detection**:

```bash
./whisper_transcribe.py conference.mp3 --diarize --min-speakers 2 --max-speakers 5
```

**Generate multiple output formats with diarization**:

```bash
./whisper_transcribe.py group_discussion.mp3 --diarize --formats txt,srt,json
```

**Using a larger model and parallel diarization for better quality**:

```bash
./whisper_transcribe.py important_meeting.wav --model large-v2 --diarize --parallel-diarize
```

**Validate setup without transcribing**:

```bash
./whisper_transcribe.py recording.mp3 --diarize --dry-run
```

## Speaker Diarization

The speaker diarization feature identifies different speakers in your audio files and labels them in the transcription output. This is especially useful for:

-   Meetings with multiple participants
-   Interviews
-   Panel discussions
-   Podcast episodes
-   Any multi-speaker content

### How Speaker Diarization Works

The diarization process works by:

1. Extracting vocals from the audio to improve speaker identification accuracy
2. Transcribing the audio using Whisper
3. Aligning and correcting timestamps
4. Performing voice activity detection to identify speech segments
5. Generating speaker embeddings to distinguish between different voices
6. Mapping the identified speakers to the transcription segments

### Hardware Requirements for Diarization

Speaker diarization is more computationally intensive than standard transcription:

-   A CUDA-compatible GPU with at least 8GB VRAM is recommended
-   For larger files or parallel processing, 10GB+ VRAM is preferred
-   CPU-only operation is possible but will be significantly slower

## Model Sizes

Whisper offers several model sizes with different accuracy levels and resource requirements:

| Model    | Parameters | Disk Space | Accuracy | Speed   |
| -------- | ---------- | ---------- | -------- | ------- |
| tiny     | 39M        | ~75MB      | Lowest   | Fastest |
| base     | 74M        | ~142MB     | Low      | Fast    |
| small    | 244M       | ~466MB     | Medium   | Medium  |
| medium   | 769M       | ~1.5GB     | High     | Slow    |
| large    | 1550M      | ~2.9GB     | Highest  | Slowest |
| large-v3 | 1550M      | ~3.1GB     | Best     | Slowest |

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

-   **GPU Acceleration**: Using a CUDA-compatible GPU can significantly speed up transcription and diarization
-   **Shorter Files**: Breaking longer recordings into smaller files can improve processing time
-   **Model Selection**: The "tiny" or "base" models are much faster but less accurate
-   **Audio Quality**: Clear audio with minimal background noise produces better results
-   **Parallel Processing**: Use `--parallel-diarize` for faster diarization if you have sufficient VRAM

## Troubleshooting

-   **FFmpeg Missing**: Ensure FFmpeg is installed and in your PATH
-   **CUDA Not Working**: Check PyTorch installation with proper CUDA support
-   **Memory Issues**: Try a smaller model size or processing shorter audio segments
-   **File Not Found**: Verify the file path and use quotes if the path contains spaces
-   **Diarization Dependencies**: If you encounter errors with diarization, ensure all dependencies are installed correctly

## Credits

This tool was created as a collaboration between a human developer and Claude AI (Anthropic). The code was generated with Claude's assistance, including comprehensive error handling and validation features.

The speaker diarization functionality is powered by [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization), which combines Whisper's transcription capabilities with speaker embedding technologies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
