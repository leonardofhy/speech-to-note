#!/usr/bin/env python3
"""
Audio Transcription Tool with Google Drive Integration
Optimized for GTX 1080 GPU with faster-whisper
"""

import os
import json
import logging
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import subprocess

import gradio as gr
import torch
import numpy as np
from faster_whisper import WhisperModel
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
# Note: Avoid importing ffmpeg-python due to frequent namespace conflicts with a different 'ffmpeg' package.
"""
We use the system ffmpeg/ffprobe via subprocess for robust audio conversion and probing.
Ensure FFmpeg is installed and available in PATH. See test_installation.py for checks.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
drive = None
whisper_model = None
device = None


class TranscriptionEngine:
    """Handles audio transcription using faster-whisper"""

    def __init__(self, model_size="base", device="cuda", compute_type="float16"):
        """Initialize the transcription engine"""
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the Whisper model with GPU optimization"""
        try:
            # Check CUDA availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.compute_type = "float32"

            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")

            # Try different compute types for GTX 1080 compatibility
            compute_types_to_try = []
            if self.device == "cuda":
                # GTX 1080 (Pascal) may have issues with float16
                compute_types_to_try = ["float16", "int8", "float32"]
            else:
                compute_types_to_try = ["float32"]

            for compute_type in compute_types_to_try:
                try:
                    logger.info(f"Trying compute type: {compute_type}")
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=compute_type,
                        num_workers=2,
                        download_root="./models",
                    )
                    self.compute_type = compute_type
                    logger.info(f"Model loaded successfully with {compute_type}")
                    break
                except Exception as e:
                    logger.warning(f"Failed with {compute_type}: {str(e)}")
                    continue
            else:
                raise Exception("Failed to load model with any compute type")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def transcribe(
        self,
        audio_path,
        language=None,
        task="transcribe",
        word_timestamps=False,
        progress_callback=None,
    ):
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            task: 'transcribe' or 'translate'
            word_timestamps: Include word-level timestamps
            progress_callback: Function to call with progress updates

        Returns:
            Dict with transcription results
        """
        try:
            logger.info(f"Transcribing: {audio_path}")

            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=5,
                best_of=5,
                patience=1,
                length_penalty=1,
                temperature=0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=word_timestamps,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=2000,
                    window_size_samples=1024,
                    speech_pad_ms=400,
                ),
            )

            # Process results
            result = {
                "text": "",
                "segments": [],
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
            }

            # Collect segments
            for segment in segments:
                seg_dict = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }

                if word_timestamps and segment.words:
                    seg_dict["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ]

                result["segments"].append(seg_dict)
                result["text"] += segment.text + " "

                # Update progress
                if progress_callback:
                    progress = (segment.end / info.duration) * 100
                    progress_callback(progress, f"Processing: {progress:.1f}%")

            result["text"] = result["text"].strip()

            logger.info(f"Transcription completed. Language: {result['language']}")
            return result

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise


class GoogleDriveManager:
    """Manages Google Drive operations"""

    def __init__(self):
        """Initialize Google Drive connection"""
        self.drive = None
        self.authenticated = False

    def authenticate(self):
        """Authenticate with Google Drive"""
        try:
            gauth = GoogleAuth()

            # Try to load saved credentials
            gauth.LoadCredentialsFile("credentials.json")

            if gauth.credentials is None:
                # Authenticate if credentials don't exist
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                # Refresh if expired
                gauth.Refresh()
            else:
                # Authorize
                gauth.Authorize()

            # Save credentials for next run
            gauth.SaveCredentialsFile("credentials.json")

            self.drive = GoogleDrive(gauth)
            self.authenticated = True

            logger.info("Google Drive authentication successful")
            return True

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False

    def list_audio_files(self, folder_id=None):
        """List audio files from Google Drive"""
        if not self.authenticated:
            return []

        try:
            # Build query for audio files
            audio_mimes = [
                "audio/mpeg",
                "audio/mp3",
                "audio/wav",
                "audio/x-wav",
                "audio/mp4",
                "audio/x-m4a",
                "audio/ogg",
                "audio/flac",
                "audio/aac",
                "audio/webm",
            ]

            mime_query = " or ".join([f"mimeType='{mime}'" for mime in audio_mimes])

            if folder_id:
                query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"
            else:
                query = f"({mime_query}) and trashed=false"

            file_list = self.drive.ListFile({"q": query}).GetList()

            # Format file information
            files = []
            for file in file_list:
                files.append(
                    {
                        "id": file["id"],
                        "title": file["title"],
                        "size": int(file.get("fileSize", 0)),
                        "mimeType": file["mimeType"],
                        "createdDate": file.get("createdDate"),
                        "modifiedDate": file.get("modifiedDate"),
                    }
                )

            logger.info(f"Found {len(files)} audio files")
            return files

        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []

    def download_file(self, file_id, destination):
        """Download file from Google Drive"""
        if not self.authenticated:
            return None

        try:
            file = self.drive.CreateFile({"id": file_id})
            file.GetContentFile(destination)
            logger.info(f"Downloaded file to: {destination}")
            return destination

        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return None


class AudioProcessor:
    """Handles audio format conversion and processing"""

    @staticmethod
    def convert_audio(input_path, output_path, sample_rate=16000):
        """Convert audio to 16 kHz mono PCM WAV using ffmpeg CLI"""
        try:
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vn",  # no video
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                output_path,
            ]
            subprocess.run(cmd, check=True)

            # Basic sanity check
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FFmpeg produced empty output")

            logger.info(f"Converted audio to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio conversion error: {str(e)}")
            return None

    @staticmethod
    def get_audio_info(file_path):
        """Get audio file information using ffprobe CLI"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                file_path,
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            info = json.loads(result.stdout)

            audio_stream = next(
                (
                    stream
                    for stream in info.get("streams", [])
                    if stream.get("codec_type") == "audio"
                ),
                None,
            )

            if audio_stream:
                return {
                    "duration": float(info.get("format", {}).get("duration", 0) or 0),
                    "bitrate": int(info.get("format", {}).get("bit_rate", 0) or 0),
                    "sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
                    "channels": int(audio_stream.get("channels", 0) or 0),
                    "codec": audio_stream.get("codec_name", "unknown"),
                }

            return None

        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return None


class OutputFormatter:
    """Formats transcription output in various formats"""

    @staticmethod
    def to_text(result):
        """Convert to plain text"""
        return result["text"]

    @staticmethod
    def to_json(result):
        """Convert to JSON format"""
        return json.dumps(result, indent=2, ensure_ascii=False)

    @staticmethod
    def to_srt(result):
        """Convert to SRT subtitle format"""
        srt_content = []

        for i, segment in enumerate(result["segments"], 1):
            start_time = OutputFormatter._seconds_to_srt_time(segment["start"])
            end_time = OutputFormatter._seconds_to_srt_time(segment["end"])

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment["text"].strip())
            srt_content.append("")

        return "\n".join(srt_content)

    @staticmethod
    def to_vtt(result):
        """Convert to WebVTT subtitle format"""
        vtt_content = ["WEBVTT", ""]

        for segment in result["segments"]:
            start_time = OutputFormatter._seconds_to_vtt_time(segment["start"])
            end_time = OutputFormatter._seconds_to_vtt_time(segment["end"])

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(segment["text"].strip())
            vtt_content.append("")

        return "\n".join(vtt_content)

    @staticmethod
    def _seconds_to_srt_time(seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @staticmethod
    def _seconds_to_vtt_time(seconds):
        """Convert seconds to WebVTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


# Gradio Interface Functions


def init_app():
    """Initialize the application"""
    global drive, whisper_model, device

    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        compute_type = "float32"
        logger.info("CUDA not available. Using CPU")

    # Initialize transcription engine
    try:
        whisper_model = TranscriptionEngine(
            model_size="base", device=device, compute_type=compute_type
        )
    except Exception as e:
        logger.error(f"Failed to initialize transcription engine: {str(e)}")
        return False

    # Initialize Google Drive
    drive = GoogleDriveManager()

    return True


def authenticate_drive():
    """Authenticate with Google Drive"""
    global drive

    if drive.authenticate():
        return "SUCCESS Google Drive authenticated successfully!", gr.update(
            visible=True
        )
    else:
        return "ERROR Authentication failed. Please check credentials.", gr.update(
            visible=False
        )


def list_drive_files():
    """List audio files from Google Drive"""
    global drive

    if not drive.authenticated:
        return "Please authenticate with Google Drive first."

    files = drive.list_audio_files()

    if not files:
        return "No audio files found in Google Drive."

    # Format file list for display
    file_list = []
    for file in files:
        size_mb = file["size"] / (1024 * 1024)
        file_list.append(f"- {file['title']} ({size_mb:.2f} MB) - ID: {file['id']}")

    return "\n".join(file_list)


def transcribe_file(
    file_input,
    drive_file_id,
    language,
    task,
    word_timestamps,
    output_format,
    progress=gr.Progress(),
):
    """Transcribe audio file"""
    global whisper_model, drive

    temp_files = []

    try:
        # Determine input source
        if file_input is not None:
            # Local file upload
            audio_path = file_input
            logger.info(f"Using uploaded file: {audio_path}")

        elif drive_file_id:
            # Google Drive file
            if not drive.authenticated:
                return None, "Please authenticate with Google Drive first."

            # Download from Drive
            temp_audio = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
            temp_files.append(temp_audio.name)

            progress(0.1, "Downloading from Google Drive...")
            audio_path = drive.download_file(drive_file_id.strip(), temp_audio.name)

            if not audio_path:
                return None, "Failed to download file from Google Drive."
        else:
            return None, "Please provide an audio file or Google Drive file ID."

        # Convert audio if needed
        progress(0.2, "Processing audio...")
        converted_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        temp_files.append(converted_path)

        converted = AudioProcessor.convert_audio(audio_path, converted_path)
        if not converted:
            return None, "ERROR Failed to convert audio. Ensure FFmpeg is installed and the input is a valid audio file."

    # Get audio info
    audio_info = AudioProcessor.get_audio_info(converted_path)

        # Transcribe
        progress(0.3, "Starting transcription...")

        def update_progress(percent, message):
            progress(0.3 + (percent * 0.6 / 100), message)

        result = whisper_model.transcribe(
            converted_path,
            language=language if language != "auto" else None,
            task=task,
            word_timestamps=word_timestamps,
            progress_callback=update_progress,
        )

        # Format output
        progress(0.9, "Formatting output...")

        if output_format == "txt":
            output_content = OutputFormatter.to_text(result)
            file_extension = ".txt"
        elif output_format == "json":
            output_content = OutputFormatter.to_json(result)
            file_extension = ".json"
        elif output_format == "srt":
            output_content = OutputFormatter.to_srt(result)
            file_extension = ".srt"
        elif output_format == "vtt":
            output_content = OutputFormatter.to_vtt(result)
            file_extension = ".vtt"
        else:
            output_content = OutputFormatter.to_text(result)
            file_extension = ".txt"

        # Save output
        output_filename = (
            f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
        )
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)

        # Prepare summary
        summary = f"""
        SUCCESS Transcription Complete!
        
        Text Length: {len(result['text'])} characters
        Detected Language: {result['language']} (confidence: {result['language_probability']:.2%})
        Duration: {result['duration']:.1f} seconds
        Segments: {len(result['segments'])}
        Output Format: {output_format.upper()}
        """

        progress(1.0, "Complete!")

        return output_path, summary

    except Exception as e:
        error_msg = (
            f"ERROR Error during transcription: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        return None, error_msg

    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass


def batch_transcribe(file_list, output_format, progress=gr.Progress()):
    """Batch transcribe multiple files"""
    global whisper_model

    if not file_list:
        return None, "No files provided for batch processing."

    results = []
    output_files = []

    for i, file in enumerate(file_list):
        progress((i / len(file_list)), f"Processing file {i+1}/{len(file_list)}")

        try:
            # Process each file
            output_path, summary = transcribe_file(
                file, None, "auto", "transcribe", False, output_format
            )

            if output_path:
                output_files.append(output_path)
                results.append(f"SUCCESS {os.path.basename(file)}: Success")
            else:
                results.append(f"ERROR {os.path.basename(file)}: Failed")

        except Exception as e:
            results.append(f"ERROR {os.path.basename(file)}: {str(e)}")

    # Create summary
    summary = "Batch Processing Complete!\n\n" + "\n".join(results)

    # Create zip file with all outputs
    if output_files:
        import zipfile

        zip_path = os.path.join(
            tempfile.gettempdir(),
            f"batch_transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        )

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in output_files:
                zipf.write(file, os.path.basename(file))

        return zip_path, summary

    return None, summary


# Create Gradio interface
def create_interface():
    """Create the Gradio web interface"""

    with gr.Blocks(title="Audio Transcription Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
        # Audio Transcription Tool
        ### Powered by Whisper AI | GPU Accelerated | Google Drive Integration
        """
        )

        with gr.Tabs():
            # Single File Tab
            with gr.TabItem("Single File Transcription"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Options")

                        file_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )

                        gr.Markdown("**OR**")

                        with gr.Group():
                            auth_btn = gr.Button(
                                "Authenticate Google Drive", variant="secondary"
                            )
                            auth_status = gr.Textbox(
                                label="Authentication Status", interactive=False
                            )

                            with gr.Column(visible=False) as drive_section:
                                list_btn = gr.Button(
                                    "List Drive Files", variant="secondary"
                                )
                                drive_files = gr.Textbox(
                                    label="Available Files", lines=5, interactive=False
                                )
                                drive_file_id = gr.Textbox(
                                    label="Google Drive File ID",
                                    placeholder="Enter file ID from the list above",
                                )

                    with gr.Column(scale=1):
                        gr.Markdown("### Transcription Settings")

                        language = gr.Dropdown(
                            label="Language",
                            choices=["auto"]
                            + [
                                "en",
                                "es",
                                "fr",
                                "de",
                                "it",
                                "pt",
                                "ru",
                                "ja",
                                "ko",
                                "zh",
                            ],
                            value="auto",
                        )

                        task = gr.Radio(
                            label="Task",
                            choices=["transcribe", "translate"],
                            value="transcribe",
                        )

                        word_timestamps = gr.Checkbox(
                            label="Include Word Timestamps", value=False
                        )

                        output_format = gr.Radio(
                            label="Output Format",
                            choices=["txt", "json", "srt", "vtt"],
                            value="txt",
                        )

                        transcribe_btn = gr.Button(
                            "Start Transcription", variant="primary", size="lg"
                        )

                with gr.Row():
                    output_file = gr.File(label="Download Transcription")
                    output_summary = gr.Textbox(label="Summary", lines=8)

            # Batch Processing Tab
            with gr.TabItem("Batch Processing"):
                gr.Markdown("### Batch Transcribe Multiple Files")

                batch_input = gr.File(
                    label="Upload Multiple Audio Files",
                    file_count="multiple",
                    file_types=["audio"],
                )

                batch_format = gr.Radio(
                    label="Output Format",
                    choices=["txt", "json", "srt", "vtt"],
                    value="txt",
                )

                batch_btn = gr.Button(
                    "Start Batch Processing", variant="primary", size="lg"
                )

                batch_output = gr.File(label="Download All Transcriptions (ZIP)")
                batch_summary = gr.Textbox(label="Batch Summary", lines=10)

            # Settings Tab
            with gr.TabItem("Settings & Info"):
                gr.Markdown(
                    """
                ### System Information
                """
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            f"""
                        **Hardware:**
                        - Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
                        - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB if CUDA available
                        
                        **Model:**
                        - Whisper Base (optimal for GTX 1080)
                        - faster-whisper implementation
                        - INT8 quantization available
                        """
                        )

                    with gr.Column():
                        gr.Markdown(
                            """
                        **Supported Formats:**
                        - Audio: MP3, WAV, M4A, OGG, FLAC, AAC
                        - Output: TXT, JSON, SRT, VTT
                        
                        **Features:**
                        - 99+ language support
                        - Auto language detection
                        - Word-level timestamps
                        - GPU acceleration
                        """
                        )

        # Event handlers
        auth_btn.click(authenticate_drive, outputs=[auth_status, drive_section])

        list_btn.click(list_drive_files, outputs=drive_files)

        transcribe_btn.click(
            transcribe_file,
            inputs=[
                file_input,
                drive_file_id,
                language,
                task,
                word_timestamps,
                output_format,
            ],
            outputs=[output_file, output_summary],
        )

        batch_btn.click(
            batch_transcribe,
            inputs=[batch_input, batch_format],
            outputs=[batch_output, batch_summary],
        )

    return app


# Main execution
if __name__ == "__main__":
    # Initialize application
    if init_app():
        # Create and launch interface
        app = create_interface()
        app.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
    else:
        logger.error("Failed to initialize application")
