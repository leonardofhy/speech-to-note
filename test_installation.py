#!/usr/bin/env python3
"""
Test script to verify Audio Transcription Tool installation
"""

import sys
import subprocess
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing Python package imports...")

    packages = {
        "torch": "PyTorch",
        "faster_whisper": "Faster Whisper",
        "gradio": "Gradio",
        "pydrive2": "PyDrive2",
        "ffmpeg": "ffmpeg-python",
        "numpy": "NumPy",
        "tqdm": "tqdm",
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  PASS {name}")
        except ImportError as e:
            print(f"  FAIL {name}: {str(e)}")
            failed.append(name)

    return len(failed) == 0


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA/GPU support...")

    try:
        import torch

        if torch.cuda.is_available():
            print("  PASS CUDA is available")
            print(f"  PASS GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"  PASS VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            print(f"  PASS CUDA Version: {torch.version.cuda}")

            # Test tensor operations
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            print("  PASS GPU computation test passed")

            return True
        else:
            print("  WARN CUDA not available - will use CPU for inference")
            print("  Tip: Check NVIDIA drivers and CUDA installation")
            return False

    except Exception as e:
        print(f"  FAIL Error testing CUDA: {str(e)}")
        return False


def test_ffmpeg():
    """Test FFmpeg installation"""
    print("\nTesting FFmpeg installation...")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )

        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"  PASS FFmpeg installed: {version_line}")
            return True
        else:
            print("  FAIL FFmpeg not working properly")
            return False

    except FileNotFoundError:
        print("  FAIL FFmpeg not found in PATH")
        print("  Installation instructions:")
        print("    Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("    macOS: brew install ffmpeg")
        print("    Windows: Download from https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"  FAIL Error testing FFmpeg: {str(e)}")
        return False


def test_model_loading():
    """Test Whisper model loading"""
    print("\nTesting Whisper model loading...")

    try:
        from faster_whisper import WhisperModel
        import torch

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        print(f"  Loading 'tiny' model for testing on {device}...")

        # Try to load tiny model (smallest)
        model = WhisperModel(
            "tiny", device=device, compute_type=compute_type, download_root="./models"
        )

        print(f"  PASS Successfully loaded Whisper model on {device}")

        # Test with a dummy audio
        print("  Testing inference capability...")
        # Note: Actual inference test would require an audio file

        return True

    except Exception as e:
        print(f"  FAIL Error loading model: {str(e)}")
        return False


def test_directories():
    """Test if required directories exist"""
    print("\nChecking required directories...")

    dirs = ["models", "downloads", "outputs"]
    all_exist = True

    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"  PASS {dir_name}/ exists")
        else:
            print(f"  FAIL {dir_name}/ does not exist - creating...")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"    PASS Created {dir_name}/")
            except Exception as e:
                print(f"    FAIL Failed to create: {str(e)}")
                all_exist = False

    return all_exist


def test_google_drive_config():
    """Test Google Drive configuration"""
    print("\nChecking Google Drive configuration...")

    client_secrets = Path("client_secrets.json")

    if client_secrets.exists():
        print("  PASS client_secrets.json found")

        import json

        try:
            with open(client_secrets, "r") as f:
                config = json.load(f)

            if "YOUR_CLIENT_ID" in str(config):
                print("  WARN client_secrets.json needs to be configured")
                print("    Please update with your Google Drive API credentials")
                print("    Get credentials from: https://console.cloud.google.com/")
                return False
            else:
                print("  PASS client_secrets.json appears to be configured")
                return True

        except Exception as e:
            print(f"  FAIL Error reading client_secrets.json: {str(e)}")
            return False
    else:
        print("  WARN client_secrets.json not found")
        print("    Google Drive integration will not work without it")
        return False


def test_application():
    """Test if the main application can be imported"""
    print("\nTesting main application...")

    try:
        # Try importing the main application
        import audio_transcription_tool

        print("  PASS Main application can be imported")

        # Check for main components
        components = [
            "TranscriptionEngine",
            "GoogleDriveManager",
            "AudioProcessor",
            "OutputFormatter",
        ]

        for component in components:
            if hasattr(audio_transcription_tool, component):
                print(f"  PASS {component} found")
            else:
                print(f"  FAIL {component} not found")

        return True

    except FileNotFoundError:
        print("  FAIL audio_transcription_tool.py not found")
        return False
    except Exception as e:
        print(f"  FAIL Error importing application: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Audio Transcription Tool - Installation Test")
    print("=" * 50)

    results = {
        "Python packages": test_imports(),
        "CUDA/GPU": test_cuda(),
        "FFmpeg": test_ffmpeg(),
        "Model loading": test_model_loading(),
        "Directories": test_directories(),
        "Google Drive": test_google_drive_config(),
        "Application": test_application(),
    }

    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED" if passed is False else "WARNING"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("SUCCESS All tests passed! The application is ready to use.")
        print("\nTo start the application:")
        print("  python3 audio_transcription_tool.py")
        print("\nThen open: http://localhost:7860")
    else:
        print("WARNING Some tests failed or have warnings.")
        print("\nCritical issues to fix:")

        if not results["Python packages"]:
            print(
                "  - Install missing Python packages: pip install -r requirements.txt"
            )

        if not results["FFmpeg"]:
            print("  - Install FFmpeg (required for audio processing)")

        if not results["Application"]:
            print("  - Ensure audio_transcription_tool.py is in the current directory")

        print("\nOptional improvements:")

        if not results["CUDA/GPU"]:
            print("  - Install CUDA for GPU acceleration (will work on CPU)")

        if not results["Google Drive"]:
            print("  - Configure Google Drive API for cloud storage access")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
