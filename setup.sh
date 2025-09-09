#!/bin/bash
# Audio Transcription Tool - Setup Script
# Optimized for GTX 1080 GPU with CUDA 11.8

set -e

echo "=========================================="
echo "Audio Transcription Tool Setup"
echo "=========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -e "\n1. Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $PYTHON_VERSION found"
    
    # Check if version is 3.8+
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        echo "✗ Python 3.8+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "✗ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check CUDA installation
echo -e "\n2. Checking CUDA installation..."
if command_exists nvidia-smi; then
    echo "✓ NVIDIA driver found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo "✓ CUDA $CUDA_VERSION found"
    else
        echo "⚠ CUDA toolkit not found. GPU acceleration may not work."
        echo "  Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive"
    fi
else
    echo "⚠ NVIDIA driver not found. Will use CPU for inference."
fi

# Check FFmpeg installation
echo -e "\n3. Checking FFmpeg installation..."
if command_exists ffmpeg; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo "✓ FFmpeg $FFMPEG_VERSION found"
else
    echo "✗ FFmpeg not found. Installing..."
    
    if [ "$OS" == "linux" ]; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif [ "$OS" == "macos" ]; then
        if command_exists brew; then
            brew install ffmpeg
        else
            echo "Please install Homebrew first: https://brew.sh"
            exit 1
        fi
    elif [ "$OS" == "windows" ]; then
        echo "Please download FFmpeg from: https://ffmpeg.org/download.html"
        echo "Add it to your PATH environment variable"
        exit 1
    fi
fi

# Create virtual environment
echo -e "\n4. Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activating..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ "$OS" == "windows" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo -e "\n5. Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo -e "\n6. Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo -e "\n7. Installing other dependencies..."
pip install faster-whisper==1.0.3
pip install gradio==4.19.2
pip install PyDrive2==1.19.0
pip install ffmpeg-python==0.2.0
pip install numpy==1.24.3
pip install tqdm==4.66.1

# Create necessary directories
echo -e "\n8. Creating necessary directories..."
mkdir -p models
mkdir -p downloads
mkdir -p outputs
echo "✓ Directories created"

# Create Google Drive credentials template
echo -e "\n9. Setting up Google Drive credentials..."
if [ ! -f "client_secrets.json" ]; then
    cat > client_secrets.json << 'EOF'
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "YOUR_PROJECT_ID",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost:8080/"]
  }
}
EOF
    echo "✓ Created client_secrets.json template"
    echo "  ⚠ Please update with your Google Drive API credentials"
    echo "  Get credentials from: https://console.cloud.google.com/"
else
    echo "✓ client_secrets.json already exists"
fi

# Create .env file for configuration
echo -e "\n10. Creating configuration file..."
cat > .env << 'EOF'
# Model Configuration
WHISPER_MODEL_SIZE=base
DEVICE=cuda
COMPUTE_TYPE=float16

# Paths
MODEL_CACHE_DIR=./models
DOWNLOAD_DIR=./downloads
OUTPUT_DIR=./outputs

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=7860
SHARE_GRADIO=true

# Processing Configuration
DEFAULT_LANGUAGE=auto
BATCH_SIZE=1
NUM_WORKERS=2
EOF
echo "✓ Configuration file created"

# Test installation
echo -e "\n11. Testing installation..."
python3 -c "
import torch
import faster_whisper
import gradio
import ffmpeg
print('✓ All imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update client_secrets.json with your Google Drive API credentials"
echo "2. Run the application: python3 audio_transcription_tool.py"
echo "3. Access the interface at: http://localhost:7860"
echo ""
echo "For Google Drive setup:"
echo "- Go to https://console.cloud.google.com/"
echo "- Create a new project or select existing"
echo "- Enable Google Drive API"
echo "- Create OAuth 2.0 credentials"
echo "- Download and replace client_secrets.json"
echo ""
echo "Troubleshooting:"
echo "- If CUDA is not detected, ensure CUDA 11.8 is installed"
echo "- For GTX 1080, use 'base' or 'small' model size"
echo "- Check GPU memory with: nvidia-smi"
echo ""