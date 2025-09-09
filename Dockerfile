# Audio Transcription Tool Docker Image
# Optimized for NVIDIA GTX 1080 GPU

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip3 install \
    faster-whisper==1.0.3 \
    gradio==4.19.2 \
    PyDrive2==1.19.0 \
    ffmpeg-python==0.2.0 \
    numpy==1.24.3 \
    tqdm==4.66.1

# Copy application files
COPY audio_transcription_tool.py .
COPY .env .

# Create necessary directories
RUN mkdir -p models downloads outputs

# Download Whisper model during build (optional, for faster startup)
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', download_root='./models')"

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python3", "audio_transcription_tool.py"]