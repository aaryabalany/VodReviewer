#!/usr/bin/env bash
set -e

echo "=== VodReviewer Setup ==="

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
fi

echo ""
echo "Setup complete. Make sure AWS_BEARER_TOKEN_BEDROCK and AWS_REGION are set in your shell, then run:"
echo "  python main.py analyze <path_to_video>"
