"""ffmpeg-based video ingestion: extract frames and metadata."""
from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from config import FRAMES_DIR, FRAME_SAMPLE_RATE, DARKNESS_THRESHOLD


@dataclass
class VideoMetadata:
    path: str
    duration: float       # seconds
    width: int
    height: int
    fps: float
    total_frames: int


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install with: brew install ffmpeg"
        )


def get_video_metadata(video_path: str) -> VideoMetadata:
    _require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None
    )
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    duration = float(info["format"].get("duration", 0))

    # Parse fps from r_frame_rate (e.g. "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den else 30.0

    total_frames = int(video_stream.get("nb_frames", duration * fps))

    return VideoMetadata(
        path=video_path,
        duration=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        total_frames=total_frames,
    )


def extract_frames(
    video_path: str,
    output_dir: Path = FRAMES_DIR,
    sample_rate: float = FRAME_SAMPLE_RATE,
    force: bool = False,
) -> List[Tuple[Path, float]]:
    """
    Extract frames at `sample_rate` fps using ffmpeg.
    Returns list of (frame_path, timestamp_seconds) sorted by timestamp.
    Skips extraction if frames already exist and force=False.
    """
    _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and not force:
        return _frames_to_timestamped(existing)

    # Clear any old frames
    for f in output_dir.glob("frame_*.jpg"):
        f.unlink()

    # Extract at sample_rate fps, scale to 1280x720 for consistent HUD reading
    # Naming: frame_XXXXXXX.jpg where XXXXXXX is timestamp in milliseconds
    filter_chain = f"fps={sample_rate},scale=1280:720"
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", filter_chain,
        "-q:v", "3",   # JPEG quality (lower = better, 2-5 is good range)
        "-frame_pts", "1",
        str(output_dir / "frame_%07d.jpg"),
        "-y",
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return _frames_to_timestamped(frames)


def _frames_to_timestamped(frames: List[Path]) -> List[Tuple[Path, float]]:
    """
    Convert frame filenames back to (path, timestamp) tuples.
    ffmpeg names frames sequentially starting at 1, so frame N = (N-1)/sample_rate seconds.
    """
    result = []
    for frame in sorted(frames):
        # e.g. frame_0000001.jpg → index 1
        stem = frame.stem  # "frame_0000001"
        try:
            idx = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        timestamp = (idx - 1) / FRAME_SAMPLE_RATE
        result.append((frame, timestamp))
    return result


def is_dark_frame(frame_path: Path, threshold: int = DARKNESS_THRESHOLD) -> bool:
    """
    Check if a frame is predominantly dark (loading screen, transition).
    Uses PIL for a fast mean brightness check without importing cv2 everywhere.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(frame_path).convert("L")  # grayscale
        arr = np.array(img)
        return float(arr.mean()) < threshold
    except Exception:
        return False
