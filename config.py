from pathlib import Path

# Models — Bedrock cross-region inference profile IDs
VISION_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
COACHING_MODEL = "global.anthropic.claude-sonnet-4-6"

# Frame sampling — 0.1fps = 1 frame every 10 seconds
# A 3-hour VOD produces ~1080 frames ≈ $4-6 in API cost
FRAME_SAMPLE_RATE = 0.1  # frames per second

# Skip frames below this darkness threshold (loading screens, black screens)
# Value 0-255: mean pixel brightness. Below this = skip.
DARKNESS_THRESHOLD = 30

# Consecutive dark frames needed to mark a "game boundary" candidate
DARK_FRAME_SEQUENCE_MIN = 3

# Output paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"

# Intermediate JSON files (allow resuming a run)
FRAME_ANALYSES_FILE = DATA_DIR / "frame_analyses.json"
TIMELINE_FILE = DATA_DIR / "timeline.json"
COACHING_FILE = DATA_DIR / "coaching.json"
REPORT_FILE = DATA_DIR / "report.md"

# Minimum frames to constitute a "real game" (filters out menu/lobby clips)
MIN_FRAMES_PER_GAME = 30  # 5 minutes at 0.1fps

# Max tokens for coaching responses
COACHING_MAX_TOKENS = 8000
FRAME_ANALYSIS_MAX_TOKENS = 512

# Ensure data dirs exist at import time
DATA_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
