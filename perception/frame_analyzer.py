"""Per-frame HUD reading via Claude Haiku Vision."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import List, Optional

import anthropic
from anthropic import AnthropicBedrock

from config import VISION_MODEL, FRAME_ANALYSIS_MAX_TOKENS, FRAME_ANALYSES_FILE
from game_state.models import FrameAnalysis, GamePhase
from ingestion.video_processor import is_dark_frame


FRAME_ANALYSIS_PROMPT = """Analyze this Fortnite gameplay screenshot. Read the HUD and return ONLY a JSON object — no markdown, no commentary.

Fortnite HUD locations:
- Bottom-center: health bar (red) and shield bar (blue) with white numbers
- Bottom-right corner: materials — Wood (brown), Brick (grey), Metal (dark), each showing a number
- Top-right area: kill feed showing recent player eliminations
- Top-center or top-right: storm timer and phase number
- Bottom-center: currently held weapon name
- Top-left or mini-map area: players remaining count

JSON schema to return:
{
  "game_phase": "<one of: loading|lobby|bus|skydiving|landing|early_game|mid_game|late_game|endgame|dead|spectating|unknown>",
  "health": <integer 0-100, or null if not visible>,
  "shield": <integer 0-100, or null if not visible>,
  "wood": <integer 0-999, or null if not visible>,
  "brick": <integer 0-999, or null if not visible>,
  "metal": <integer 0-999, or null if not visible>,
  "total_players_left": <integer, or null if not visible>,
  "is_in_combat": <true or false>,
  "weapon_equipped": "<weapon name string, or null>",
  "kill_feed": ["<each visible kill feed line as a string>"],
  "storm_phase": <integer 1-9, or null if not visible>,
  "notable_events": ["<brief description of anything significant happening>"],
  "scene_description": "<one sentence describing the scene>"
}

Rules:
- If a value is unclear or not visible, use null rather than guessing
- game_phase "dead" means the player is eliminated and watching a death cam or results screen
- game_phase "spectating" means the player is watching another player after their own death
- game_phase "endgame" means there are fewer than 10 players remaining
- Return ONLY the JSON object, nothing else"""


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def analyze_frame(
    frame_path: Path,
    timestamp: float,
    client: AnthropicBedrock,
    retries: int = 2,
) -> FrameAnalysis:
    """Send a single frame to Claude Haiku Vision and parse the response."""

    # Fast brightness check — skip dark/loading frames
    if is_dark_frame(frame_path):
        return FrameAnalysis(
            timestamp=timestamp,
            frame_path=str(frame_path),
            game_phase=GamePhase.LOADING,
            skipped=True,
        )

    image_data = _encode_image(frame_path)

    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=VISION_MODEL,
                max_tokens=FRAME_ANALYSIS_MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": FRAME_ANALYSIS_PROMPT},
                    ],
                }],
            )

            raw_text = response.content[0].text.strip()
            # Strip markdown code fences if the model wraps it anyway
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            data = json.loads(raw_text)

            return FrameAnalysis(
                timestamp=timestamp,
                frame_path=str(frame_path),
                game_phase=_safe_phase(data.get("game_phase")),
                health=_safe_int(data.get("health")),
                shield=_safe_int(data.get("shield")),
                wood=_safe_int(data.get("wood")),
                brick=_safe_int(data.get("brick")),
                metal=_safe_int(data.get("metal")),
                total_players_left=_safe_int(data.get("total_players_left")),
                is_in_combat=bool(data.get("is_in_combat", False)),
                weapon_equipped=data.get("weapon_equipped"),
                kill_feed=data.get("kill_feed") or [],
                storm_phase=_safe_int(data.get("storm_phase")),
                notable_events=data.get("notable_events") or [],
                scene_description=data.get("scene_description", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == retries:
                # Return a fallback rather than crashing the whole pipeline
                return FrameAnalysis(
                    timestamp=timestamp,
                    frame_path=str(frame_path),
                    game_phase=GamePhase.UNKNOWN,
                    scene_description="[parse error]",
                )
            time.sleep(1)

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            time.sleep(wait)

        except anthropic.APIError:
            if attempt == retries:
                raise
            time.sleep(5)

    # Should never reach here, but satisfy type checker
    return FrameAnalysis(timestamp=timestamp, frame_path=str(frame_path))


def analyze_frames_batch(
    frames: List[tuple[Path, float]],
    resume: bool = True,
    progress_callback=None,
) -> List[FrameAnalysis]:
    """
    Analyze all frames, saving progress to disk after each frame so runs can be resumed.

    frames: list of (frame_path, timestamp) from video_processor.extract_frames
    resume: if True and FRAME_ANALYSES_FILE exists, load previously analyzed frames
    progress_callback: optional callable(current, total) for progress display
    """
    client = AnthropicBedrock()

    # Load any previously completed analyses
    completed: dict[str, FrameAnalysis] = {}
    if resume and FRAME_ANALYSES_FILE.exists():
        with open(FRAME_ANALYSES_FILE) as f:
            for item in json.load(f):
                try:
                    fa = FrameAnalysis.model_validate(item)
                    completed[fa.frame_path] = fa
                except Exception:
                    pass

    results: List[FrameAnalysis] = []
    total = len(frames)

    for idx, (frame_path, timestamp) in enumerate(frames):
        key = str(frame_path)

        if key in completed:
            results.append(completed[key])
        else:
            fa = analyze_frame(frame_path, timestamp, client)
            results.append(fa)
            completed[key] = fa

            # Persist after every frame so we can resume if interrupted
            _save_analyses(list(completed.values()))

        if progress_callback:
            progress_callback(idx + 1, total)

    return results


def _save_analyses(analyses: List[FrameAnalysis]) -> None:
    with open(FRAME_ANALYSES_FILE, "w") as f:
        json.dump([a.model_dump() for a in analyses], f, indent=2)


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_phase(val) -> GamePhase:
    if not val:
        return GamePhase.UNKNOWN
    try:
        return GamePhase(val)
    except ValueError:
        return GamePhase.UNKNOWN
