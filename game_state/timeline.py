"""Build a structured timeline from raw frame analyses and segment into games."""
from __future__ import annotations

import json
from typing import List, Tuple

from config import MIN_FRAMES_PER_GAME, TIMELINE_FILE
from game_state.models import FrameAnalysis, GameEvent, GamePhase


def assign_game_numbers(analyses: List[FrameAnalysis]) -> List[FrameAnalysis]:
    """
    Walk the frame timeline and assign a game_number to each frame.
    A new game starts after a transition through loading/lobby frames.
    """
    game_number = 1
    in_between_games = True  # Start in "between games" state
    transition_count = 0
    TRANSITION_MIN = 3  # require N consecutive between-game frames to confirm game ended

    for frame in analyses:
        if frame.skipped:
            frame.game_number = game_number
            continue

        is_between = frame.game_phase.is_between_games

        if is_between:
            transition_count += 1
            if transition_count >= TRANSITION_MIN and not in_between_games:
                # We just finished a game — next active frame starts a new one
                in_between_games = True
        else:
            if in_between_games and frame.game_phase.is_active_gameplay:
                # First active frame after a gap — new game
                game_number += 1
                in_between_games = False
            transition_count = 0

        frame.game_number = game_number

    # Game numbers start at 2 because we increment on first active gameplay.
    # Re-number them to start at 1 for clean output.
    offset = (analyses[0].game_number or 1) - 1
    for frame in analyses:
        if frame.game_number is not None:
            frame.game_number -= offset

    return analyses


def filter_real_games(
    analyses: List[FrameAnalysis],
) -> dict[int, List[FrameAnalysis]]:
    """
    Group frames by game_number and drop groups with too few active frames
    (these are lobby/menu segments that leaked through).
    """
    games: dict[int, List[FrameAnalysis]] = {}
    for fa in analyses:
        gn = fa.game_number or 0
        games.setdefault(gn, []).append(fa)

    real_games: dict[int, List[FrameAnalysis]] = {}
    new_number = 1
    for gn in sorted(games):
        frames = games[gn]
        active = [f for f in frames if f.game_phase.is_active_gameplay]
        if len(active) >= MIN_FRAMES_PER_GAME:
            # Re-number cleanly
            for f in frames:
                f.game_number = new_number
            real_games[new_number] = frames
            new_number += 1

    return real_games


def extract_game_events(
    game_frames: List[FrameAnalysis],
    game_number: int,
) -> List[GameEvent]:
    """
    Scan a single game's frame sequence and emit discrete events:
    - Fight started / ended
    - Player died (phase transitions to dead/spectating)
    - Low health warning
    - Low materials warning
    - Late rotation pressure (in storm / high storm phase)
    - Kill feed: our player got a kill
    """
    events: List[GameEvent] = []
    prev = None

    for frame in game_frames:
        if frame.skipped or not frame.game_phase.is_active_gameplay:
            prev = frame
            continue

        ts = frame.timestamp

        # ── Death detection ────────────────────────────────────────────────
        if prev and prev.game_phase.is_active_gameplay and frame.game_phase == GamePhase.DEAD:
            hp_info = ""
            if prev.health is not None:
                hp_info = f" (was at {prev.health}hp/{prev.shield or 0} shield, {prev.total_materials or '?'} mats)"
            events.append(GameEvent(
                timestamp=ts,
                game_number=game_number,
                event_type="death",
                description=f"Player eliminated{hp_info}",
                severity="high",
                frame_snapshot=frame,
            ))

        # ── Combat transitions ─────────────────────────────────────────────
        if prev and not prev.is_in_combat and frame.is_in_combat:
            mat_str = ""
            if frame.total_materials is not None:
                mat_str = f" with {frame.total_materials} total materials"
            events.append(GameEvent(
                timestamp=ts,
                game_number=game_number,
                event_type="fight_start",
                description=f"Fight started{mat_str}",
                severity="medium",
                frame_snapshot=frame,
            ))

        if prev and prev.is_in_combat and not frame.is_in_combat:
            events.append(GameEvent(
                timestamp=ts,
                game_number=game_number,
                event_type="fight_end",
                description="Fight ended",
                severity="low",
                frame_snapshot=frame,
            ))

        # ── Resource warnings ──────────────────────────────────────────────
        if frame.total_materials is not None and frame.total_materials < 100:
            events.append(GameEvent(
                timestamp=ts,
                game_number=game_number,
                event_type="low_materials",
                description=f"Critically low materials: {frame.total_materials} total",
                severity="high" if frame.total_materials < 50 else "medium",
                frame_snapshot=frame,
            ))

        if frame.effective_hp is not None and frame.effective_hp < 50:
            events.append(GameEvent(
                timestamp=ts,
                game_number=game_number,
                event_type="low_hp",
                description=f"Low effective HP: {frame.effective_hp} ({frame.health}hp + {frame.shield or 0} shield)",
                severity="high" if frame.effective_hp < 25 else "medium",
                frame_snapshot=frame,
            ))

        # ── Storm pressure (high phase + in combat = dangerous rotation) ──
        if frame.storm_phase is not None and frame.storm_phase >= 5 and frame.is_in_combat:
            events.append(GameEvent(
                timestamp=ts,
                game_number=game_number,
                event_type="late_rotation_fight",
                description=f"Fighting during storm phase {frame.storm_phase} — risky late rotation",
                severity="high",
                frame_snapshot=frame,
            ))

        prev = frame

    # Deduplicate nearby events of the same type (within 30s window)
    return _dedup_events(events, window_seconds=30)


def _dedup_events(events: List[GameEvent], window_seconds: float = 30) -> List[GameEvent]:
    deduped: List[GameEvent] = []
    last_by_type: dict[str, float] = {}

    for event in sorted(events, key=lambda e: e.timestamp):
        last = last_by_type.get(event.event_type, -999)
        if event.timestamp - last >= window_seconds:
            deduped.append(event)
            last_by_type[event.event_type] = event.timestamp

    return deduped


def build_timeline(
    analyses: List[FrameAnalysis],
    save: bool = True,
) -> Tuple[dict[int, List[FrameAnalysis]], dict[int, List[GameEvent]]]:
    """
    Full pipeline: assign game numbers → filter real games → extract events.
    Returns (games_dict, events_dict) keyed by game_number.
    """
    analyses = assign_game_numbers(analyses)
    games = filter_real_games(analyses)

    events: dict[int, List[GameEvent]] = {}
    for gn, frames in games.items():
        events[gn] = extract_game_events(frames, gn)

    if save:
        _save_timeline(games, events)

    return games, events


def _save_timeline(
    games: dict[int, List[FrameAnalysis]],
    events: dict[int, List[GameEvent]],
) -> None:
    data = {
        "games": {
            str(gn): [f.model_dump() for f in frames]
            for gn, frames in games.items()
        },
        "events": {
            str(gn): [e.model_dump() for e in evts]
            for gn, evts in events.items()
        },
    }
    with open(TIMELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_timeline(
) -> Tuple[dict[int, List[FrameAnalysis]], dict[int, List[GameEvent]]]:
    with open(TIMELINE_FILE) as f:
        data = json.load(f)

    games = {
        int(gn): [FrameAnalysis.model_validate(fa) for fa in frames]
        for gn, frames in data["games"].items()
    }
    events = {
        int(gn): [GameEvent.model_validate(e) for e in evts]
        for gn, evts in data["events"].items()
    }
    return games, events
