"""Claude Sonnet coaching analysis — per-game and tournament-level."""
from __future__ import annotations

import json
from typing import List

from anthropic import AnthropicBedrock

from config import COACHING_MODEL, COACHING_MAX_TOKENS
from game_state.models import (
    FrameAnalysis, GameEvent, CoachingMoment, GameSummary,
)


def _format_game_context(
    game_number: int,
    frames: List[FrameAnalysis],
    events: List[GameEvent],
) -> str:
    """Build a compact text summary of a game's data for the LLM prompt."""
    if not frames:
        return "(no frames)"

    start_ts = frames[0].timestamp
    end_ts = frames[-1].timestamp
    duration = (end_ts - start_ts) / 60

    # Sample key frames spread across the game (max 20 data points)
    step = max(1, len(frames) // 20)
    sampled = frames[::step]

    def fmt_frame(f: FrameAnalysis) -> str:
        t = int(f.timestamp - start_ts)
        hp = f"hp={f.health}/{f.shield or 0}shield" if f.health is not None else "hp=?"
        mats = f"mats={f.total_materials}" if f.total_materials is not None else "mats=?"
        combat = " [IN FIGHT]" if f.is_in_combat else ""
        players = f" players_left={f.total_players_left}" if f.total_players_left else ""
        return f"  +{t}s {f.game_phase.value} {hp} {mats}{combat}{players}"

    frames_text = "\n".join(fmt_frame(f) for f in sampled)

    event_lines = [
        f"  [{_fmt_rel_ts(e.timestamp, start_ts)}] {e.event_type}: {e.description}"
        for e in events
    ]
    events_text = "\n".join(event_lines) if event_lines else "  (no notable events detected)"

    # Determine apparent outcome
    last_phase = frames[-1].game_phase
    outcome = "unknown outcome"
    if last_phase == GamePhase_dead:
        outcome = "eliminated"
    elif any(f.total_players_left == 1 for f in frames if f.total_players_left):
        outcome = "potential victory"

    return f"""GAME {game_number} ({duration:.1f} min, {outcome})
Key frame data (sampled):
{frames_text}

Detected events:
{events_text}"""


GamePhase_dead = "dead"  # avoid circular import; string compare is fine here


def _fmt_rel_ts(ts: float, start: float) -> str:
    s = int(ts - start)
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"


def analyze_single_game(
    game_number: int,
    frames: List[FrameAnalysis],
    events: List[GameEvent],
    client: AnthropicBedrock,
) -> GameSummary:
    """
    Send one game's data to Claude Sonnet and parse coaching moments.
    """
    game_context = _format_game_context(game_number, frames, events)
    start_ts = frames[0].timestamp if frames else 0
    end_ts = frames[-1].timestamp if frames else 0

    prompt = f"""You are a professional Fortnite competitive coach analyzing tournament VOD footage.

Below is structured data extracted from a Fortnite game via computer vision — health/shield values, material counts, fight detections, and notable events at sampled timestamps.

{game_context}

Your job:
1. Identify SPECIFIC coaching moments — mistakes and good decisions
2. For each moment, answer: what happened, why it matters competitively, and what to do differently
3. Focus on decisions and patterns, not just events (e.g., "fought with 45 mats in a box fight" is more useful than "had a fight")
4. Be direct and specific. Avoid generic advice.

Respond with ONLY valid JSON (no markdown):
{{
  "placement": <integer placement if determinable, or null>,
  "kills": <integer kills if determinable, or null>,
  "overall_assessment": "<2-3 sentence honest assessment of this game>",
  "key_mistakes": ["<brief mistake 1>", "<brief mistake 2>"],
  "key_strengths": ["<brief strength 1>"],
  "coaching_moments": [
    {{
      "timestamp_offset": <seconds from game start as integer>,
      "type": "<mistake|good_decision|neutral>",
      "category": "<rotation|resource_management|fight_selection|positioning|game_sense|mechanics|endgame>",
      "what_happened": "<specific description of what occurred>",
      "why_it_matters": "<competitive impact — why this costs placement/health/resources>",
      "what_to_do_differently": "<concrete actionable change, or null for good_decision>",
      "severity": "<low|medium|high|critical>"
    }}
  ]
}}

Only include moments where you have enough data to be specific. 3-6 moments per game is ideal.
"""

    with client.messages.stream(
        model=COACHING_MODEL,
        max_tokens=COACHING_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    moments: List[CoachingMoment] = []
    for m in data.get("coaching_moments", []):
        offset = m.get("timestamp_offset", 0)
        moments.append(CoachingMoment(
            timestamp=start_ts + offset,
            game_number=game_number,
            type=m.get("type", "neutral"),
            category=m.get("category", "general"),
            what_happened=m.get("what_happened", ""),
            why_it_matters=m.get("why_it_matters", ""),
            what_to_do_differently=m.get("what_to_do_differently"),
            severity=m.get("severity", "low"),
        ))

    return GameSummary(
        game_number=game_number,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        placement=data.get("placement"),
        kills=data.get("kills"),
        coaching_moments=moments,
        key_mistakes=data.get("key_mistakes") or [],
        key_strengths=data.get("key_strengths") or [],
        overall_assessment=data.get("overall_assessment", ""),
    )


def generate_tournament_coaching(
    game_summaries: List[GameSummary],
    client: AnthropicBedrock,
) -> dict:
    """
    Cross-game analysis: recurring issues, patterns, overall assessment.
    Returns a dict with keys: recurring_issues, strengths, overall_summary,
    priority_focus_areas, session_rating.
    """
    # Build a compact cross-game summary for the prompt
    game_texts = []
    for gs in game_summaries:
        mistakes = "\n".join(f"  - {m}" for m in gs.key_mistakes) or "  (none noted)"
        strengths = "\n".join(f"  - {s}" for s in gs.key_strengths) or "  (none noted)"
        moments_text = ""
        for cm in gs.coaching_moments:
            moments_text += f"\n  [{cm.timestamp_fmt}] [{cm.category}] {cm.what_happened}"

        game_texts.append(
            f"GAME {gs.game_number} (~{gs.duration_minutes:.0f}min):\n"
            f"Assessment: {gs.overall_assessment}\n"
            f"Mistakes:\n{mistakes}\n"
            f"Strengths:\n{strengths}\n"
            f"Moments:{moments_text}"
        )

    all_games_text = "\n\n---\n\n".join(game_texts)

    prompt = f"""You are a professional Fortnite competitive coach reviewing a full tournament session ({len(game_summaries)} games).

Here is per-game coaching data:

{all_games_text}

Your job:
1. Identify RECURRING ISSUES — mistakes that appear in multiple games (these are the highest-priority coaching points)
2. Identify cross-game STRENGTHS
3. Write an honest overall tournament assessment
4. Prioritize 2-3 focus areas for the player's next practice session

Critical: A recurring issue appearing in 3+ games should be treated as a CRITICAL priority.
Be specific about the pattern, not just the category.

Respond with ONLY valid JSON (no markdown):
{{
  "session_rating": "<strong|average|needs_work>",
  "overall_coaching_summary": "<3-4 sentences honest tournament assessment>",
  "priority_focus_areas": ["<area 1>", "<area 2>", "<area 3>"],
  "strengths": ["<cross-game strength 1>", "<cross-game strength 2>"],
  "recurring_issues": [
    {{
      "category": "<rotation|resource_management|fight_selection|positioning|game_sense|mechanics|endgame>",
      "title": "<short descriptive title>",
      "occurrences": <integer number of games this appeared in>,
      "game_numbers": [<list of game numbers>],
      "severity": "<low|medium|high|critical>",
      "description": "<what specifically keeps happening>",
      "pattern": "<when and why this pattern occurs>",
      "short_term_fix": "<what to focus on immediately in next session>",
      "long_term_fix": "<what to practice to eliminate this tendency long-term>"
    }}
  ]
}}

Only report issues that genuinely recur. Quality over quantity — 2-4 well-described recurring issues beats 8 vague ones.
"""

    with client.messages.stream(
        model=COACHING_MODEL,
        max_tokens=COACHING_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)
