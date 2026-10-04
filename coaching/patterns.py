"""Cross-game pattern detection to supplement the LLM analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

from game_state.models import GameSummary, CoachingMoment, RecurringIssue


def find_recurring_issues(
    game_summaries: List[GameSummary],
    min_occurrences: int = 2,
) -> List[RecurringIssue]:
    """
    Group coaching moments by category across games and surface recurring patterns.
    This is a rule-based supplement to the LLM tournament analysis.
    """
    # Group mistakes by category
    by_category: Dict[str, List[CoachingMoment]] = defaultdict(list)

    for gs in game_summaries:
        for moment in gs.coaching_moments:
            if moment.type == "mistake":
                by_category[moment.category].append(moment)

    issues: List[RecurringIssue] = []

    for category, moments in by_category.items():
        if len(moments) < min_occurrences:
            continue

        game_numbers = sorted(set(m.game_number for m in moments))
        if len(game_numbers) < min_occurrences:
            continue

        # Pick highest severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_severity = max(moments, key=lambda m: severity_order.get(m.severity, 0))

        example_timestamps = [m.timestamp for m in moments[:3]]

        # Build a summary from the descriptions
        descriptions = [m.what_happened for m in moments]
        pattern_summary = "; ".join(descriptions[:3])
        if len(descriptions) > 3:
            pattern_summary += f" (and {len(descriptions) - 3} more)"

        issues.append(RecurringIssue(
            category=category,
            title=f"Recurring {category.replace('_', ' ').title()} Issue",
            occurrences=len(moments),
            game_numbers=game_numbers,
            severity=max_severity.severity,
            description=f"Appeared in {len(game_numbers)} games: {pattern_summary}",
            pattern="Rule-based detection — see LLM analysis for detailed pattern description",
            short_term_fix=max_severity.what_to_do_differently or "Review these moments in the VOD",
            long_term_fix=f"Practice {category.replace('_', ' ')} decisions in custom lobbies",
            example_timestamps=example_timestamps,
        ))

    # Sort by severity then occurrences
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    issues.sort(
        key=lambda i: (severity_order.get(i.severity, 0), i.occurrences),
        reverse=True,
    )

    return issues


def identify_strengths(game_summaries: List[GameSummary]) -> List[str]:
    """
    Find categories where the player consistently made good decisions.
    """
    by_category: Dict[str, int] = defaultdict(int)

    for gs in game_summaries:
        for moment in gs.coaching_moments:
            if moment.type == "good_decision":
                by_category[moment.category] += 1

    return [
        f"{cat.replace('_', ' ').title()} ({count} good decisions across {len(game_summaries)} games)"
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True)
        if count >= 2
    ]
