"""Markdown coaching report generator."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from game_state.models import TournamentReport, GameSummary, RecurringIssue, CoachingMoment
from config import REPORT_FILE


SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}

TYPE_LABEL = {
    "mistake": "❌ Mistake",
    "good_decision": "✅ Good Decision",
    "neutral": "ℹ️ Note",
}


def generate_report(report: TournamentReport) -> str:
    lines: List[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Fortnite Tournament VOD Coaching Report",
        "",
        f"**Video:** `{Path(report.video_path).name}`  ",
        f"**Duration:** {_fmt_duration(report.video_duration_seconds)}  ",
        f"**Games Analyzed:** {report.total_games}  ",
        f"**Generated:** {report.analysis_timestamp}",
        "",
    ]

    # ── Overall Assessment ───────────────────────────────────────────────────
    rating_display = {
        "strong": "Strong Session ✅",
        "average": "Average Session ➖",
        "needs_work": "Needs Work ❌",
    }.get(report.session_rating or "", "")

    if rating_display:
        lines += [f"## Session Rating: {rating_display}", ""]

    if report.overall_coaching_summary:
        lines += [
            "## Overall Assessment",
            "",
            report.overall_coaching_summary,
            "",
        ]

    # ── Priority Focus Areas ─────────────────────────────────────────────────
    if report.priority_focus_areas:
        lines += ["## Priority Focus Areas for Next Session", ""]
        for i, area in enumerate(report.priority_focus_areas, 1):
            lines.append(f"{i}. {area}")
        lines.append("")

    # ── Recurring Issues ─────────────────────────────────────────────────────
    if report.recurring_issues:
        lines += ["## Recurring Issues (Cross-Game Patterns)", ""]
        lines += [
            "> These patterns appeared across multiple games. Fixing one recurring issue",
            "> will have more impact than fixing five one-off mistakes.",
            "",
        ]
        for issue in sorted(
            report.recurring_issues,
            key=lambda i: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(i.severity, 0),
            reverse=True,
        ):
            lines += _format_recurring_issue(issue)

    # ── Per-Game Analysis ────────────────────────────────────────────────────
    lines += ["## Per-Game Analysis", ""]

    for gs in sorted(report.games, key=lambda g: g.game_number):
        lines += _format_game_summary(gs)

    # ── Strengths ────────────────────────────────────────────────────────────
    if report.strengths:
        lines += ["## Strengths", ""]
        for strength in report.strengths:
            lines.append(f"- {strength}")
        lines.append("")

    return "\n".join(lines)


def _format_recurring_issue(issue: RecurringIssue) -> List[str]:
    emoji = SEVERITY_EMOJI.get(issue.severity, "⚪")
    games_str = ", ".join(f"G{n}" for n in issue.game_numbers)
    lines = [
        f"### {emoji} {issue.title}",
        f"**Severity:** {issue.severity.title()} | **Occurrences:** {issue.occurrences} games ({games_str})",
        "",
        f"**What's happening:** {issue.description}",
        "",
        f"**Pattern:** {issue.pattern}",
        "",
        f"**Short-term fix:** {issue.short_term_fix}",
        "",
        f"**Long-term fix:** {issue.long_term_fix}",
        "",
    ]
    if issue.example_timestamps:
        ts_str = " · ".join(_fmt_ts(t) for t in issue.example_timestamps)
        lines += [f"**Example timestamps:** {ts_str}", ""]
    lines.append("---")
    lines.append("")
    return lines


def _format_game_summary(gs: GameSummary) -> List[str]:
    header_parts = [f"### Game {gs.game_number}"]
    if gs.placement:
        header_parts.append(f"(#{gs.placement})")
    if gs.kills is not None:
        header_parts.append(f"— {gs.kills}K")

    lines = [
        " ".join(header_parts),
        f"**Duration:** {gs.duration_minutes:.0f}min | "
        f"**Timeframe:** {_fmt_ts(gs.start_timestamp)} — {_fmt_ts(gs.end_timestamp)}",
        "",
    ]

    if gs.overall_assessment:
        lines += [gs.overall_assessment, ""]

    if gs.key_strengths:
        lines.append("**Strengths:**")
        for s in gs.key_strengths:
            lines.append(f"- ✅ {s}")
        lines.append("")

    if gs.key_mistakes:
        lines.append("**Key Mistakes:**")
        for m in gs.key_mistakes:
            lines.append(f"- ❌ {m}")
        lines.append("")

    if gs.coaching_moments:
        lines.append("**Coaching Moments:**")
        lines.append("")
        for moment in sorted(gs.coaching_moments, key=lambda m: m.timestamp):
            lines += _format_coaching_moment(moment)

    lines.append("")
    return lines


def _format_coaching_moment(moment: CoachingMoment) -> List[str]:
    emoji = SEVERITY_EMOJI.get(moment.severity, "⚪")
    type_label = TYPE_LABEL.get(moment.type, moment.type)

    lines = [
        f"#### {emoji} `{moment.timestamp_fmt}` — {type_label}: {moment.category.replace('_', ' ').title()}",
        "",
        f"**What happened:** {moment.what_happened}",
        "",
        f"**Why it matters:** {moment.why_it_matters}",
        "",
    ]
    if moment.what_to_do_differently:
        lines += [f"**Do differently:** {moment.what_to_do_differently}", ""]

    return lines


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def save_report(report: TournamentReport, path: Path = REPORT_FILE) -> Path:
    content = generate_report(report)
    with open(path, "w") as f:
        f.write(content)
    return path
