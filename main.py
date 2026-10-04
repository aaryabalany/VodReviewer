"""VodReviewer CLI — Fortnite tournament VOD coaching tool."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

import os

from anthropic import AnthropicBedrock

from config import (
    FRAME_ANALYSES_FILE, TIMELINE_FILE, COACHING_FILE, REPORT_FILE, DATA_DIR
)
from game_state.models import TournamentReport, GameSummary, RecurringIssue

load_dotenv()
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────

def _step_done(label: str) -> None:
    console.print(f"  [green]✓[/green] {label}")


def _step_skip(label: str) -> None:
    console.print(f"  [dim]↩ {label} (cached)[/dim]")


def _load_game_summaries() -> list[GameSummary]:
    if not COACHING_FILE.exists():
        return []
    with open(COACHING_FILE) as f:
        data = json.load(f)
    return [GameSummary.model_validate(gs) for gs in data.get("game_summaries", [])]


def _save_game_summaries(summaries: list[GameSummary]) -> None:
    with open(COACHING_FILE, "w") as f:
        json.dump({"game_summaries": [gs.model_dump() for gs in summaries]}, f, indent=2)


# ── Commands ───────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """Fortnite tournament VOD coaching tool."""


@cli.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--force", is_flag=True, help="Re-run all steps, ignoring cached data.")
@click.option(
    "--skip-to",
    type=click.Choice(["frames", "perception", "timeline", "coaching", "report"]),
    default=None,
    help="Skip directly to a later pipeline stage (uses cached data from earlier stages).",
)
def analyze(video_path: str, force: bool, skip_to: str | None) -> None:
    """Run the full coaching pipeline on a tournament recording."""

    console.print(Panel.fit(
        "[bold cyan]Fortnite VOD Coach[/bold cyan]\n"
        f"[dim]{Path(video_path).name}[/dim]",
        border_style="cyan",
    ))

    # ── Stage ordering for --skip-to logic ──────────────────────────────────
    stages = ["frames", "perception", "timeline", "coaching", "report"]
    skip_idx = stages.index(skip_to) if skip_to else -1

    # ── Stage 1: Extract frames ─────────────────────────────────────────────
    console.print("\n[bold]Stage 1/5 — Frame Extraction[/bold]")
    from ingestion.video_processor import get_video_metadata, extract_frames

    meta = get_video_metadata(video_path)
    console.print(
        f"  [dim]Duration: {meta.duration/3600:.1f}h | "
        f"{meta.width}×{meta.height} @ {meta.fps:.1f}fps[/dim]"
    )

    if skip_idx >= 1 and not force:
        frames = extract_frames(video_path, force=False)
        _step_skip(f"Frames already extracted ({len(frames)} frames)")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("  Extracting frames…", total=None)
            frames = extract_frames(video_path, force=force)
            progress.update(task, total=1, completed=1)
        _step_done(f"Extracted {len(frames)} frames")

    if not frames:
        console.print("[red]No frames extracted — check the video file.[/red]")
        sys.exit(1)

    # ── Stage 2: Per-frame HUD analysis ─────────────────────────────────────
    console.print("\n[bold]Stage 2/5 — Frame Perception (Claude Haiku)[/bold]")
    from perception.frame_analyzer import analyze_frames_batch
    from game_state.models import FrameAnalysis

    if skip_idx >= 2 and not force and FRAME_ANALYSES_FILE.exists():
        with open(FRAME_ANALYSES_FILE) as fh:
            analyses = [FrameAnalysis.model_validate(item) for item in json.load(fh)]
        _step_skip(f"Loaded {len(analyses)} frame analyses from cache")
    else:
        analyzed_count = [0]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("  Analyzing frames…", total=len(frames))

            def on_progress(current: int, total: int) -> None:
                progress.update(task, completed=current)
                analyzed_count[0] = current

            analyses = analyze_frames_batch(
                frames,
                resume=(not force),
                progress_callback=on_progress,
            )

        _step_done(f"Analyzed {len(analyses)} frames")

    # ── Stage 3: Build timeline ──────────────────────────────────────────────
    console.print("\n[bold]Stage 3/5 — Timeline Construction[/bold]")
    from game_state.timeline import build_timeline, load_timeline

    if skip_idx >= 3 and not force and TIMELINE_FILE.exists():
        games, events = load_timeline()
        _step_skip(f"Loaded timeline: {len(games)} games from cache")
    else:
        games, events = build_timeline(analyses, save=True)
        _step_done(f"Identified {len(games)} games")

    if not games:
        console.print("[red]No games detected — check the video or lower MIN_FRAMES_PER_GAME in config.py.[/red]")
        sys.exit(1)

    _print_game_table(games, events)

    # ── Stage 4: Coaching analysis ───────────────────────────────────────────
    console.print("\n[bold]Stage 4/5 — Coaching Analysis (Claude Sonnet)[/bold]")
    from coaching.llm_coach import analyze_single_game, generate_tournament_coaching
    from coaching.patterns import find_recurring_issues, identify_strengths

    client = AnthropicBedrock()

    if skip_idx >= 4 and not force and COACHING_FILE.exists():
        game_summaries = _load_game_summaries()
        _step_skip(f"Loaded {len(game_summaries)} game analyses from cache")
    else:
        existing_summaries = _load_game_summaries() if not force else []
        existing_by_game = {gs.game_number: gs for gs in existing_summaries}
        game_summaries = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("  Coaching games…", total=len(games))

            for gn in sorted(games.keys()):
                if gn in existing_by_game:
                    game_summaries.append(existing_by_game[gn])
                    progress.advance(task)
                    continue

                progress.update(task, description=f"  Coaching game {gn}…")
                gs = analyze_single_game(gn, games[gn], events.get(gn, []), client)
                game_summaries.append(gs)
                _save_game_summaries(game_summaries)
                progress.advance(task)

        _step_done(f"Coached {len(game_summaries)} games")

    # Tournament-level cross-game analysis
    console.print("  Running tournament-level pattern analysis…")
    tournament_data = generate_tournament_coaching(game_summaries, client)
    rule_based_issues = find_recurring_issues(game_summaries)
    cross_game_strengths = identify_strengths(game_summaries)
    _step_done("Tournament analysis complete")

    # ── Stage 5: Generate report ─────────────────────────────────────────────
    console.print("\n[bold]Stage 5/5 — Report Generation[/bold]")
    from game_state.models import RecurringIssue
    from reports.generator import save_report

    # Merge LLM recurring issues with rule-based ones (LLM takes priority)
    llm_issues = [
        RecurringIssue(**issue) for issue in tournament_data.get("recurring_issues", [])
    ]
    # Add rule-based issues that weren't already caught by the LLM
    llm_categories = {i.category for i in llm_issues}
    merged_issues = llm_issues + [i for i in rule_based_issues if i.category not in llm_categories]

    report = TournamentReport(
        video_path=video_path,
        video_duration_seconds=meta.duration,
        total_games=len(game_summaries),
        analysis_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        games=game_summaries,
        recurring_issues=merged_issues,
        strengths=tournament_data.get("strengths", []) + cross_game_strengths,
        overall_coaching_summary=tournament_data.get("overall_coaching_summary", ""),
        priority_focus_areas=tournament_data.get("priority_focus_areas", []),
        session_rating=tournament_data.get("session_rating"),
    )

    output_path = save_report(report)
    _step_done(f"Report saved → {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print("\n" + "─" * 60)
    rprint(f"\n[bold green]Done![/bold green] Coaching report: [cyan]{output_path}[/cyan]\n")

    if report.session_rating:
        rating_map = {
            "strong": "[bold green]Strong Session ✅[/bold green]",
            "average": "[bold yellow]Average Session ➖[/bold yellow]",
            "needs_work": "[bold red]Needs Work ❌[/bold red]",
        }
        console.print(f"Session Rating: {rating_map.get(report.session_rating, report.session_rating)}")

    if report.recurring_issues:
        console.print(f"\n[bold]Top Recurring Issues:[/bold]")
        for issue in report.recurring_issues[:3]:
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(issue.severity, "⚪")
            console.print(f"  {emoji} {issue.title} ({issue.occurrences} games)")

    if report.priority_focus_areas:
        console.print(f"\n[bold]Priority Focus for Next Session:[/bold]")
        for i, area in enumerate(report.priority_focus_areas[:3], 1):
            console.print(f"  {i}. {area}")

    console.print()


@cli.command()
def clean() -> None:
    """Remove all cached data (frames, analyses, timeline, coaching)."""
    from ingestion.video_processor import FRAMES_DIR

    targets = [
        (FRAMES_DIR, "frames directory"),
        (FRAME_ANALYSES_FILE, "frame analyses"),
        (TIMELINE_FILE, "timeline"),
        (COACHING_FILE, "coaching data"),
        (REPORT_FILE, "report"),
    ]

    for target, label in targets:
        if target.exists():
            if target.is_dir():
                import shutil
                shutil.rmtree(target)
                target.mkdir()
            else:
                target.unlink()
            console.print(f"  [green]✓[/green] Cleared {label}")
        else:
            console.print(f"  [dim]- {label} (not found)[/dim]")

    console.print("\n[bold green]Cache cleared.[/bold green]")


@cli.command()
def status() -> None:
    """Show the status of cached pipeline data."""
    from config import FRAMES_DIR

    table = Table(title="Pipeline Cache Status", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("File", style="dim")
    table.add_column("Status")

    def _status(path: Path) -> str:
        if not path.exists():
            return "[red]missing[/red]"
        if path.is_dir():
            count = len(list(path.glob("*.jpg")))
            return f"[green]{count} frames[/green]" if count else "[yellow]empty[/yellow]"
        size_kb = path.stat().st_size // 1024
        return f"[green]✓ {size_kb}KB[/green]"

    table.add_row("1 · Frame Extraction", str(FRAMES_DIR.relative_to(DATA_DIR.parent)), _status(FRAMES_DIR))
    table.add_row("2 · Frame Perception", str(FRAME_ANALYSES_FILE.relative_to(DATA_DIR.parent)), _status(FRAME_ANALYSES_FILE))
    table.add_row("3 · Timeline", str(TIMELINE_FILE.relative_to(DATA_DIR.parent)), _status(TIMELINE_FILE))
    table.add_row("4 · Coaching", str(COACHING_FILE.relative_to(DATA_DIR.parent)), _status(COACHING_FILE))
    table.add_row("5 · Report", str(REPORT_FILE.relative_to(DATA_DIR.parent)), _status(REPORT_FILE))

    console.print(table)


# ── Helpers ────────────────────────────────────────────────────────────────

def _print_game_table(
    games: dict,
    events: dict,
) -> None:
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Game", justify="right", style="bold")
    table.add_column("Duration")
    table.add_column("Frames")
    table.add_column("Events")

    for gn in sorted(games.keys()):
        frames = games[gn]
        evts = events.get(gn, [])
        active = [f for f in frames if f.game_phase.is_active_gameplay]
        start = frames[0].timestamp
        end = frames[-1].timestamp
        duration = (end - start) / 60
        table.add_row(
            str(gn),
            f"{duration:.0f}min",
            f"{len(active)} active",
            str(len(evts)),
        )

    console.print(table)


if __name__ == "__main__":
    cli()
