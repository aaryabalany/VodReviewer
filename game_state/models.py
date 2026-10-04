from __future__ import annotations
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class GamePhase(str, Enum):
    LOADING = "loading"
    LOBBY = "lobby"
    BUS = "bus"
    SKYDIVING = "skydiving"
    LANDING = "landing"
    EARLY_GAME = "early_game"
    MID_GAME = "mid_game"
    LATE_GAME = "late_game"
    ENDGAME = "endgame"
    DEAD = "dead"
    SPECTATING = "spectating"
    UNKNOWN = "unknown"

    @property
    def is_active_gameplay(self) -> bool:
        return self in {
            GamePhase.LANDING,
            GamePhase.EARLY_GAME,
            GamePhase.MID_GAME,
            GamePhase.LATE_GAME,
            GamePhase.ENDGAME,
        }

    @property
    def is_between_games(self) -> bool:
        return self in {GamePhase.LOADING, GamePhase.LOBBY}


class FrameAnalysis(BaseModel):
    """Raw output from Claude Vision for a single video frame."""
    timestamp: float
    frame_path: str

    game_phase: GamePhase = GamePhase.UNKNOWN
    health: Optional[int] = None
    shield: Optional[int] = None
    wood: Optional[int] = None
    brick: Optional[int] = None
    metal: Optional[int] = None
    total_players_left: Optional[int] = None
    is_in_combat: bool = False
    weapon_equipped: Optional[str] = None
    kill_feed: List[str] = Field(default_factory=list)
    storm_phase: Optional[int] = None
    notable_events: List[str] = Field(default_factory=list)
    scene_description: str = ""

    # Derived fields set during timeline construction
    game_number: Optional[int] = None
    skipped: bool = False  # True if frame was too dark / not analyzed

    @property
    def total_materials(self) -> Optional[int]:
        if self.wood is None and self.brick is None and self.metal is None:
            return None
        return (self.wood or 0) + (self.brick or 0) + (self.metal or 0)

    @property
    def effective_hp(self) -> Optional[int]:
        if self.health is None:
            return None
        return (self.health or 0) + (self.shield or 0)


class GameEvent(BaseModel):
    """A meaningful event extracted from the frame timeline."""
    timestamp: float
    game_number: int
    event_type: str  # "fight_start", "death", "rotation_pressure", "low_resources", etc.
    description: str
    severity: str = "low"  # "low", "medium", "high", "critical"
    frame_snapshot: Optional[FrameAnalysis] = None


class CoachingMoment(BaseModel):
    """A single teachable moment from a game."""
    timestamp: float
    game_number: int
    type: str  # "mistake", "good_decision", "neutral"
    category: str  # "rotation", "resource_management", "fight_selection", "positioning", etc.

    what_happened: str
    why_it_matters: str
    what_to_do_differently: Optional[str] = None
    severity: str = "low"  # "low", "medium", "high", "critical"

    # For display in reports
    @property
    def timestamp_fmt(self) -> str:
        h = int(self.timestamp // 3600)
        m = int((self.timestamp % 3600) // 60)
        s = int(self.timestamp % 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


class RecurringIssue(BaseModel):
    """A pattern that repeats across multiple games — the highest-value coaching output."""
    category: str
    title: str
    occurrences: int
    game_numbers: List[int]
    severity: str  # "low", "medium", "high", "critical"

    description: str
    pattern: str  # What specifically keeps happening
    short_term_fix: str
    long_term_fix: str
    example_timestamps: List[float] = Field(default_factory=list)


class GameSummary(BaseModel):
    """Summary of a single game within the tournament."""
    game_number: int
    start_timestamp: float
    end_timestamp: float
    placement: Optional[int] = None
    kills: Optional[int] = None

    coaching_moments: List[CoachingMoment] = Field(default_factory=list)
    key_mistakes: List[str] = Field(default_factory=list)
    key_strengths: List[str] = Field(default_factory=list)
    overall_assessment: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end_timestamp - self.start_timestamp) / 60


class TournamentReport(BaseModel):
    """Complete coaching report for a full tournament VOD."""
    video_path: str
    video_duration_seconds: float
    total_games: int
    analysis_timestamp: str

    games: List[GameSummary] = Field(default_factory=list)
    recurring_issues: List[RecurringIssue] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)

    overall_coaching_summary: str = ""
    priority_focus_areas: List[str] = Field(default_factory=list)
    session_rating: Optional[str] = None  # "strong", "average", "needs_work"
