"""Pipeline state management for the charades guessing agent.

Holds rolling frame/skeleton buffers, guess budget tracking, and feedback
management across analyze() calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image

# ── Configuration Constants ──────────────────────────────────────────

MIN_SEQUENCE_LEN = 5  # Minimum skeleton frames before HAR runs
HIGH_THRESH = 0.85  # HAR confidence for direct submission (fast path)
MED_THRESH = 0.40  # HAR confidence for LLM reasoning (medium path)
MAX_KEYFRAMES_TO_LLM = 4  # Number of raw frames sent to LLM
MAX_FRAME_BUFFER = 8  # Rolling window size for raw frames
MAX_SKELETON_BUFFER = 16  # Rolling window size for skeleton sequences
ROUND_DURATION_S = 120.0  # Total round duration in seconds
MAX_GUESSES = 10  # Max guesses per round


# ── Guess Budget Manager ─────────────────────────────────────────────


class GuessBudgetManager:
    """Tracks guesses used and decides whether to submit based on confidence
    and time pressure. Lowers the confidence threshold as time runs out."""

    def __init__(self, max_guesses: int = MAX_GUESSES):
        self.max_guesses = max_guesses
        self.used = 0

    def should_guess(self, confidence: float, elapsed_seconds: float) -> bool:
        """Decide whether to spend a guess.

        Args:
            confidence: HAR or LLM confidence score (0.0–1.0).
            elapsed_seconds: Seconds since round start.

        Returns:
            True if we should submit a guess.
        """
        remaining = self.max_guesses - self.used
        if remaining <= 0:
            return False

        # Time pressure: lower the threshold as time runs out
        time_ratio = min(elapsed_seconds / ROUND_DURATION_S, 1.0)

        # Adaptive threshold: starts at HIGH_THRESH, drops to ~0.3 by end
        adjusted_threshold = HIGH_THRESH * (1.0 - 0.6 * time_ratio)

        return confidence >= adjusted_threshold

    def reset(self) -> None:
        self.used = 0


# ── Feedback Tracker ──────────────────────────────────────────────────


class FeedbackTracker:
    """Tracks rejected guesses using Approach C.

    Since __main__.py breaks the loop on correct (201) but continues on
    wrong (409), every guess we return that is followed by another
    analyze() call was incorrect. We infer rejection passively.
    """

    def __init__(self):
        self.pending_guess: str | None = None
        self.rejected: set[str] = set()

    def on_new_frame(self) -> None:
        """Called at the start of each analyze(). If there was a pending
        guess, it means the previous guess was wrong (we'd have stopped
        on 201). Move it to the rejected set."""
        if self.pending_guess:
            self.rejected.add(self.pending_guess.lower().strip())
            self.pending_guess = None

    def register_guess(self, guess: str) -> None:
        """Called when we're about to return a guess from analyze()."""
        self.pending_guess = guess

    def is_rejected(self, candidate: str) -> bool:
        """Check if candidate matches any previously rejected guess."""
        return candidate.lower().strip() in self.rejected

    def reset(self) -> None:
        self.pending_guess = None
        self.rejected.clear()


# ── Pipeline State ────────────────────────────────────────────────────


@dataclass
class PipelineState:
    """Singleton mutable state shared across analyze() calls."""

    frame_buffer: deque[Image.Image] = field(
        default_factory=lambda: deque(maxlen=MAX_FRAME_BUFFER)
    )
    skeleton_buffer: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=MAX_SKELETON_BUFFER)
    )
    guess_budget: GuessBudgetManager = field(default_factory=GuessBudgetManager)
    feedback: FeedbackTracker = field(default_factory=FeedbackTracker)
    round_start: datetime | None = None

    def reset(self, timestamp: datetime) -> None:
        """Reset all state for a new round."""
        self.frame_buffer.clear()
        self.skeleton_buffer.clear()
        self.guess_budget.reset()
        self.feedback.reset()
        self.round_start = timestamp


# Module-level singleton
_state: PipelineState | None = None


def get_state() -> PipelineState:
    """Get or create the pipeline state singleton."""
    global _state
    if _state is None:
        _state = PipelineState()
    return _state


def get_or_init_state(timestamp: datetime) -> PipelineState:
    """Get pipeline state, initializing round_start on first call."""
    state = get_state()
    if state.round_start is None:
        state.round_start = timestamp
    return state


def reset_state(timestamp: datetime) -> None:
    """Force-reset the pipeline state (e.g., on new round)."""
    state = get_state()
    state.reset(timestamp)
