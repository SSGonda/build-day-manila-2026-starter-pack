"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import lru_cache
from io import BytesIO

from core import Frame
from pydantic_ai import Agent, BinaryContent

# ---------------------------------------------------------------------------
# System prompt — tweak this to improve your agent's guessing ability.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are playing a visual guessing game. You will receive a screenshot from a
live camera feed. You will receive multiple recent frames in chronological
order (oldest to newest). Your goal is to identify what is being shown as
quickly and accurately as possible.

Rules:
- Give your best guess as a short, specific answer (1-5 words).
- If you're not confident enough yet, respond with exactly "SKIP".
- Use temporal clues across frames (movement, reveal, perspective changes).
"""

ANSWER_SYSTEM_PROMPT = """\
You are the final answer model for a visual guessing game. You will receive
multiple recent frames in chronological order (oldest to newest). Use temporal
clues across frames and always return your single best 1-5 word guess.

Rules:
- Respond with exactly one short guess (1-5 words).
- Do not explain your reasoning.
- Never respond with SKIP.
"""

FAST_MODEL_NAME = "openrouter:google/gemini-3.1-flash-lite-preview"
ANSWER_MODEL_NAME = "openrouter:google/gemini-2.5-pro"
FAST_MODEL_INTERVAL_FRAMES = 3
ANSWER_MODEL_INTERVAL_FRAMES = 10

_MAX_IMAGE_SIDE = 896
_JPEG_QUALITY = 75
_analysis_step = 0


@lru_cache(maxsize=1)
def _get_fast_agent() -> Agent:
    """Build and cache the fast screening model client."""
    return Agent(FAST_MODEL_NAME, system_prompt=SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def _get_answer_agent() -> Agent:
    """Build and cache the answer model client."""
    return Agent(ANSWER_MODEL_NAME, system_prompt=ANSWER_SYSTEM_PROMPT)


def _to_image_content(frame: Frame) -> BinaryContent:
    """Convert a Frame image to BinaryContent for multimodal model input."""
    # Shrink large frames before upload to reduce model latency and payload size.
    image = frame.image.copy()
    image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return BinaryContent(data=buffer.getvalue(), media_type="image/jpeg")


def _build_prompt_parts(
    frames: Sequence[Frame],
    prompt: str,
) -> list[str | BinaryContent]:
    parts: list[str | BinaryContent] = [prompt]
    for index, frame in enumerate(frames, start=1):
        parts.append(
            f"Frame {index}/{len(frames)} captured at {frame.timestamp.isoformat()}."
        )
        parts.append(_to_image_content(frame))
    return parts


def _normalize_answer(text: str) -> str | None:
    answer = text.strip()
    if not answer:
        return None
    if answer.upper() == "SKIP":
        return None
    return " ".join(answer.split())


async def _run_model(
    *,
    agent: Agent,
    model_name: str,
    frames: Sequence[Frame],
    prompt: str,
) -> str | None:
    prompt_parts = _build_prompt_parts(frames, prompt)
    try:
        result = await agent.run(prompt_parts)
    except Exception as exc:
        print(f"  [agent] LLM request failed for {model_name}, skipping: {exc}")
        return None

    normalized = _normalize_answer(str(result.output))
    if normalized is None:
        print(f"  [agent] {model_name} returned SKIP/empty")
    return normalized


async def analyze(frames: Sequence[Frame]) -> str | None:
    """Analyze recent frames and return a guess, or None to skip.

    This implementation sends all provided frames to the configured
    vision model and returns either a short guess or None to skip.

    Args:
        frames: Ordered sequence of recent Frame objects.
            - frames[0] is the oldest
            - frames[-1] is the newest

    Returns:
        A text guess string, or None to skip this frame.
    """
    global _analysis_step

    if not frames:
        return None

    _analysis_step += 1
    step = _analysis_step

    run_fast = step % FAST_MODEL_INTERVAL_FRAMES == 0
    run_answer = step % ANSWER_MODEL_INTERVAL_FRAMES == 0

    if not run_fast and not run_answer:
        return None

    newest = frames[-1]
    oldest = frames[0]
    print(
        "  [agent] Got "
        f"{len(frames)} frame(s) from {oldest.timestamp.isoformat()} "
        f"to {newest.timestamp.isoformat()}"
    )
    print(
        "  [agent] Newest frame: "
        f"{newest.image.size[0]}x{newest.image.size[1]}"
    )
    print(
        "  [agent] Step "
        f"{step} | run_fast={run_fast} run_answer={run_answer}"
    )

    fast_guess: str | None = None
    answer_guess: str | None = None

    if run_fast and run_answer:
        print(
            "  [agent] Running models concurrently: "
            f"{FAST_MODEL_NAME} + {ANSWER_MODEL_NAME}"
        )
        fast_task = asyncio.create_task(
            _run_model(
                agent=_get_fast_agent(),
                model_name=FAST_MODEL_NAME,
                frames=frames,
                prompt=(
                    "What object is shown across these frames? "
                    "Respond with your best 1-5 word guess, or exactly SKIP if unsure."
                ),
            )
        )
        answer_task = asyncio.create_task(
            _run_model(
                agent=_get_answer_agent(),
                model_name=ANSWER_MODEL_NAME,
                frames=frames,
                prompt=(
                    "Give your best single 1-5 word answer for what object is shown "
                    "across these frames. Do not output SKIP."
                ),
            )
        )
        fast_guess, answer_guess = await asyncio.gather(fast_task, answer_task)
    else:
        if run_fast:
            print(f"  [agent] Using fast model: {FAST_MODEL_NAME}")
            fast_guess = await _run_model(
                agent=_get_fast_agent(),
                model_name=FAST_MODEL_NAME,
                frames=frames,
                prompt=(
                    "What object is shown across these frames? "
                    "Respond with your best 1-5 word guess, or exactly SKIP if unsure."
                ),
            )

        if run_answer:
            print(f"  [agent] Using answer model: {ANSWER_MODEL_NAME}")
            answer_guess = await _run_model(
                agent=_get_answer_agent(),
                model_name=ANSWER_MODEL_NAME,
                frames=frames,
                prompt=(
                    "Give your best single 1-5 word answer for what object is shown "
                    "across these frames. Do not output SKIP."
                ),
            )

    if answer_guess:
        return answer_guess

    return fast_guess
