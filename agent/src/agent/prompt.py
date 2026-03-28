"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

import io

from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent

from core import Frame
from agent.pipeline import (
    MAX_SKELETON_BUFFER,
    MIN_SEQUENCE_LEN,
    get_or_init_state,
    reset_state,
)
from agent.pose import run_yolo_pose

load_dotenv()

SYSTEM_PROMPT = """\
You are playing a charades guessing game. A person will act out a word or phrase using gestures, body movements, and facial expressions. Your goal is to identify what is being acted out as quickly as possible.

What to look for:
- Hand signals and gestures (thumbs up, pointing, counting on fingers)
- Body movements (walking, swimming, flying, dancing)
- Facial expressions (happy, scared, confused, trophy pose)
- Props or objects held up
- Mouthing words (lip-reading)
- Common charades signals: fingers indicating word count, hand over mouth = "can't talk"

Categories (actor may hint at which one):
- Movies, TV shows
- Songs, books
- Common idioms or phrases
- Objects, animals, activities

Strategy:
- If you see someone gesturing, describe what they're DOING, not just what they're wearing
- Look for the overall action-concept, not details
- If unclear, wait for more frames - the actor will give clearer signals
- Answer in 1-5 words, be specific: "flying a kite" not just "kite"

Response format (ALWAYS follow exactly):
GUESS: <your answer>
CONFIDENCE: <0-100>
If uncertain, set confidence below 50.
"""

_agent = Agent("openrouter:qwen/qwen3.5-flash-02-23")


def record_wrong_guess(guess: str) -> None:
    """Call this when a guess is rejected (409 response)."""
    # Handled internally by FeedbackTracker via Approach C
    pass


def reset_buffer() -> None:
    """Clear state after a correct guess or new round."""
    from datetime import datetime, timezone

    reset_state(datetime.now(timezone.utc))


async def analyze(frame: Frame) -> str | None:
    """Analyze a frame through the YOLO-Pose + LLM pipeline.

    1. Push frame into buffers (frame + skeleton keypoints)
    2. Check budget and feedback
    3. If enough frames, run LLM reasoning with keyframes + skeleton data
    4. Return guess or None to skip

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    state = get_or_init_state(frame.timestamp)

    # Approach C: if previous guess was returned, it was wrong
    state.feedback.on_new_frame()

    print(
        f"  [agent] Got frame at {frame.timestamp.isoformat()} "
        f"({frame.image.size[0]}x{frame.image.size[1]})"
    )

    # 1. Buffer the raw frame
    state.frame_buffer.append(frame.image)

    # 2. Run YOLO-Pose → keypoints
    keypoints = run_yolo_pose(frame.image)
    has_person = keypoints.sum() > 0
    state.skeleton_buffer.append(keypoints)

    print(
        f"  [agent] Buffer: {len(state.frame_buffer)} frames, "
        f"person detected: {has_person}"
    )

    # 3. Not enough frames yet? Skip.
    if len(state.skeleton_buffer) < MIN_SEQUENCE_LEN:
        print(
            f"  [agent] Accumulating frames ({len(state.skeleton_buffer)}/{MIN_SEQUENCE_LEN})"
        )
        return None

    # 4. No person detected — skip this cycle
    if not has_person:
        print("  [agent] No person detected, skipping")
        return None

    # 5. Build context from rejected guesses
    excluded = ""
    if state.feedback.rejected:
        excluded = "\n\nThese previous guesses were WRONG — do not guess them again:\n"
        for g in sorted(state.feedback.rejected):
            excluded += f"- {g}\n"

    # 6. Build skeleton motion summary from keypoint buffer
    motion_summary = _build_motion_summary(state.skeleton_buffer)

    # 7. Build LLM prompt with skeleton context
    prompt = f"""{SYSTEM_PROMPT}

You are viewing {len(state.frame_buffer)} frames from a live charades performance.

SKELETON ANALYSIS (from pose detection):
{motion_summary}

Analyze all frames together to understand the action being performed.

Respond in this exact format:
GUESS: <your answer>
CONFIDENCE: <0-100>
If you are uncertain, set confidence below 50.{excluded}"""

    # 8. Attach keyframes (last N frames as images)
    message_parts: list = [prompt]
    img_bytes_list = []
    for img in state.frame_buffer:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_data = buf.getvalue()
        img_bytes_list.append(img_data)
        message_parts.append(BinaryContent(data=img_data, media_type="image/jpeg"))

    # 9. Run LLM
    result = await _agent.run(message_parts)
    response = result.output.strip()

    guess = None
    confidence = 0

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("GUESS:"):
            guess = line[6:].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line[11:].strip())
            except ValueError:
                pass

    if confidence < 50 or not guess:
        print(f"  [agent] Low confidence ({confidence}) or no guess, skipping")
        return None

    # 10. Check if this guess was already rejected
    if state.feedback.is_rejected(guess):
        print(f"  [agent] Guess '{guess}' was already rejected, skipping")
        return None

    # 11. Budget check — should we spend a guess?
    elapsed = (frame.timestamp - state.round_start).total_seconds()
    if not state.guess_budget.should_guess(confidence / 100.0, elapsed):
        print(
            f"  [agent] Budget check failed (conf={confidence}%, "
            f"elapsed={elapsed:.1f}s, used={state.guess_budget.used})"
        )
        return None

    # 12. Register and return the guess
    state.guess_budget.used += 1
    state.feedback.register_guess(guess)
    print(
        f"  [agent] Submitting guess #{state.guess_budget.used}: "
        f"'{guess}' (conf={confidence}%)"
    )
    return guess


def _build_motion_summary(skeleton_buffer) -> str:
    """Build a text summary of skeleton motion from keypoint sequences.

    Tracks wrist, elbow, knee, and ankle movement across frames to give
    the LLM structured motion context.
    """
    if len(skeleton_buffer) < 2:
        return "Insufficient frames for motion analysis."

    import numpy as np

    joints_of_interest = {
        "left_wrist": 9,
        "right_wrist": 10,
        "left_knee": 13,
        "right_knee": 14,
        "nose": 0,
    }

    lines = []
    frames = list(skeleton_buffer)

    for name, idx in joints_of_interest.items():
        positions = []
        for f in frames:
            if f[idx][2] > 0.3:  # confidence threshold
                positions.append((f[idx][0], f[idx][1]))

        if len(positions) < 2:
            continue

        # Compute displacement
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 10:
            movement = "stationary"
        elif abs(dy) > abs(dx):
            movement = "moving UP" if dy < 0 else "moving DOWN"
        else:
            movement = "moving RIGHT" if dx > 0 else "moving LEFT"

        lines.append(f"  - {name}: {movement} (displacement={dist:.0f}px)")

    if not lines:
        return "No clear motion detected in tracked joints."

    return "Tracked joint movements:\n" + "\n".join(lines)
