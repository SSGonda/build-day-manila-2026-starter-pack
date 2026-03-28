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

_wrong_guesses: list[str] = []
_frame_buffer: list[bytes] = []
_MAX_BUFFERED_FRAMES = 5


def record_wrong_guess(guess: str) -> None:
    """Call this when a guess is rejected (429 response)."""
    _wrong_guesses.append(guess)


def reset_buffer() -> None:
    """Clear the frame buffer after a correct guess."""
    _frame_buffer.clear()


async def analyze(frame: Frame) -> str | None:
    """Analyze buffered frames and return a guess, or None to skip.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    print(
        f"  [agent] Got frame at {frame.timestamp.isoformat()} "
        f"({frame.image.size[0]}x{frame.image.size[1]})"
    )

    img_bytes = io.BytesIO()
    frame.image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    _frame_buffer.append(img_bytes)
    if len(_frame_buffer) > _MAX_BUFFERED_FRAMES:
        _frame_buffer.pop(0)

    print(f"  [agent] Buffer: {len(_frame_buffer)} frames")

    context = ""
    if _wrong_guesses:
        context = f"\n\nThese previous guesses were WRONG - do not guess them again:\n"
        for g in _wrong_guesses:
            context += f"- {g}\n"

    prompt = f"""{SYSTEM_PROMPT}

You are viewing a sequence of {len(_frame_buffer)} frames from a video.
Analyze all frames together to understand the action being performed.

Respond in this exact format:
GUESS: <your answer>
CONFIDENCE: <0-100>
If you are uncertain, set confidence below 50.{context}"""

    message_parts: list = [prompt]
    for fb in _frame_buffer:
        message_parts.append(BinaryContent(data=fb, media_type="image/jpeg"))

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
        return None

    return guess
