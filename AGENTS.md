# Agent Customization Guide

Everything you need to edit is in `agent/src/agent/prompt.py`.

## The Two Things You Control

### 1. `SYSTEM_PROMPT`

This is the system prompt sent to your vision LLM with every frame batch. Tips:

- Be specific about the output format ("respond with 1-5 words")
- Tell the model to say "SKIP" if uncertain (saves guesses)
- Consider adding reasoning: "First describe what you see, then guess"
- Experiment with chain-of-thought vs. direct answers

### 2. `analyze(frames) -> str | None`

This function receives an ordered sequence of `Frame` objects:
- `frames[0]` — oldest frame in the current batch
- `frames[-1]` — newest frame in the current batch

Each `Frame` has:
- `frame.image` — a `PIL.Image.Image` (RGB)
- `frame.timestamp` — a `datetime` (UTC)

Return a guess string, or `None` to skip this frame.

## Example Implementation

```python
from pydantic_ai import Agent

from core import Frame
from agent.prompt import SYSTEM_PROMPT
from collections.abc import Sequence

agent = Agent("claude-sonnet-4-20250514", system_prompt=SYSTEM_PROMPT)

async def analyze(frames: Sequence[Frame]) -> str | None:
    latest = frames[-1]
    result = await agent.run(
        "What is being shown? Give your best guess.",
        # Attach latest.image or several frames from `frames` here
    )
    answer = result.output.strip()
    return None if answer == "SKIP" else answer
```

## Strategies to Try

- **Accumulate context**: Keep a history of recent frames to spot patterns
- **Confidence threshold**: Only guess when the model is highly confident
- **Multi-model**: Use a fast model for initial screening, a strong model for final guesses
- **Prompt iteration**: Test different prompts in practice mode before going live

## Practice Mode Tips

1. Point your camera at various objects
2. Run `uv run -m agent --practice --analysis-frames 6`
3. Watch what your agent outputs
4. Use the preview slider UI to change how many recent frames are sent
5. Tweak `SYSTEM_PROMPT` and `analyze()` until it reliably identifies things
6. Try `--fps 2` to see if more frames help your strategy
