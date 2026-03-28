"""Casper Agent CLI entry point.

Usage:
    uv run -m agent --practice     # Local camera, no network
    uv run -m agent --live         # Connect to live game
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from typing import cast

from dotenv import load_dotenv
from core import Frame

from agent.preview import AnalysisPreviewUI

_JUDGE_UNAVAILABLE_BACKOFF_CAP_S = 30.0
_MAX_JUDGE_UNAVAILABLE_RETRIES = 5
_JUDGE_UNAVAILABLE_BACKOFF_S = 1.0


AnalyzeBatchFn = Callable[[Sequence[Frame]], Awaitable[str | None]]
AnalyzeSingleFn = Callable[[Frame], Awaitable[str | None]]
AnalyzeFn = AnalyzeBatchFn | AnalyzeSingleFn


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Casper guessing game AI agent",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--practice",
        action="store_true",
        help="Use local camera for offline development",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Connect to a live game round",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for practice mode (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=_positive_int,
        default=1,
        help="Frames per second to sample (default: 1)",
    )
    parser.add_argument(
        "--analysis-frames",
        type=_positive_int,
        default=4,
        help=(
            "Initial number of recent frames passed to analyze "
            "(default: 4)"
        ),
    )
    parser.add_argument(
        "--max-analysis-frames",
        type=_positive_int,
        default=12,
        help=(
            "Maximum size of the rolling frame history used for analysis "
            "(default: 12)"
        ),
    )
    parser.add_argument(
        "--no-preview-ui",
        action="store_true",
        help="Disable the frame preview window",
    )

    args = parser.parse_args()
    if args.analysis_frames > args.max_analysis_frames:
        parser.error("--analysis-frames cannot exceed --max-analysis-frames")
    return args


def _analyze_expects_batch(analyze: AnalyzeFn) -> bool:
    try:
        params = list(inspect.signature(analyze).parameters.values())
    except (TypeError, ValueError):
        return True

    if not params:
        return True

    first = params[0]
    annotation = first.annotation
    if annotation is inspect._empty:
        return first.name.lower() in {"frames", "batch", "history", "window"}

    text = str(annotation).lower()
    return any(token in text for token in ("sequence", "list", "tuple", "iterable"))


def _prepare_batch(
    frame: Frame,
    history: deque[Frame],
    preview: AnalysisPreviewUI,
) -> list[Frame]:
    history.append(frame)
    preview.process_events()
    frame_count = min(preview.frame_count, len(history))
    batch = list(history)[-frame_count:]
    preview.render(batch)
    return batch


async def _run_analyze(
    analyze: AnalyzeFn,
    batch: Sequence[Frame],
    expects_batch: bool,
) -> str | None:
    if expects_batch:
        return await cast(AnalyzeBatchFn, analyze)(batch)
    return await cast(AnalyzeSingleFn, analyze)(batch[-1])


def _queue_latest_batch(
    queue: asyncio.Queue[list[Frame] | None],
    batch: list[Frame] | None,
) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(batch)


async def _feed_batches(
    frame_source: AsyncIterator[Frame],
    history: deque[Frame],
    preview: AnalysisPreviewUI,
    queue: asyncio.Queue[list[Frame] | None],
) -> None:
    try:
        async for frame in frame_source:
            batch = _prepare_batch(frame, history, preview)
            _queue_latest_batch(queue, batch)
    finally:
        # Signal consumer shutdown, keeping only the latest state.
        _queue_latest_batch(queue, None)


async def _next_latest_batch(
    queue: asyncio.Queue[list[Frame] | None],
) -> list[Frame] | None:
    latest = await queue.get()
    while True:
        try:
            latest = queue.get_nowait()
        except asyncio.QueueEmpty:
            return latest


async def run_practice(
    camera: int,
    fps: int,
    analysis_frames: int,
    max_analysis_frames: int,
    preview_ui: bool,
) -> None:
    """Run the agent in practice mode with a local camera."""
    from core import start_practice

    from agent.prompt import analyze

    preview = AnalysisPreviewUI(
        initial_count=analysis_frames,
        max_count=max_analysis_frames,
        enabled=preview_ui,
    )
    frame_history: deque[Frame] = deque(maxlen=max_analysis_frames)
    expects_batch = _analyze_expects_batch(analyze)

    print("=" * 50)
    print("  PRACTICE MODE")
    print("  Local camera — no network required")
    print("=" * 50)
    print(f"  analyze frame count: {analysis_frames} (max {max_analysis_frames})")
    if preview.available:
        print("  preview UI: enabled (adjust frame count with the slider)")
    else:
        print("  preview UI: unavailable or disabled")
    print()

    batch_queue: asyncio.Queue[list[Frame] | None] = asyncio.Queue(maxsize=1)
    capture_task = asyncio.create_task(
        _feed_batches(
            start_practice(camera_index=camera, fps=fps),
            frame_history,
            preview,
            batch_queue,
        )
    )

    try:
        while True:
            batch = await _next_latest_batch(batch_queue)
            if batch is None:
                if capture_task.done():
                    exc = capture_task.exception()
                    if exc is not None:
                        raise exc
                break

            expected = preview.frame_count
            if len(batch) < expected:
                print(f"  [warmup] Buffering frames: {len(batch)}/{expected}")

            guess = await _run_analyze(analyze, batch, expects_batch)
            if guess:
                print(f"  [guess] {guess}")
            else:
                print("  [skip]  No guess this frame")
    finally:
        if not capture_task.done():
            capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await capture_task
        preview.close()


async def run_live(
    analysis_frames: int,
    max_analysis_frames: int,
    preview_ui: bool,
) -> None:
    """Run the agent in live mode against the game server."""
    from api import (
        CasperAPI,
        JudgeUnavailable,
        MaxGuessesReached,
        NoActiveRound,
        Unauthorized,
    )
    from core import start_stream

    from agent.prompt import analyze

    preview = AnalysisPreviewUI(
        initial_count=analysis_frames,
        max_count=max_analysis_frames,
        enabled=preview_ui,
    )
    frame_history: deque[Frame] = deque(maxlen=max_analysis_frames)
    expects_batch = _analyze_expects_batch(analyze)

    print("=" * 50)
    print("  LIVE MODE")
    print("  Connecting to game server...")
    print("=" * 50)
    print(f"  analyze frame count: {analysis_frames} (max {max_analysis_frames})")
    if preview.available:
        print("  preview UI: enabled (adjust frame count with the slider)")
    else:
        print("  preview UI: unavailable or disabled")
    print()

    client = CasperAPI.from_env()

    try:
        feed = await client.get_feed()
    except Unauthorized:
        print("[!] Unauthorized. Check TEAM_TOKEN matches your team's API key.")
        sys.exit(1)
    except NoActiveRound:
        print("[!] No active round. Wait for the admin to start one.")
        sys.exit(1)
    except Exception as exc:
        print(f"[!] Could not connect to game server: {exc}")
        sys.exit(1)

    print(f"[+] Joined round: {feed.round_id}")
    print(f"[+] LiveKit URL:  {feed.livekit_url}")
    print()

    guess_count = 0
    batch_queue: asyncio.Queue[list[Frame] | None] = asyncio.Queue(maxsize=1)
    capture_task = asyncio.create_task(
        _feed_batches(
            start_stream(feed.livekit_url, feed.token),
            frame_history,
            preview,
            batch_queue,
        )
    )

    try:
        while True:
            batch = await _next_latest_batch(batch_queue)
            if batch is None:
                if capture_task.done():
                    exc = capture_task.exception()
                    if exc is not None:
                        raise exc
                break

            expected = preview.frame_count
            if len(batch) < expected:
                print(f"  [warmup] Buffering frames: {len(batch)}/{expected}")

            guess = await _run_analyze(analyze, batch, expects_batch)

            if guess:
                result = None
                n_503 = 0
                try:
                    while True:
                        try:
                            result = await client.guess(guess)
                            break
                        except JudgeUnavailable:
                            if n_503 >= _MAX_JUDGE_UNAVAILABLE_RETRIES:
                                break
                            delay = min(
                                _JUDGE_UNAVAILABLE_BACKOFF_S * (2**n_503),
                                _JUDGE_UNAVAILABLE_BACKOFF_CAP_S,
                            )
                            await asyncio.sleep(delay)
                            n_503 += 1
                except Unauthorized:
                    print("[!] Unauthorized. Check TEAM_TOKEN matches your team's API key.")
                    break
                except NoActiveRound:
                    print("[!] No active round (round may have ended).")
                    break
                except MaxGuessesReached:
                    print("[!] Maximum guesses reached for this round.")
                    break

                if result is None:
                    attempts = 1 + _MAX_JUDGE_UNAVAILABLE_RETRIES
                    print(
                        f"[!] Judge unavailable (503) after {attempts} attempt(s). "
                        "Skipping this guess; will try again on the next frame."
                    )
                    continue

                guess_count += 1
                id_suffix = f" id={result.guess_id}" if result.guess_id is not None else ""
                print(f"  [guess #{guess_count}{id_suffix}] {guess}")

                if result.correct:
                    print()
                    print("=" * 50)
                    print(f"  CORRECT! Solved in {guess_count} guesses.")
                    print("=" * 50)
                    break
            else:
                print("  [skip] No guess this frame")

    except (KeyboardInterrupt, ConnectionError):
        print("\n[!] Disconnected from stream.")
    finally:
        if not capture_task.done():
            capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await capture_task
        preview.close()
        await client.close()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.practice:
        await run_practice(
            camera=args.camera,
            fps=args.fps,
            analysis_frames=args.analysis_frames,
            max_analysis_frames=args.max_analysis_frames,
            preview_ui=not args.no_preview_ui,
        )
    else:
        await run_live(
            analysis_frames=args.analysis_frames,
            max_analysis_frames=args.max_analysis_frames,
            preview_ui=not args.no_preview_ui,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
        import os
        os._exit(0)
