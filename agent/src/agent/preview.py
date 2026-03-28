"""Tk-based preview UI for frame batches sent to analyze()."""

from __future__ import annotations

from datetime import timezone
from typing import Any, Sequence

from PIL import ImageOps

_BG = "#0f1217"
_CARD = "#1f2630"
_TEXT = "#d3dae3"
_SUBTLE = "#8c97a8"
_ACCENT = "#4fb6ff"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


class AnalysisPreviewUI:
    """Preview and control the frame batch sent to analyze()."""

    def __init__(
        self,
        initial_count: int,
        max_count: int,
        enabled: bool = True,
    ) -> None:
        self.max_count = max(1, max_count)
        self._frame_count = _clamp(initial_count, 1, self.max_count)
        self._enabled = enabled
        self._available = False

        self._tk: Any | None = None
        self._root: Any | None = None
        self._count_var: Any | None = None
        self._count_label: Any | None = None
        self._status_var: Any | None = None
        self._grid: Any | None = None
        self._ImageTk: Any | None = None
        self._photo_refs: list[Any] = []

        if not self._enabled:
            return

        try:
            import tkinter as tk
            from PIL import ImageTk
        except Exception as exc:  # pragma: no cover - environment-specific
            print(f"[ui] Preview UI unavailable ({exc}).")
            return

        self._tk = tk
        self._ImageTk = ImageTk

        try:
            root = tk.Tk()
        except Exception as exc:  # pragma: no cover - environment-specific
            print(f"[ui] Preview UI could not start ({exc}).")
            return

        root.title("Casper Analysis Preview")
        root.configure(bg=_BG)
        root.minsize(760, 420)
        root.protocol("WM_DELETE_WINDOW", self.close)

        header = tk.Frame(root, bg=_BG, padx=14, pady=12)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="Frames queued for analysis",
            bg=_BG,
            fg=_TEXT,
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text=(
                "Adjust how many recent frames are sent to analyze(). "
                "Frames are ordered oldest to newest."
            ),
            bg=_BG,
            fg=_SUBTLE,
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w", pady=(2, 8))

        self._count_var = tk.IntVar(value=self._frame_count)

        controls = tk.Frame(header, bg=_BG)
        controls.pack(fill="x")

        self._count_label = tk.Label(
            controls,
            text=f"Frames to send: {self._frame_count}",
            bg=_BG,
            fg=_TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        self._count_label.pack(anchor="w")

        slider = tk.Scale(
            controls,
            from_=1,
            to=self.max_count,
            orient="horizontal",
            variable=self._count_var,
            command=self._on_slider,
            showvalue=False,
            length=340,
            bg=_BG,
            fg=_TEXT,
            troughcolor="#293243",
            highlightthickness=0,
            activebackground=_ACCENT,
        )
        slider.pack(anchor="w")

        self._status_var = tk.StringVar(value="Waiting for frames...")
        status = tk.Label(
            root,
            textvariable=self._status_var,
            bg=_BG,
            fg=_ACCENT,
            font=("Consolas", 10),
            padx=14,
            pady=4,
        )
        status.pack(anchor="w")

        self._grid = tk.Frame(root, bg=_BG, padx=12, pady=8)
        self._grid.pack(fill="both", expand=True)

        self._root = root
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def frame_count(self) -> int:
        if self._count_var is not None:
            try:
                return _clamp(int(self._count_var.get()), 1, self.max_count)
            except Exception:
                return self._frame_count
        return self._frame_count

    def _on_slider(self, value: str) -> None:
        try:
            self._frame_count = _clamp(int(float(value)), 1, self.max_count)
        except ValueError:
            return
        if self._count_label is not None:
            self._count_label.configure(text=f"Frames to send: {self._frame_count}")

    def process_events(self) -> None:
        if not self._available or self._root is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:  # pragma: no cover - UI lifecycle errors
            self.close()

    def render(self, frames: Sequence[Any]) -> None:
        if not self._available or self._root is None or self._grid is None:
            return

        tk = self._tk
        image_tk = self._ImageTk
        if tk is None or image_tk is None:
            return

        self.process_events()

        for widget in self._grid.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        if self._status_var is not None:
            self._status_var.set(
                f"Sending {len(frames)} frame(s) to analyze()"
            )

        if not frames:
            waiting = tk.Label(
                self._grid,
                text="No frames buffered yet.",
                bg=_BG,
                fg=_SUBTLE,
                font=("Segoe UI", 11),
            )
            waiting.pack(anchor="w")
            return

        cols = min(4, max(1, len(frames)))
        for idx, frame in enumerate(frames, start=1):
            row, col = divmod(idx - 1, cols)

            card = tk.Frame(
                self._grid,
                bg=_CARD,
                padx=8,
                pady=8,
                bd=1,
                relief="solid",
            )
            card.grid(row=row, column=col, padx=6, pady=6, sticky="n")

            preview_img = frame.image.copy()
            preview_img.thumbnail((220, 140))
            preview_img = ImageOps.expand(preview_img, border=1, fill="#425067")

            photo = image_tk.PhotoImage(preview_img)
            self._photo_refs.append(photo)

            img_label = tk.Label(card, image=photo, bg=_CARD)
            img_label.pack()

            label = "newest" if idx == len(frames) else "oldest" if idx == 1 else "recent"
            ts_text = frame.timestamp.astimezone(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            meta = tk.Label(
                card,
                text=(
                    f"#{idx} ({label})  {frame.image.size[0]}x{frame.image.size[1]}\n"
                    f"{ts_text} UTC"
                ),
                bg=_CARD,
                fg=_TEXT,
                justify="left",
                font=("Consolas", 9),
            )
            meta.pack(anchor="w", pady=(6, 0))

    def close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._available = False
