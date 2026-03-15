"""Tkinter GUI for AI Text Humanizer.

Provides a desktop window with:
  - An input text area for pasting AI-generated text.
  - A Bypass Strength selector (Light / Medium / Aggressive / Custom).
  - A merge-rate slider (active only in Custom mode).
  - An Adversarial Mode checkbox that enables zero-width space injection and
    homoglyph swapping for maximum AI-detection evasion.
  - A "Humanize" button that calls the core pipeline (multi-pass for
    Medium and Aggressive strengths).
  - An output text area showing the humanized result.
  - A status bar with run statistics.

Launch via::

    aip-gui          # if installed via pip
    python -m aip.gui
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from aip.humanizer import humanize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_TITLE = "AI Text Humanizer"
_PAD = 10
_BTN_PAD = (4, 4)
_SLIDER_FROM = 0.0
_SLIDER_TO = 1.0
_SLIDER_RESOLUTION = 0.05

_PLACEHOLDER_INPUT = (
    "Paste your AI-generated text here…\n\n"
    "Example:\n"
    "Furthermore, it is important to note that natural language processing "
    "has advanced significantly. In conclusion, these models demonstrate "
    "remarkable capabilities."
)

# ---------------------------------------------------------------------------
# Bypass-strength presets
# ---------------------------------------------------------------------------

# Each preset defines the number of sequential humanize() passes and the
# merge rate.  Adversarial mode is controlled separately via the GUI checkbox.
_BYPASS_PRESETS: dict[str, dict[str, int | float] | None] = {
    "Light":      {"passes": 1, "merge_rate": 0.30},
    "Medium":     {"passes": 2, "merge_rate": 0.45},
    "Aggressive": {"passes": 3, "merge_rate": 0.60},
    "Custom":     None,  # uses the manual slider values
}

_DEFAULT_STRENGTH = "Aggressive"

_DARK_BG = "#1e1e2e"
_PANEL_BG = "#2a2a3e"
_ACCENT = "#7c6af5"
_ACCENT_HOVER = "#6a59d1"
_TEXT_FG = "#cdd6f4"
_MUTED_FG = "#6c7086"
_BORDER = "#45475a"
_SUCCESS = "#a6e3a1"
_ERROR = "#f38ba8"


class _HoverButton(tk.Button):
    """Button that changes colour on mouse-over."""

    def __init__(self, master: tk.Widget, **kw: object) -> None:
        self._normal_bg = str(kw.get("bg", kw.get("background", _ACCENT)))
        self._hover_bg = str(kw.get("activebackground", _ACCENT_HOVER))
        kw.setdefault("bg", self._normal_bg)
        kw.setdefault("activebackground", self._hover_bg)
        kw.setdefault("bd", 0)
        kw.setdefault("relief", tk.FLAT)
        kw.setdefault("cursor", "hand2")
        super().__init__(master, **kw)
        self.bind("<Enter>", lambda _e: self.config(bg=self._hover_bg))
        self.bind("<Leave>", lambda _e: self.config(bg=self._normal_bg))


class HumanizerApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title(_WINDOW_TITLE)
        self._root.configure(bg=_DARK_BG)
        self._root.minsize(860, 600)
        self._root.geometry("1080x700")

        # Keep window reasonably scaled
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(1, weight=1)

        self._merge_rate = tk.DoubleVar(value=0.25)
        self._status_text = tk.StringVar(value="Ready.")
        self._bypass_strength = tk.StringVar(value=_DEFAULT_STRENGTH)
        self._adversarial_mode = tk.BooleanVar(value=False)

        self._build_header()
        self._build_body()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        header = tk.Frame(self._root, bg=_PANEL_BG, pady=_PAD)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        tk.Label(
            header,
            text="✦  AI Text Humanizer",
            font=title_font,
            bg=_PANEL_BG,
            fg=_ACCENT,
            padx=_PAD,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            header,
            text="Rewrite AI-generated text so it reads as human-written",
            bg=_PANEL_BG,
            fg=_MUTED_FG,
        ).grid(row=0, column=1, sticky="w", padx=_PAD)

    def _build_body(self) -> None:
        body = tk.Frame(self._root, bg=_DARK_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=_PAD, pady=_PAD)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(1, weight=1)

        # ---- Input ----
        tk.Label(
            body, text="Input Text", bg=_DARK_BG, fg=_TEXT_FG, anchor="w"
        ).grid(row=0, column=0, sticky="ew")

        input_frame = tk.Frame(body, bg=_BORDER, bd=0)
        input_frame.grid(row=1, column=0, sticky="nsew")
        input_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)

        self._input_text = tk.Text(
            input_frame,
            wrap=tk.WORD,
            bg=_PANEL_BG,
            fg=_MUTED_FG,
            insertbackground=_TEXT_FG,
            selectbackground=_ACCENT,
            relief=tk.FLAT,
            bd=8,
            font=("Helvetica", 11),
        )
        self._input_text.grid(row=0, column=0, sticky="nsew")
        self._input_text.insert("1.0", _PLACEHOLDER_INPUT)

        in_scroll = ttk.Scrollbar(
            input_frame, orient=tk.VERTICAL, command=self._input_text.yview
        )
        in_scroll.grid(row=0, column=1, sticky="ns")
        self._input_text.configure(yscrollcommand=in_scroll.set)

        # Clear placeholder on first focus
        self._input_text.bind("<FocusIn>", self._clear_placeholder)

        # ---- Controls (centre column) ----
        controls = tk.Frame(body, bg=_DARK_BG, width=170)
        controls.grid(row=0, column=1, rowspan=2, sticky="ns", padx=_PAD)
        controls.columnconfigure(0, weight=1)
        controls.grid_propagate(False)

        lbl_style = {"bg": _DARK_BG, "fg": _TEXT_FG, "anchor": "w"}
        val_style = {"bg": _DARK_BG, "fg": _ACCENT}
        slider_style = {
            "orient": tk.HORIZONTAL,
            "bg": _DARK_BG,
            "fg": _ACCENT,
            "troughcolor": _BORDER,
            "highlightthickness": 0,
            "bd": 0,
            "sliderlength": 18,
            "from_": _SLIDER_FROM,
            "to": _SLIDER_TO,
            "resolution": _SLIDER_RESOLUTION,
            "length": 140,
        }

        # -- Bypass Strength selector
        tk.Label(controls, text="Bypass Strength", **lbl_style).pack(
            anchor="w", pady=(14, 2)
        )
        for strength in _BYPASS_PRESETS:
            tk.Radiobutton(
                controls,
                text=strength,
                variable=self._bypass_strength,
                value=strength,
                command=self._on_bypass_change,
                bg=_DARK_BG,
                fg=_TEXT_FG,
                selectcolor=_PANEL_BG,
                activebackground=_DARK_BG,
                activeforeground=_ACCENT,
                highlightthickness=0,
                anchor="w",
            ).pack(fill=tk.X)

        tk.Frame(controls, bg=_BORDER, height=1).pack(fill=tk.X, pady=(8, 0))

        # -- Merge rate
        tk.Label(controls, text="Merge Rate", **lbl_style).pack(
            anchor="w", pady=(8, 0)
        )
        self._merge_val_lbl = tk.Label(
            controls, textvariable=self._merge_rate, **val_style
        )
        self._merge_val_lbl.pack(anchor="e")
        self._merge_slider = tk.Scale(
            controls,
            variable=self._merge_rate,
            command=lambda _v: self._update_slider_labels(),
            **slider_style,
        )
        self._merge_slider.pack(fill=tk.X)

        tk.Frame(controls, bg=_BORDER, height=1).pack(fill=tk.X, pady=(8, 0))

        # -- Adversarial Mode checkbox
        tk.Label(controls, text="Evasion Options", **lbl_style).pack(
            anchor="w", pady=(8, 2)
        )
        tk.Checkbutton(
            controls,
            text="Adversarial Mode",
            variable=self._adversarial_mode,
            bg=_DARK_BG,
            fg=_TEXT_FG,
            selectcolor=_PANEL_BG,
            activebackground=_DARK_BG,
            activeforeground=_ACCENT,
            highlightthickness=0,
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            controls,
            text="(ZWS + homoglyphs)",
            bg=_DARK_BG,
            fg=_MUTED_FG,
            font=("Helvetica", 8),
            anchor="w",
        ).pack(anchor="w")

        # Apply initial preset to sliders
        self._on_bypass_change()

        # -- Humanize button
        self._humanize_btn = _HoverButton(
            controls,
            text="▶  Humanize",
            command=self._run_humanize,
            bg=_ACCENT,
            fg="#ffffff",
            activeforeground="#ffffff",
            font=("Helvetica", 12, "bold"),
            padx=12,
            pady=8,
            width=14,
        )
        self._humanize_btn.pack(pady=(30, 6), fill=tk.X)

        # -- Clear button
        _HoverButton(
            controls,
            text="⌫  Clear",
            command=self._clear_all,
            bg=_PANEL_BG,
            fg=_TEXT_FG,
            activeforeground=_TEXT_FG,
            font=("Helvetica", 10),
            padx=8,
            pady=6,
            width=14,
        ).pack(fill=tk.X)

        # -- Copy button
        _HoverButton(
            controls,
            text="⎘  Copy Result",
            command=self._copy_result,
            bg=_PANEL_BG,
            fg=_TEXT_FG,
            activeforeground=_TEXT_FG,
            font=("Helvetica", 10),
            padx=8,
            pady=6,
            width=14,
        ).pack(pady=(6, 0), fill=tk.X)

        # ---- Output ----
        tk.Label(
            body, text="Humanized Output", bg=_DARK_BG, fg=_TEXT_FG, anchor="w"
        ).grid(row=0, column=2, sticky="ew")

        output_frame = tk.Frame(body, bg=_BORDER, bd=0)
        output_frame.grid(row=1, column=2, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self._output_text = tk.Text(
            output_frame,
            wrap=tk.WORD,
            bg=_PANEL_BG,
            fg=_TEXT_FG,
            insertbackground=_TEXT_FG,
            selectbackground=_ACCENT,
            relief=tk.FLAT,
            bd=8,
            font=("Helvetica", 11),
            state=tk.DISABLED,
        )
        self._output_text.grid(row=0, column=0, sticky="nsew")

        out_scroll = ttk.Scrollbar(
            output_frame, orient=tk.VERTICAL, command=self._output_text.yview
        )
        out_scroll.grid(row=0, column=1, sticky="ns")
        self._output_text.configure(yscrollcommand=out_scroll.set)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self._root, bg=_PANEL_BG, pady=4)
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        tk.Label(
            bar,
            textvariable=self._status_text,
            bg=_PANEL_BG,
            fg=_MUTED_FG,
            anchor="w",
            padx=_PAD,
        ).grid(row=0, column=0, sticky="w")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _clear_placeholder(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Remove placeholder text on first focus."""
        current = self._input_text.get("1.0", tk.END).strip()
        if current == _PLACEHOLDER_INPUT.strip():
            self._input_text.delete("1.0", tk.END)
            self._input_text.configure(fg=_TEXT_FG)

    def _update_slider_labels(self) -> None:
        self._merge_val_lbl.configure(
            text=f"{self._merge_rate.get():.2f}"
        )

    def _on_bypass_change(self) -> None:
        """Sync sliders to the selected preset; disable them for non-Custom modes."""
        strength = self._bypass_strength.get()
        preset = _BYPASS_PRESETS.get(strength)
        if preset is not None:
            # Apply preset rates to sliders
            self._merge_rate.set(preset["merge_rate"])
            # Disable slider – rate is managed by the preset
            self._merge_slider.configure(state=tk.DISABLED)
        else:
            # Custom: let the user control the slider freely
            self._merge_slider.configure(state=tk.NORMAL)
        self._update_slider_labels()

    def _run_humanize(self) -> None:
        """Kick off humanisation in a background thread to keep UI responsive."""
        text = self._input_text.get("1.0", tk.END).strip()
        if not text or text == _PLACEHOLDER_INPUT.strip():
            messagebox.showwarning(
                "No input", "Please paste some text into the Input field first."
            )
            return

        strength = self._bypass_strength.get()
        preset = _BYPASS_PRESETS.get(strength)
        passes = int(preset["passes"]) if preset is not None else 1
        pass_label = HumanizerApp._format_pass_label(passes)

        self._humanize_btn.configure(state=tk.DISABLED, text="⏳  Processing…")
        self._status_text.set(
            f"Running humanizer pipeline… ({strength}, {pass_label})"
        )
        self._set_output("")

        # daemon=True: humanize() is pure computation (no I/O, no locks), so
        # letting it die with the main window is safe and prevents the process
        # from hanging when the user closes the app mid-run.
        thread = threading.Thread(target=self._humanize_worker, args=(text,), daemon=True)
        thread.start()

    def _humanize_worker(self, text: str) -> None:
        """Run humanize() (possibly multiple passes) in a background thread."""
        try:
            strength = self._bypass_strength.get()
            preset = _BYPASS_PRESETS.get(strength)
            if preset is not None:
                passes = int(preset["passes"])
                merge_rate = float(preset["merge_rate"])
            else:
                passes = 1
                merge_rate = self._merge_rate.get()

            adversarial = self._adversarial_mode.get()
            current_text = text
            result = None
            total_markers_removed = 0
            total_sentences_merged = 0

            for _ in range(passes):
                result = humanize(
                    current_text,
                    merge_rate=merge_rate,
                    adversarial_mode=adversarial,
                )
                total_markers_removed += result.markers_removed
                total_sentences_merged += result.sentences_merged
                current_text = result.humanized_text

            self._root.after(
                0,
                self._on_success,
                result,
                passes,
                total_markers_removed,
                total_sentences_merged,
            )
        except Exception as exc:  # noqa: BLE001
            self._root.after(0, self._on_error, exc)

    @staticmethod
    def _format_pass_label(passes: int) -> str:
        return f"{passes} pass{'es' if passes > 1 else ''}"

    def _on_success(
        self,
        result: object,
        passes: int = 1,
        total_markers_removed: int = 0,
        total_sentences_merged: int = 0,
    ) -> None:  # result is HumanizeResult
        self._set_output(result.humanized_text)  # type: ignore[attr-defined]
        pass_label = self._format_pass_label(passes)
        stats = (
            f"Done ✓  |  {pass_label}  |  Words: {result.original_word_count} → "  # type: ignore[attr-defined]
            f"{result.humanized_word_count}  |  "
            f"Markers removed: {total_markers_removed}  |  "
            f"Sentences merged: {total_sentences_merged}"
        )
        self._status_text.set(stats)
        self._humanize_btn.configure(state=tk.NORMAL, text="▶  Humanize")

    def _on_error(self, exc: Exception) -> None:
        self._status_text.set(f"Error: {exc}")
        messagebox.showerror("Humanizer Error", str(exc))
        self._humanize_btn.configure(state=tk.NORMAL, text="▶  Humanize")

    def _set_output(self, text: str) -> None:
        self._output_text.configure(state=tk.NORMAL)
        self._output_text.delete("1.0", tk.END)
        if text:
            self._output_text.insert("1.0", text)
        self._output_text.configure(state=tk.DISABLED)

    def _clear_all(self) -> None:
        self._input_text.delete("1.0", tk.END)
        self._input_text.configure(fg=_MUTED_FG)
        self._input_text.insert("1.0", _PLACEHOLDER_INPUT)
        self._set_output("")
        self._status_text.set("Ready.")

    def _copy_result(self) -> None:
        text = self._output_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Nothing to copy", "Run the humanizer first.")
            return
        self._root.clipboard_clear()
        self._root.clipboard_append(text)
        self._root.update()
        self._status_text.set("Result copied to clipboard.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Tkinter GUI (entry point for ``aip-gui`` console script)."""
    root = tk.Tk()
    _style_ttk(root)
    HumanizerApp(root)
    root.mainloop()


def _style_ttk(root: tk.Tk) -> None:
    """Apply dark-theme styling to ttk widgets (scrollbars)."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "Vertical.TScrollbar",
        background=_BORDER,
        troughcolor=_PANEL_BG,
        bordercolor=_PANEL_BG,
        arrowcolor=_MUTED_FG,
        relief=tk.FLAT,
    )


if __name__ == "__main__":
    main()
