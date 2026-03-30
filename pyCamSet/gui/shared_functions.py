"""
Shared utilities, mixins, widget factories, and constants for the pyCamSet GUI.

Conventions
-----------
- All tkinter widgets are created via the factory helpers at module bottom.
- Hover tooltips are gated by a single shared ``tk.BooleanVar`` (``info_var``).
- The terminal pane is a dark ``tk.Text`` in DISABLED state; writes go through
  :meth:`TerminalMixin.terminal_append`.
- :class:`WorkspaceManager` owns all disk I/O; nothing else writes to the
  workspace directory.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Tab name constants  — used for cross-tab navigation
# ---------------------------------------------------------------------------

TAB_PHASE0 = "Phase 0"
TAB_PHASE0_DIAG = "Phase 0 Diagnostics"
TAB_PHASE1 = "Phase 1"
TAB_PHASE1_DIAG = "Phase 1 Diagnostics"


# ---------------------------------------------------------------------------
# Info-hover mixin
# ---------------------------------------------------------------------------

class InfoHoverMixin:
    """Mixin that gates tooltip/info-hover pop-ups via a shared ``BooleanVar``.

    Call :meth:`init_hover` once during ``__init__``, then use
    :meth:`bind_hover` on any widget you want to annotate.
    """

    def init_hover(self, info_var: tk.BooleanVar) -> None:
        """Store the shared toggle and reset the active tooltip window."""
        self._info_var: tk.BooleanVar = info_var
        self._tooltip_win: Optional[tk.Toplevel] = None

    def bind_hover(self, widget: tk.Widget, text: str) -> None:
        """Bind ``<Enter>``/``<Leave>`` events to show/hide a tooltip."""
        widget.bind("<Enter>", lambda e, t=text: self._show_tooltip(e, t))
        widget.bind("<Leave>", lambda _e: self._hide_tooltip())

    # ------------------------------------------------------------------

    def _show_tooltip(self, event: tk.Event, text: str) -> None:
        if not getattr(self, "_info_var", None) or not self._info_var.get():
            return
        self._hide_tooltip()
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 20
        win = tk.Toplevel()
        win.wm_overrideredirect(True)
        win.wm_geometry(f"+{x}+{y}")
        tk.Label(
            win,
            text=text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=280,
            justify="left",
            padx=4,
            pady=2,
        ).pack()
        self._tooltip_win = win

    def _hide_tooltip(self) -> None:
        if self._tooltip_win:
            try:
                self._tooltip_win.destroy()
            except tk.TclError:
                pass
            self._tooltip_win = None


# ---------------------------------------------------------------------------
# Terminal pane mixin
# ---------------------------------------------------------------------------

class TerminalMixin:
    """Mixin that provides a scrollable, dark terminal pane with show/hide.

    Call :meth:`init_terminal` once, then use :meth:`terminal_append` and
    :meth:`terminal_clear` to write to it.  The pane honours the shared
    ``show_var`` ``BooleanVar`` and hides/shows itself when it changes.
    """

    def init_terminal(self, parent: tk.Widget, show_var: tk.BooleanVar) -> tk.Frame:
        """Build the terminal frame and return it (already packed conditionally).

        :param parent: The containing widget the terminal frame is packed into.
        :param show_var: Shared BooleanVar controlling visibility.
        """
        self._terminal_frame = tk.Frame(parent, bg="#1e1e1e")
        self._show_terminal_var = show_var

        sb = ttk.Scrollbar(self._terminal_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._terminal_text = tk.Text(
            self._terminal_frame,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Courier", 10),
            wrap=tk.WORD,
            yscrollcommand=sb.set,
            height=8,
            selectbackground="#264f78",
        )
        self._terminal_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self._terminal_text.yview)

        show_var.trace_add("write", lambda *_: self._on_show_terminal_change())
        self._on_show_terminal_change()
        return self._terminal_frame

    def terminal_append(self, text: str) -> None:
        """Append *text* followed by a newline to the terminal pane."""
        w = getattr(self, "_terminal_text", None)
        if w is None:
            return
        w.config(state=tk.NORMAL)
        w.insert(tk.END, text + "\n")
        w.see(tk.END)
        w.config(state=tk.DISABLED)

    def terminal_clear(self) -> None:
        """Erase all content from the terminal pane."""
        w = getattr(self, "_terminal_text", None)
        if w is None:
            return
        w.config(state=tk.NORMAL)
        w.delete("1.0", tk.END)
        w.config(state=tk.DISABLED)

    def _on_show_terminal_change(self) -> None:
        f = getattr(self, "_terminal_frame", None)
        if f is None:
            return
        if getattr(self, "_show_terminal_var", None) and self._show_terminal_var.get():
            f.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0, 4))
        else:
            f.pack_forget()


# ---------------------------------------------------------------------------
# Workspace manager
# ---------------------------------------------------------------------------

class WorkspaceManager:
    """Manages the ``<dataset>/.pycamset_workspace`` directory and run metadata.

    Directory layout::

        <workspace>/
          phase0_runs/<YYYYMMDD_HHMMSS_<hex>>/metadata.json
          phase1_runs/<YYYYMMDD_HHMMSS_<hex>>/metadata.json
          handoff.json

    :param workspace_path: Root of the workspace.  Created on first use.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = Path(workspace_path)
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create the standard sub-directories if they do not exist."""
        for sub in ("phase0_runs", "phase1_runs"):
            (self.workspace_path / sub).mkdir(parents=True, exist_ok=True)

    def save_run(self, phase: str, run_id: str, metadata: dict) -> Path:
        """Persist *metadata* to ``<workspace>/<phase>_runs/<run_id>/metadata.json``.

        :param phase: ``"phase0"`` or ``"phase1"``.
        :param run_id: Unique identifier (from :func:`make_run_id`).
        :param metadata: JSON-serialisable dict of parameters and diagnostics.
        :returns: Path to the written file.
        """
        run_dir = self.workspace_path / f"{phase}_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        return meta_path

    def load_runs(self, phase: str) -> list[dict]:
        """Return all saved runs for *phase* sorted oldest-first.

        :param phase: ``"phase0"`` or ``"phase1"``.
        :returns: List of metadata dicts (may be empty).
        """
        runs_dir = self.workspace_path / f"{phase}_runs"
        if not runs_dir.exists():
            return []
        results: list[dict] = []
        for run_dir in sorted(runs_dir.iterdir()):
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as fh:
                        data = json.load(fh)
                    data.setdefault("run_id", run_dir.name)
                    results.append(data)
                except (json.JSONDecodeError, OSError):
                    pass
        return results

    def write_handoff(self, payload: dict) -> None:
        """Write *payload* to ``<workspace>/handoff.json``.

        Used by "Continue to Next Phase" to pass selected run(s) downstream.
        """
        with open(self.workspace_path / "handoff.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=str)


def make_run_id() -> str:
    """Return a unique, timestamp-ordered run identifier.

    Format: ``YYYYMMDD_HHMMSS_<6-char hex>``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Run selector widget
# ---------------------------------------------------------------------------

class RunSelectorFrame(tk.Frame):
    """Listbox-based multi-select widget that shows available saved runs.

    The most recent ``min(3, n)`` runs are pre-selected on every :meth:`refresh`.

    :param parent: Parent widget.
    :param runs: Initial list of run-metadata dicts.
    :param on_select: Optional callback invoked with ``list[dict]`` when the
        selection changes.
    """

    def __init__(
        self,
        parent: tk.Widget,
        runs: list[dict],
        on_select: Optional[Callable[[list[dict]], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._on_select = on_select
        self._runs: list[dict] = []

        tk.Label(self, text="Saved runs", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

        list_frame = tk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=sb.set,
            height=10,
            exportselection=False,
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self._empty_label = tk.Label(self, text="No runs saved yet.", fg="gray")
        self.refresh(runs)

    def _on_listbox_select(self, _event: Optional[tk.Event] = None) -> None:
        if self._on_select:
            self._on_select(self.get_selected())

    def get_selected(self) -> list[dict]:
        """Return the currently selected run-metadata dicts."""
        return [self._runs[i] for i in self._listbox.curselection()]

    def refresh(self, runs: list[dict]) -> None:
        """Repopulate the listbox with *runs* and pre-select most recent 1–3."""
        self._runs = runs
        self._listbox.delete(0, tk.END)
        if not runs:
            self._listbox.pack_forget()
            self._empty_label.pack(anchor="w")
            return
        self._empty_label.pack_forget()
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for run in runs:
            self._listbox.insert(tk.END, run.get("run_id", str(run)))
        # pre-select the most recent min(3, n) entries
        n = len(runs)
        for i in range(max(0, n - 3), n):
            self._listbox.selection_set(i)


# ---------------------------------------------------------------------------
# Widget factory helpers
# ---------------------------------------------------------------------------

def make_section_label(parent: tk.Widget, text: str) -> tk.Label:
    """Return a bold section-header label (not yet placed in a layout)."""
    return tk.Label(parent, text=text, font=("TkDefaultFont", 10, "bold"), anchor="w")


def make_labeled_entry(
    parent: tk.Widget,
    label: str,
    default: str = "",
    width: int = 40,
) -> tuple[tk.Label, ttk.Entry, tk.StringVar]:
    """Return ``(label_widget, entry_widget, string_var)`` — not yet placed."""
    lbl = tk.Label(parent, text=label, anchor="w")
    var = tk.StringVar(value=default)
    entry = ttk.Entry(parent, textvariable=var, width=width)
    return lbl, entry, var


def make_labeled_checkbox(
    parent: tk.Widget,
    label: str,
    default: bool = False,
) -> tuple[ttk.Checkbutton, tk.BooleanVar]:
    """Return ``(checkbutton, bool_var)`` — not yet placed."""
    var = tk.BooleanVar(value=default)
    cb = ttk.Checkbutton(parent, text=label, variable=var)
    return cb, var


def make_labeled_spinbox(
    parent: tk.Widget,
    label: str,
    from_: int,
    to: int,
    default: int,
) -> tuple[tk.Label, ttk.Spinbox, tk.IntVar]:
    """Return ``(label_widget, spinbox_widget, int_var)`` — not yet placed."""
    lbl = tk.Label(parent, text=label, anchor="w")
    var = tk.IntVar(value=default)
    spinbox = ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=8)
    return lbl, spinbox, var


def make_orange_button(parent: tk.Widget, text: str, command: Callable) -> tk.Button:
    """Return an orange ``tk.Button`` — not yet placed."""
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg="#e07b00",
        fg="white",
        activebackground="#c06000",
        activeforeground="white",
        relief="raised",
        font=("TkDefaultFont", 10, "bold"),
        cursor="hand2",
    )


def make_continue_button(parent: tk.Widget, command: Callable) -> tk.Button:
    """Return a green "Continue to Next Phase" ``tk.Button`` — not yet placed."""
    return tk.Button(
        parent,
        text="Continue to Next Phase ▶",
        command=command,
        bg="#2e7d32",
        fg="white",
        activebackground="#1b5e20",
        activeforeground="white",
        relief="raised",
        font=("TkDefaultFont", 10, "bold"),
        cursor="hand2",
    )
