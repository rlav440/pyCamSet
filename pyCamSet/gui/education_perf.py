"""Hardware and platform policy for education-mode drawing: when to use the cheaper
settings (macOS by default, weak hardware), the MSAA sample count, the canvas
device-pixel-ratio cap and the playback frame-rate floor, in one place.
and the phase 2-4 figure canvases.

Every probe here is cheap and never imports a new hard dependency: ``psutil`` is used only
when it happens to be installed, otherwise the RAM figure comes from the operating system
(POSIX ``sysconf``, Windows ``GlobalMemoryStatusEx``) or is unknown.
"""
from __future__ import annotations

import math
import os
import sys
import weakref
from typing import Any

#: Logical CPUs at or below this count are treated as weak hardware (a dual-core machine
#: without hyper-threading, or fewer).
WEAK_CPU_COUNT = 2
#: Physical RAM below this many bytes is treated as weak hardware.
WEAK_RAM_BYTES = 6 * 2**30
#: MSAA samples for the embedded PyVista view, normal and reduced.
MSAA_SAMPLES = 8
#: 0 means no multisampling: the view uses FXAA instead, which costs a fraction as much.
MSAA_SAMPLES_LOW = 0
#: Mesh-resolution factor for rounded 3D primitives on weak hardware and macOS.
MESH_DETAIL_LOW = 0.6
#: A GL renderer whose name contains one of these is software rendering: no GPU.
SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "software", "gdi generic")
#: Lowest OpenGL version the embedded view needs (VTK 9 asks for 3.2 core).
MIN_GL_VERSION = (3, 2)
#: Median time (ms) of a full-frame clear in the probe above which the GPU is too slow.
MAX_CLEAR_MS = 40.0
#: One render slower than this (seconds) is a stall; this many in a row switch the view off.
SLOW_RENDER_SECONDS = 1.0
SLOW_RENDER_STRIKES = 3
#: Largest device-pixel ratio a canvas uses on weak hardware (2.0 on Retina is 4x the pixels).
DPR_CAP_WEAK = 1.5
#: Physical pixels a single education canvas may use before its ratio is lowered.
PIXEL_BUDGET = 6_000_000
#: Shortest playback tween tick (ms) on a normal machine, and on weak hardware or macOS.
TWEEN_FLOOR_MS = 16
TWEEN_FLOOR_LOW_MS = 33
#: Shortest interval (ms) between two 2D side-panel redraws while a solver frame plays.
REDRAW_INTERVAL_MS = 50
#: Shortest interval (ms) between two matplotlib 3D redraws while the scene is animating.
STATIC_DRAW_INTERVAL_MS = 33


def is_macos() -> bool:
    """True on macOS (read at call time, so tests can patch ``sys.platform``)."""
    return sys.platform == "darwin"


def logical_cpu_count() -> int:
    """Logical CPUs reported by the OS (1 when the OS does not say)."""
    count = os.cpu_count()
    return int(count) if count else 1


def physical_ram_bytes() -> int | None:
    """Physical RAM in bytes, or None when no probe can tell. Never raises."""
    try:  # optional: psutil is not a declared dependency
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - absent or failing psutil is normal
        pass
    try:  # POSIX (Linux; macOS on recent Pythons)
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return int(pages) * int(size)
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
        except Exception:  # noqa: BLE001 - the probe is advisory only
            pass
    return None


def weak_hardware_reason() -> str | None:
    """A short description of why this machine counts as weak, or None when it does not."""
    cpus = logical_cpu_count()
    ram = physical_ram_bytes()
    reasons = []
    if cpus <= WEAK_CPU_COUNT:
        reasons.append(f"{cpus} logical CPU{'' if cpus == 1 else 's'}")
    if ram is not None and ram < WEAK_RAM_BYTES:
        reasons.append(f"{ram / 2**30:.1f} GiB RAM")
    return ", ".join(reasons) or None


def is_weak_hardware() -> bool:
    """True when :func:`weak_hardware_reason` finds a reason."""
    return weak_hardware_reason() is not None


def reduced_settings() -> bool:
    """True when the cheaper drawing settings apply: weak hardware or macOS."""
    return is_macos() or is_weak_hardware()


def msaa_samples() -> int:
    """Multisample anti-aliasing samples for the embedded 3D view."""
    return MSAA_SAMPLES_LOW if reduced_settings() else MSAA_SAMPLES


def mesh_detail() -> float:
    """Factor (1.0 or :data:`MESH_DETAIL_LOW`) applied to the resolution of rounded meshes."""
    return MESH_DETAIL_LOW if reduced_settings() else 1.0


def scaled_resolution(n: int, minimum: int = 6) -> int:
    """``n`` mesh divisions, reduced by :func:`mesh_detail` (never below ``minimum``)."""
    return max(int(minimum), int(round(n * mesh_detail())))


def configure_surface_format() -> bool:
    """On macOS, ask Qt for an OpenGL 4.1 core context before the application exists.

    macOS gives a legacy 2.1 context unless a core profile is requested, and 4.1 is the most
    it offers. Must run before ``QApplication`` is created; returns True when it set the
    default format. Does nothing on other platforms or once an application exists.
    """
    if not is_macos():
        return False
    try:
        from PySide6.QtGui import QGuiApplication, QSurfaceFormat
        if QGuiApplication.instance() is not None:
            return False
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:  # noqa: BLE001 - advisory: the probe reports a context that is unusable
        return False
    return True


def parse_gl_version(text: str | None) -> tuple[int, int] | None:
    """(major, minor) from a GL_VERSION string such as '4.1 Metal - 88' or '3.3.0 NVIDIA'."""
    import re
    match = re.match(r"\s*(?:OpenGL ES\s+)?(\d+)\.(\d+)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def gl_verdict(info: dict[str, Any]) -> str | None:
    """Why a working GL context is still unsuitable for the embedded view, or None when fine.

    Looks at the probe's renderer string (software rendering), the version (below
    :data:`MIN_GL_VERSION`) and the timed clear (``clear_ms`` above :data:`MAX_CLEAR_MS`).
    Missing fields are not held against the machine.
    """
    renderer = str(info.get("renderer") or "")
    if any(name in renderer.casefold() for name in SOFTWARE_RENDERERS):
        return f"software rendering ({renderer})"
    version = parse_gl_version(info.get("version"))
    if version is not None and version < MIN_GL_VERSION:
        return (f"OpenGL {version[0]}.{version[1]} is older than "
                f"{MIN_GL_VERSION[0]}.{MIN_GL_VERSION[1]}")
    clear_ms = info.get("clear_ms")
    if isinstance(clear_ms, (int, float)) and clear_ms > MAX_CLEAR_MS:
        return f"a full-frame clear took {clear_ms:.0f} ms"
    return None


# -- live safety net: too many slow renders switch the embedded view off for new scenes ----------

_slow_reason: str | None = None
_slow_strikes = 0


def note_render_time(seconds: float) -> bool:
    """Record how long one embedded-3D render took. After :data:`SLOW_RENDER_STRIKES` slow
    ones in a row, remember that the view is too slow here and return True (once)."""
    global _slow_reason, _slow_strikes
    if seconds < SLOW_RENDER_SECONDS:
        _slow_strikes = 0
        return False
    _slow_strikes += 1
    if _slow_strikes >= SLOW_RENDER_STRIKES and _slow_reason is None:
        _slow_reason = (f"{SLOW_RENDER_STRIKES} renders in a row took over "
                        f"{SLOW_RENDER_SECONDS:g} s")
        return True
    return False


def too_slow_reason() -> str | None:
    """Why the embedded view was switched off during this session, or None."""
    return _slow_reason


def reset_slow_state() -> None:
    """Forget slow renders (tests)."""
    global _slow_reason, _slow_strikes
    _slow_reason, _slow_strikes = None, 0


def tween_floor_ms() -> int:
    """Shortest playback tween tick, in milliseconds."""
    return TWEEN_FLOOR_LOW_MS if reduced_settings() else TWEEN_FLOOR_MS


def pixel_ratio_cap(width: float, height: float, ratio: float) -> float:
    """The device-pixel ratio a canvas of ``width`` x ``height`` logical pixels may use.

    Unchanged on a normal machine while the canvas fits :data:`PIXEL_BUDGET`. Weak hardware
    is capped at :data:`DPR_CAP_WEAK`; a canvas whose physical size would exceed the budget
    gets the largest ratio that fits it (never below 1). Text on a 1x display is untouched.
    """
    ratio = float(ratio or 1.0)
    if ratio <= 1.0:
        return 1.0
    cap = ratio
    if is_weak_hardware():
        cap = min(cap, DPR_CAP_WEAK)
    area = float(width) * float(height)
    if area > 0 and area * ratio * ratio > PIXEL_BUDGET:
        cap = min(cap, max(1.0, math.sqrt(PIXEL_BUDGET / area)))
    return cap


def cap_canvas_pixel_ratio(canvas: Any) -> None:
    """Route a matplotlib Qt canvas's device-pixel ratio through :func:`pixel_ratio_cap`.

    The Qt backend sets the ratio through ``FigureCanvasBase._set_device_pixel_ratio`` (the
    hook matplotlib documents for high-DPI subclasses); this shadows that method on the
    instance, so the canvas renders at the capped ratio and Qt scales the result. The budget
    is checked each time the backend asks for a ratio (creation and screen changes), so it
    does not follow later resizes. Calling it twice is harmless.
    """
    if getattr(canvas, "_edu_dpr_capped", False) or not hasattr(type(canvas),
                                                               "_set_device_pixel_ratio"):
        return
    # Hold the canvas weakly and the plain class method (not a bound method): the canvas then
    # keeps no reference cycle through its own attribute, so closing a view frees its widgets.
    canvas_ref = weakref.ref(canvas)
    base_method = type(canvas)._set_device_pixel_ratio

    def capped(ratio: float) -> bool:
        target = canvas_ref()
        if target is None:
            return False
        return base_method(target, pixel_ratio_cap(target.width(), target.height(), float(ratio)))

    canvas._set_device_pixel_ratio = capped
    canvas._edu_dpr_capped = True
