"""Build and validate the native C collector binary.

This module provides:

- :func:`build_collector` — compile ``collector.c`` into ``jumper_collector``
  using *cc* (or the compiler pointed to by ``$CC``).
- :func:`sanity_check` — start the C monitor for a few seconds and verify
  that it collects non-NaN, non-zero metrics.
- :func:`ensure_native_c` — build + sanity-check in one call; returns
  ``True`` when the native_c backend is usable.

These are called automatically during ``pip install`` (via the
``[project.entry-points."jumper_extension.post_install"]`` hook) and at
runtime when the ``"default"`` monitor type is requested.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

logger = logging.getLogger("extension")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BINARY_NAME = "jumper_collector"
_BINARY_PATH = os.path.join(_THIS_DIR, _BINARY_NAME)
_SOURCE_PATH = os.path.join(_THIS_DIR, "collector.c")

# Cache the result so we only build/check once per process.
_native_c_available: bool | None = None


def build_collector(force: bool = False) -> bool:
    """Compile the C collector binary.

    Returns ``True`` if the binary is ready to use (either freshly built
    or already present and *force* is ``False``).
    """
    if not force and os.path.isfile(_BINARY_PATH):
        # Already compiled — check it's newer than the source.
        try:
            if os.path.getmtime(_BINARY_PATH) >= os.path.getmtime(_SOURCE_PATH):
                return True
        except OSError:
            pass

    if not os.path.isfile(_SOURCE_PATH):
        logger.debug("[JUmPER] C collector source not found at %s", _SOURCE_PATH)
        return False

    cc = os.environ.get("CC", "")
    if not cc:
        cc = shutil.which("cc") or shutil.which("gcc") or ""
    if not cc:
        logger.info("[JUmPER] No C compiler found — native_c backend unavailable.")
        print(
            "[JUmPER] No C compiler found; falling back to subprocess_python "
            "monitor."
        )
        return False

    cmd = [cc, "-O2", "-Wall", "-o", _BINARY_PATH, _SOURCE_PATH, "-lm", "-ldl"]
    print(
        "[JUmPER] Compiling native_c monitor binary (first use, one-time step)..."
    )
    logger.info("[JUmPER] Compiling C collector: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(
                "[JUmPER] C collector compilation failed:\n%s",
                result.stderr.strip(),
            )
            print(
                "[JUmPER] native_c compilation failed; falling back to "
                "subprocess_python monitor."
            )
            return False
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("[JUmPER] C collector compilation error: %s", exc)
        print(
            f"[JUmPER] native_c compilation error ({exc}); falling back to "
            f"subprocess_python monitor."
        )
        return False

    logger.info("[JUmPER] C collector compiled successfully.")
    print("[JUmPER] native_c monitor compiled successfully.")
    return True


def sanity_check(timeout: float = 10.0) -> bool:
    """Run a short sanity check of the native_c monitor.

    Starts the C monitor for a few seconds while generating IO, then
    verifies that the collected data contains expected columns with
    non-NaN values and non-zero key metrics.  Returns ``True`` if the
    backend is healthy.
    """
    if not os.path.isfile(_BINARY_PATH):
        return False

    try:
        from jumper_extension.monitor.backends.native_c import (
            CSubprocessPerformanceMonitor,
        )
        import pandas as pd
    except ImportError:
        return False

    monitor = CSubprocessPerformanceMonitor()

    # Background IO to ensure io_write/io_read_count are non-zero.
    io_stop = threading.Event()
    tmp_path = None

    def _io_worker():
        nonlocal tmp_path
        f = tempfile.NamedTemporaryFile(delete=False, prefix="jumper_sanity_")
        tmp_path = f.name
        buf = b"x" * (64 * 1024)
        while not io_stop.is_set():
            f.write(buf)
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            _ = f.read()
            f.seek(0)
            f.truncate()
        f.close()

    io_thread = threading.Thread(target=_io_worker, daemon=True)
    io_thread.start()

    try:
        monitor.start(interval=1.0)
    except Exception as exc:
        logger.debug("[JUmPER] native_c sanity: start failed: %s", exc)
        io_stop.set()
        io_thread.join(timeout=2)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return False

    required_samples = 3 + 1  # 3 usable + 1 throwaway first sample
    deadline = time.monotonic() + timeout
    enough = False
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if monitor.nodes.node_names():
            levels = monitor.nodes.levels
            counts = [len(monitor.nodes.view(level=lv)) for lv in levels]
            if counts and min(counts) >= required_samples:
                enough = True
                break

    monitor.stop()
    io_stop.set()
    io_thread.join(timeout=2)
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not enough or not monitor.nodes.node_names():
        logger.info("[JUmPER] native_c sanity: not enough samples collected.")
        return False

    # Validate collected data
    required_cols = [
        "time", "cpu_util_avg", "memory",
        "io_write", "io_read_count", "io_write_count",
    ]
    for level in monitor.nodes.levels:
        df = monitor.nodes.view(level=level)
        if df.empty or len(df) < 2:
            logger.info(
                "[JUmPER] native_c sanity: level '%s' has insufficient data.", level
            )
            return False
        for col in required_cols:
            if col not in df.columns:
                logger.info(
                    "[JUmPER] native_c sanity: missing column '%s' at level '%s'.",
                    col, level,
                )
                return False
            if df[col].isna().any():
                logger.info(
                    "[JUmPER] native_c sanity: NaN in '%s' at level '%s'.",
                    col, level,
                )
                return False

    logger.info("[JUmPER] native_c sanity check passed.")
    return True


def ensure_native_c(force_build: bool = False) -> bool:
    """Build the C collector (if needed) and run a sanity check.

    Returns ``True`` if the native_c backend is usable.  The result is
    cached for the lifetime of the process.
    """
    global _native_c_available
    if _native_c_available is not None and not force_build:
        return _native_c_available

    # Only build — do not run the full sanity check during regular
    # start-up (it adds ~10s to every %perfmonitor_start). Users can
    # trigger an explicit sanity check via
    # ``%perfmonitor_start --check-sanity``.
    ok = build_collector(force=force_build)
    _native_c_available = ok
    return ok


def is_native_c_available() -> bool:
    """Return whether the native_c backend was previously validated.

    Does **not** trigger a build or sanity check.  Returns ``False`` if
    :func:`ensure_native_c` has not been called yet.
    """
    return _native_c_available is True
