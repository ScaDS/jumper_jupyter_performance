"""The half of the harness that runs inside the kernel.

Everything else in this package observes the experiment from outside. These
modules are imported by cells the runner injects into the usecase notebook, so
they live next to the reviewer, in its process, and they keep to the
extension's own dependencies - no Hydra, no plotting stack.

Their job is narrow on purpose: watch the magic run, then write down what it
produced. They never steer it. The only thing installed into the reviewer is a
callback that reads token counts and latency off model calls, because those are
the one thing the reviewer does not record anywhere.
"""
from jumper_ablations.runtime.session import (
    begin,
    bootstrap,
    capture,
    current_session,
    resolve_target,
)

__all__ = ["begin", "bootstrap", "capture", "current_session", "resolve_target"]
