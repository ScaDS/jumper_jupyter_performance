"""A read-only web view of an ablation run.

Run it from the experiment root:

    python -m monitor --port 8765

then forward the port from your own machine and open localhost:8765. It
reads files and asks the queue; it never writes to a run.
"""
