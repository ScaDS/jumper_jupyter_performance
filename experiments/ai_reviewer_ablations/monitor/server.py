"""A small read-only HTTP view of a run directory.

Two things this deliberately is not. It is not a framework - the standard
library serves a page and some JSON perfectly well, and a monitor that needs
its own dependency tree is one more thing to install on a cluster before the
thing it monitors can be watched. And it is not a writer: it opens files, it
runs `squeue`, and that is the whole of its authority over a sweep costing
days of machine time.

It binds to the loopback address only. A login node is shared, and a run
directory holds the prompts and answers of every generation.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from monitor import model, slurm

STATIC = Path(__file__).parent / "static"
DEFAULT_PORT = 8765


class MonitorHandler(BaseHTTPRequestHandler):
    """Routes. Everything is a GET; nothing changes state."""

    results_root: Path = Path("results")

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        route = urlparse(self.path)
        query = parse_qs(route.query)
        try:
            if route.path in ("/", "/index.html"):
                self._send_static("index.html")
            elif route.path.startswith("/static/"):
                self._send_static(route.path[len("/static/") :])
            elif route.path == "/api/runs":
                self._send_json({"runs": model.list_runs(self.results_root)})
            elif route.path == "/api/aggregate":
                names = [
                    one
                    for one in (query.get("runs") or [""])[0].split(",")
                    if one
                ]
                self._send_json(model.aggregate(self.results_root, names))
            elif route.path == "/api/join":
                self._send_json(
                    model.join(
                        self.results_root,
                        mine=self._run_name(query, "mine"),
                        theirs=self._run_name(query, "theirs"),
                        variant=(query.get("variant") or [""])[0],
                    )
                )
            elif route.path == "/api/snapshot":
                self._send_json(self._snapshot(query))
            elif route.path == "/api/record":
                self._send_json(self._record(query))
            else:
                self._send_error(404, f"no route {route.path}")
        except FileNotFoundError as missing:
            self._send_error(404, str(missing))
        except Exception as failure:  # the page must survive a bad run
            self._send_error(500, f"{type(failure).__name__}: {failure}")

    # -- routes ----------------------------------------------------------

    def _snapshot(self, query: dict) -> dict:
        run = self._run_path(query)
        invocations = model.read_invocations(run)
        job_ids = [
            entry.get("slurm_job_id")
            for entry in invocations
            if entry.get("slurm_job_id")
        ]
        return model.snapshot(run, slurm.job_states(job_ids))

    def _record(self, query: dict) -> dict:
        """One record in full, fetched only when a person opens it.

        The snapshot carries a summary of every record because those are the
        numbers a sweep is watched for. The bodies are a different matter:
        each holds the verbatim messages sent to the model, and shipping all
        of them on every refresh would trade a readable page for a slow one.
        """
        run = self._run_path(query)
        record_id = (query.get("id") or [""])[0]
        if not record_id or "/" in record_id or ".." in record_id:
            raise FileNotFoundError(f"bad record id {record_id!r}")
        path = run / "records" / f"{record_id}.json"
        body = model._load_json(path, None)
        if body is None:
            raise FileNotFoundError(f"no record {record_id}")
        return body

    def _run_path(self, query: dict) -> Path:
        name = (query.get("run") or [""])[0]
        if not name:
            runs = model.list_runs(self.results_root)
            if not runs:
                raise FileNotFoundError(f"no runs under {self.results_root}")
            return Path(runs[0]["path"])
        candidate = (self.results_root / name).resolve()
        root = self.results_root.resolve()
        # A run is named, never pathed: the query must not be able to walk
        # out of the results directory.
        if root not in candidate.parents and candidate != root:
            raise FileNotFoundError(f"run {name!r} is not under {root}")
        if not candidate.is_dir():
            raise FileNotFoundError(f"no run {name!r}")
        return candidate

    def _run_name(self, query: dict, field: str) -> str:
        """A run named by one query field, checked the same way as `run`.

        The join takes two names, and the model resolves them against the
        results root itself, so what has to be rejected here is a name that
        would leave it.
        """
        name = (query.get(field) or [""])[0]
        if not name:
            raise FileNotFoundError(f"no {field} run named")
        return self._run_path({"run": [name]}).name

    # -- plumbing --------------------------------------------------------

    def _send_json(self, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"no asset {name}")
        body = path.read_bytes()
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # One line per request on stderr would bury the address the person
        # actually needs; failures still surface in the page.
        return


def serve(results_root: Path, port: int, host: str = "127.0.0.1") -> None:
    MonitorHandler.results_root = Path(results_root)
    server = ThreadingHTTPServer((host, port), MonitorHandler)
    runs = model.list_runs(Path(results_root))
    print(f"watching {results_root} ({len(runs)} run(s))")
    print(f"open http://{host}:{port}/")
    print(
        "from your own machine:\n"
        f"  ssh -N -L {port}:localhost:{port} "
        "<user>@login2.romeo.hpc.tu-dresden.de"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monitor",
        description="Watch an ablation run without being able to disturb it.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Directory holding the runs (default: $JUMPER_ABLATION_STORAGE"
        "/results, else ./results)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback by default, and changing that publishes every prompt "
        "and answer in the run to everyone on the node",
    )
    arguments = parser.parse_args(argv)

    results = Path(arguments.results or _default_results())
    if not results.is_dir():
        print(f"no results directory at {results}", file=sys.stderr)
        return 1
    serve(results, arguments.port, arguments.host)
    return 0


def _default_results() -> Path:
    import os

    storage = os.environ.get("JUMPER_ABLATION_STORAGE")
    if storage:
        return Path(storage) / "results"
    return Path(__file__).resolve().parents[1] / "results"


if __name__ == "__main__":
    raise SystemExit(main())
