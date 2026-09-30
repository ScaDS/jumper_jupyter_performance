"""What the queue says about the jobs a run recorded.

Optional in the strict sense: every function here degrades to an empty answer
when there is no Slurm, no job ids, or the command fails. The monitor must
work against a finished run on a laptop, and a queue it cannot reach is not
an error worth showing a person watching a sweep.

It exists for one distinction the run directory cannot make. A shard that has
not started and a shard that died before writing anything look identical on
disk - no index line, no records - and the difference is the whole question
when a sweep looks stalled.
"""

from __future__ import annotations

import subprocess

# squeue reports what is queued or running; sacct reports what has ended.
# Both are asked, because a job that finished two minutes ago has left the
# first and only the second knows how it went.
_SQUEUE_FIELDS = "JobID,State,NodeList,TimeUsed"
_SACCT_FIELDS = "JobID,State,NodeList,Elapsed"

_RUNNING = {"RUNNING", "COMPLETING"}
_QUEUED = {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED"}
_FINISHED = {"COMPLETED"}


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def classify(state: str) -> str:
    """Slurm's vocabulary reduced to what a cell needs to be coloured."""
    head = (state or "").split()[0].upper() if state else ""
    head = head.rstrip("+")
    if head in _RUNNING:
        return "running"
    if head in _QUEUED:
        return "queued"
    if head in _FINISHED:
        return "finished"
    if head:
        return "failed"
    return ""


def job_states(job_ids: list) -> dict:
    """``{job id: {state, slurm_state, node, elapsed}}`` for the ids given.

    Unknown ids are simply absent, which the caller reads as "nothing known"
    rather than as any particular state.
    """
    wanted = [str(one) for one in job_ids if one]
    if not wanted:
        return {}

    found: dict[str, dict] = {}
    for line in _run(
        ["squeue", "-h", "-j", ",".join(wanted), "-o", "%i|%T|%N|%M"]
    ).splitlines():
        _absorb(found, line, wanted)
    missing = [one for one in wanted if one not in found]
    if missing:
        for line in _run(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-j",
                ",".join(missing),
                "-o",
                _SACCT_FIELDS,
            ]
        ).splitlines():
            _absorb(found, line, missing)
    return found


def _absorb(found: dict, line: str, wanted: list) -> None:
    parts = line.strip().split("|")
    if len(parts) < 4:
        return
    job_id, state, node, elapsed = parts[0], parts[1], parts[2], parts[3]
    # An array task reports as "8996385_2"; the invocation recorded whichever
    # form SLURM_JOB_ID held, so both are accepted.
    for candidate in (job_id, job_id.split("_", 1)[0]):
        if candidate in wanted and candidate not in found:
            found[candidate] = {
                "slurm_state": state,
                "state": classify(state),
                "node": node,
                "elapsed": elapsed,
            }
            return


def available() -> bool:
    """Whether a queue can be asked at all."""
    return bool(_run(["squeue", "--version"]))
