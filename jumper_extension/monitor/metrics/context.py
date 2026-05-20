from typing import Any, TypedDict


class CollectionContext(TypedDict):
    process_pids: set[int]
    user_pids: set[int]    # process_pids ∪ user-owned pids
    slurm_pids: set[int]   # process_pids ∪ slurm-job pids
    cpu: dict[int, float]
    rss: dict[int, int]
    io: dict[int, Any]
