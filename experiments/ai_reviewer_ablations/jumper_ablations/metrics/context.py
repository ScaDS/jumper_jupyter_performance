"""What a metric is handed, so no metric ever touches storage.

Two views over the same records, matching the two scopes. A run context is one
reviewer invocation; a cell context is every invocation that shares a
(usecase, ablation) cell of the grid. Both carry the usecase manifest, because
the reference facts a judge scores against belong to the payload, not to the
preset.
"""

from __future__ import annotations

import dataclasses
import logging

from jumper_ablations.records.schema import (
    PHASE_REBENCHMARK,
    PHASE_REVIEW,
    RunRecord,
)
from jumper_ablations.usecases.registry import Usecase

logger = logging.getLogger("jumper_ablations")


@dataclasses.dataclass(frozen=True)
class RunContext:
    """One reviewer invocation, and what measuring it produced.

    The review and the benchmark are two invocations by default - that is how
    the two commands are meant to be used, and it is what puts the
    suggestions on record as the model wrote them. So a metric that judges
    the suggestions and then checks them against measurements needs both
    halves, and ``measurement`` is the benchmark record that re-opened this
    review. It is None when the suggestions were never measured.
    """

    record: RunRecord
    usecase: Usecase | None = None
    measurement: RunRecord | None = None

    @property
    def usecase_id(self) -> str:
        return self.record.identity.usecase

    @property
    def ablation_id(self) -> str:
        return self.record.identity.ablation

    @property
    def unit_id(self) -> str:
        return self.record.identity.record_id

    @property
    def sample_size(self) -> int:
        return 1

    def benchmarks(self) -> dict:
        """The benchmark verdicts for this invocation's suggestions."""
        if self.record.outputs.benchmarks:
            return self.record.outputs.benchmarks
        if self.measurement is not None:
            return self.measurement.outputs.benchmarks
        return {}


@dataclasses.dataclass(frozen=True)
class CellContext:
    """Every invocation in one cell of the (usecase x ablation) grid."""

    usecase_id: str
    ablation_id: str
    records: tuple[RunRecord, ...]
    usecase: Usecase | None = None

    @property
    def unit_id(self) -> str:
        return f"{self.usecase_id.replace('/', '-')}__{self.ablation_id}"

    @property
    def sample_size(self) -> int:
        return len(self.reviews())

    def reviews(self) -> tuple[RunRecord, ...]:
        """Records carrying an analysis and the suggestions as written."""
        return tuple(
            record
            for record in self.records
            if record.identity.phase == PHASE_REVIEW
        )

    def benchmarks(
        self, replay_mode: str | None = None
    ) -> tuple[RunRecord, ...]:
        """The records that carry measurements, optionally for one mode.

        A benchmark record is where the numbers live. Which mode produced them
        matters: the reviewer degrades a fast mode to the full replay rather
        than failing, so a mode is only itself when the record says it ran.
        """
        selected = [
            record
            for record in self.records
            if record.outputs.benchmarks
            and record.identity.phase in (PHASE_REBENCHMARK, PHASE_REVIEW)
        ]
        if replay_mode is not None:
            selected = [
                record
                for record in selected
                if record.environment.actual_replay_mode == replay_mode
            ]
        return tuple(selected)

    def measured_modes(self) -> tuple[str, ...]:
        """Replay modes that actually produced measurements here."""
        modes = {
            record.environment.actual_replay_mode
            for record in self.benchmarks()
            if record.environment.actual_replay_mode
        }
        return tuple(sorted(modes))

    def primary_benchmarks(self) -> tuple[RunRecord, ...]:
        """Benchmark records of the mode the usecase asked for.

        Everything that is not explicitly about comparing modes reads this, so
        one preset is never summarised over a mixture of instruments.
        """
        modes = self.measured_modes()
        if not modes:
            return ()
        requested = {
            record.identity.requested_replay_mode
            for record in self.benchmarks()
        }
        preferred = sorted(requested & set(modes))
        return self.benchmarks(preferred[0] if preferred else modes[0])

    def generation(self, index: int) -> tuple[RunRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.identity.generation == index
        )

    def unit_keys(self) -> tuple[tuple[int, int], ...]:
        """The experimental units in this cell, in order.

        A unit is ``(repetition, generation)``. Resampling and pairing key on
        this rather than on the generation alone, because a second repetition
        numbers its generations from one again.
        """
        return tuple(
            sorted(
                {
                    record.identity.unit_key
                    for record in self.records
                    if record.identity.generation
                }
            )
        )

    def unit(self, key: tuple[int, int]) -> tuple[RunRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.identity.unit_key == key
        )

    def restricted_to(self, keys) -> "CellContext":
        """The same cell, rebuilt from the listed units.

        A key may appear twice - that is what a bootstrap resample is - so the
        records are collected per listed key rather than filtered once.
        """
        return CellContext(
            usecase_id=self.usecase_id,
            ablation_id=self.ablation_id,
            records=tuple(
                record for key in keys for record in self.unit(key)
            ),
            usecase=self.usecase,
        )


def unit_key_of(unit_id: str) -> tuple[int, int] | None:
    """``(repetition, generation)`` out of a record id, or None.

    The unit is the pass and the generation within it - ``r00`` and ``g01`` in
    the record id - not the generation alone. Keying on the generation made a
    second repetition silently overwrite the first, so with
    ``repetitions > 1`` half the data left the comparison without a trace.
    """
    repetition, generation = None, None
    for part in unit_id.split("__"):
        if part.startswith("r") and part[1:].isdigit():
            repetition = int(part[1:])
        elif part.startswith("g") and part[1:].isdigit():
            generation = int(part[1:])
    if generation is None:
        return None
    return (repetition if repetition is not None else 0, generation)


def drop_empty_context(records: list[RunRecord]) -> list[RunRecord]:
    """Remove the generations the reviewer answered without any context.

    When the collector returns nothing the reviewer warns and calls the model
    anyway, and the model writes a confident analysis of a cell it was never
    shown. Those answers are not a weak measurement, they are not a
    measurement: leaving them in silently averages confabulation into every
    metric that reads the record.

    The benchmark of such a generation goes with it. It timed real code, but
    the suggestions it timed came out of an empty prompt, so counting them
    would put the same confabulation into the speedup and pass@k numbers by
    another route.
    """
    empty_runs = {
        record.identity.reviewer_run_id
        for record in records
        if record.identity.phase == PHASE_REVIEW
        and record.environment.context_was_empty
    }
    if not empty_runs:
        return list(records)

    kept = [
        record
        for record in records
        if record.identity.reviewer_run_id not in empty_runs
    ]
    logger.warning(
        f"excluded {len(records) - len(kept)} record(s) from "
        f"{len(empty_runs)} generation(s) whose reviewer received no "
        "context; they are reported as missing, never as zero"
    )
    return kept


def build_cell_contexts(
    records: list[RunRecord],
    usecases: dict | None = None,
) -> list[CellContext]:
    """Group records into grid cells, in a stable order."""
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in drop_empty_context(records):
        grouped.setdefault(record.cell_key, []).append(record)

    contexts = []
    for (usecase_id, ablation_id), rows in sorted(grouped.items()):
        rows.sort(
            key=lambda record: (
                record.identity.repetition,
                record.identity.generation,
                record.identity.phase,
                record.identity.record_id,
            )
        )
        contexts.append(
            CellContext(
                usecase_id=usecase_id,
                ablation_id=ablation_id,
                records=tuple(rows),
                usecase=(usecases or {}).get(usecase_id),
            )
        )
    return contexts


def build_run_contexts(
    records: list[RunRecord],
    usecases: dict | None = None,
) -> list[RunContext]:
    """One context per review, with its measurement attached.

    Only reviews get a context: they are what carries an analysis and the
    suggestions as written, and judging a benchmark record as well would ask
    a session to score the same analysis twice.
    """
    records = drop_empty_context(records)
    measurements = _measurements_by_review(records)
    contexts = []
    for record in records:
        if record.identity.phase != PHASE_REVIEW:
            continue
        key = (record.identity.usecase, record.identity.reviewer_run_id)
        contexts.append(
            RunContext(
                record=record,
                usecase=(usecases or {}).get(record.identity.usecase),
                measurement=measurements.get(key),
            )
        )
    return contexts


def _measurements_by_review(records: list[RunRecord]) -> dict:
    """The benchmark record per review, preferring the requested mode.

    A review can be measured under several replay modes; the one that answers
    "what were these suggestions worth" is the mode the usecase asked for,
    not whichever happened to be recorded last.
    """
    found: dict[tuple[str, str], RunRecord] = {}
    for record in records:
        if record.identity.phase != PHASE_REBENCHMARK:
            continue
        if not record.outputs.benchmarks:
            continue
        key = (record.identity.usecase, record.identity.reviewer_run_id)
        if key not in found or _is_requested_mode(record):
            found[key] = record
    return found


def _is_requested_mode(record: RunRecord) -> bool:
    return (
        record.environment.actual_replay_mode
        == record.identity.requested_replay_mode
    )
