"""Approximate signatures of a cell's results, so a variant that quietly
computes something else is not reported as a speedup.

Signatures are statistical (shape/dtype/mean/std/min/max), never exact hashes:
reordering a sum - which is what vectorising *does* - changes the last bits of a
float, so byte equality would flag every honest optimization as wrong.

A value this cannot summarise - a list, a dict, an object - has no signature
at all, and a cell whose result is one of those is verified by whatever else
it binds. Where that matters, the cell's result is named with
``ai.benchmark.result_names`` and the usecase provides a summarisable one
alongside it; see the ECCO payload cells, which bind a checksum next to their
list of answers for exactly this reason.
"""
import ast
import json
import math

REL_TOL = 1e-6

MATCH = "match"
DIFFERS = "differs"
UNVERIFIED = "unverified"


def assigned_names(code: str) -> list[str]:
    """Top-level names *code* assigns, in source order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            names.extend(_target_names(target))
    return list(dict.fromkeys(names))


def _target_names(target) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for element in target.elts:
            names.extend(_target_names(element))
        return names
    # x.attr = ... / x[i] = ... bind nothing new
    return []


def fingerprint(value) -> dict | None:
    """Statistical signature of *value*, or None when it cannot be summarised."""
    if value is None or isinstance(value, bool):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (int, float)):
        return {"kind": "scalar", "value": float(value)}
    if isinstance(value, str):
        return {"kind": "text", "len": len(value)}
    if hasattr(value, "columns") and hasattr(value, "shape"):
        return {
            "kind": "frame",
            "shape": list(value.shape),
            "columns": [str(column) for column in value.columns][:50],
        }
    return _array_fingerprint(value)


def _array_fingerprint(value) -> dict | None:
    """Summarise anything array-shaped: numpy, cupy, torch, ..."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    base = {
        "kind": "array",
        "shape": [int(dimension) for dimension in shape],
        "dtype": str(getattr(value, "dtype", "")),
    }
    try:
        base.update(
            mean=float(value.mean()),
            std=float(value.std()),
            min=float(value.min()),
            max=float(value.max()),
        )
    except Exception:
        # Non-numeric or exotic array: shape and dtype still catch a variant
        # that silently processed fewer elements.
        pass
    return base


def capture(names: list[str], namespace: dict) -> dict:
    """Fingerprint each of *names* present in *namespace*."""
    captured = {}
    for name in names:
        if name not in namespace:
            continue
        try:
            captured[name] = fingerprint(namespace[name])
        except Exception:
            captured[name] = None
    return captured


def dump(names: list[str], namespace: dict, path: str) -> None:
    """Write fingerprints of *names* to *path*; called from the replay script."""
    with open(path, "w") as handle:
        json.dump(capture(names, namespace), handle)


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def compare(baseline: dict | None, variant: dict | None, rel_tol: float = REL_TOL) -> str:
    """Compare two fingerprints: MATCH, DIFFERS or UNVERIFIED."""
    if not baseline or not variant:
        return UNVERIFIED
    if baseline.get("kind") != variant.get("kind"):
        return DIFFERS

    kind = baseline["kind"]
    if kind == "scalar":
        return _compare_numbers(baseline["value"], variant["value"], rel_tol)
    if kind == "text":
        return MATCH if baseline["len"] == variant["len"] else DIFFERS
    if kind == "frame":
        same = (
            baseline["shape"] == variant["shape"]
            and baseline["columns"] == variant["columns"]
        )
        return MATCH if same else DIFFERS
    return _compare_arrays(baseline, variant, rel_tol)


def _compare_numbers(baseline, variant, rel_tol: float) -> str:
    if isinstance(baseline, bool) or baseline is None or isinstance(variant, bool) or variant is None:
        return MATCH if baseline == variant else DIFFERS
    if math.isnan(baseline) and math.isnan(variant):
        return MATCH
    return MATCH if math.isclose(baseline, variant, rel_tol=rel_tol) else DIFFERS


def _compare_arrays(baseline: dict, variant: dict, rel_tol: float) -> str:
    if baseline["shape"] != variant["shape"]:
        return DIFFERS
    stats = ("mean", "std", "min", "max")
    if not all(stat in baseline and stat in variant for stat in stats):
        # Shape matched but the values were never summarised.
        return UNVERIFIED
    for stat in stats:
        if _compare_numbers(baseline[stat], variant[stat], rel_tol) == DIFFERS:
            return DIFFERS
    return MATCH


def describe_divergence(baseline: dict, variant: dict, names: list[str]) -> str:
    """Explain how a variant's results drifted, in terms a model can act on."""
    lines = [
        "Your rewrite ran without error, but it no longer computes the same "
        "result as the original.",
        "",
        "These are statistical signatures of the values each version left behind. "
        "They are compared with a tolerance, so a small floating-point difference "
        "from reordering arithmetic would have passed - this did not:",
    ]
    for name in names:
        lines.append(f"  {name}:")
        lines.append(f"    original: {_describe(baseline.get(name))}")
        lines.append(f"    yours:    {_describe(variant.get(name))}")
    return "\n".join(lines)


def _describe(print_: dict | None) -> str:
    if not print_:
        return "not produced"
    kind = print_.get("kind")
    if kind == "scalar":
        return f"{print_['value']}"
    if kind == "text":
        return f"text of {print_['len']} chars"
    if kind == "frame":
        return f"dataframe {print_['shape']}, columns {print_['columns']}"
    stats = " ".join(
        f"{stat}={print_[stat]}" for stat in ("mean", "std", "min", "max") if stat in print_
    )
    return f"array shape={print_['shape']} dtype={print_['dtype']}{f' {stats}' if stats else ''}"


def compare_all(
    baseline: dict, variant: dict, rel_tol: float = REL_TOL
) -> tuple[str, list[str], list[str]]:
    """Verdict over every captured name, the names that diverged, and the
    names the variant no longer binds at all.

    Three outcomes have to stay apart, and two of them used to arrive here as
    one. A name with no baseline signature (``None`` - a function, a handle)
    carries no verification signal, so it is skipped rather than dragging an
    otherwise-verified result down. A name the variant never bound is
    something else entirely: the rewrite stopped producing something the
    original produced. It is reported as *missing*, and it still keeps the
    verdict out of MATCH - a rewrite that dropped the cell's output while
    keeping a constant must never read as correct - but it is no longer
    indistinguishable from "nothing here could be compared". The difference
    is the whole diagnosis: a missing ``answers`` is a broken rewrite, a
    missing ``MOD`` is a constant that moved inside a function.
    """
    if not baseline:
        return UNVERIFIED, [], []

    differing: list[str] = []
    missing: list[str] = []
    verdict = MATCH
    compared = 0
    for name, expected in baseline.items():
        if not expected:
            continue  # unsummarisable baseline value: no signal, do not poison
        if name not in variant:
            missing.append(name)
            continue
        compared += 1
        result = compare(expected, variant[name], rel_tol)
        if result == DIFFERS:
            differing.append(name)
        elif result == UNVERIFIED and verdict == MATCH:
            verdict = UNVERIFIED
    if differing:
        return DIFFERS, differing, missing
    if missing or compared == 0:
        return UNVERIFIED, [], missing
    return verdict, [], missing


def describe_missing_results(names: list[str]) -> str:
    """Tell the model it dropped the cell's result, in terms it can act on.

    Separate from :func:`describe_divergence` because the failure is different:
    nothing was computed wrongly, nothing was computed at all. A rewrite that
    defines a faster function and never calls it hits this, and left to the
    divergence wording it would be asked to fix arithmetic that is fine.
    """
    listed = ", ".join(names)
    verb = "is" if len(names) == 1 else "are"
    return (
        "Your rewrite ran without error, but it no longer produces the "
        f"result of the cell: {listed} {verb} not bound when the cell "
        "finishes.\n\n"
        "The cell under review computes this at its top level, and whatever "
        "runs after it reads it from there. A faster function is only half of "
        "the rewrite: it has to be called, and its result assigned to the same "
        f"name ({listed}), at the top level of the cell."
    )
