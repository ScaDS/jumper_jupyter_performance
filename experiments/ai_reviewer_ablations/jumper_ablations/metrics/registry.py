"""Finding metrics, so adding one is adding a file.

Every module under ``metrics/analysis`` and ``metrics/suggestions`` is
imported and searched for metric classes. Nothing lists them: an ``__init__``
that had to be edited for each new metric would be exactly the coupling this
experiment is supposed to avoid, and the YAML item that switches a metric on
is meant to be the only edit besides the module itself.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

from jumper_ablations.metrics.base import (
    CATEGORY_ANALYSIS,
    CATEGORY_SUGGESTIONS,
    DeterministicMetric,
    JudgeMetric,
    Metric,
)

_PACKAGES = {
    CATEGORY_ANALYSIS: "jumper_ablations.metrics.analysis",
    CATEGORY_SUGGESTIONS: "jumper_ablations.metrics.suggestions",
}

_CACHE: dict[str, Metric] | None = None


def _discover() -> dict[str, Metric]:
    found: dict[str, Metric] = {}
    for category, package_name in _PACKAGES.items():
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(
                f"{package_name}.{module_info.name}"
            )
            for _, member in inspect.getmembers(module, inspect.isclass):
                if member.__module__ != module.__name__:
                    continue
                if not issubclass(member, (DeterministicMetric, JudgeMetric)):
                    continue
                if not getattr(member, "spec", None):
                    continue
                metric = member()
                if metric.spec.category != category:
                    raise ValueError(
                        f"{metric.id} is declared '{metric.spec.category}' but "
                        f"lives under {package_name}"
                    )
                if metric.id in found:
                    raise ValueError(f"two metrics claim the id '{metric.id}'")
                found[metric.id] = metric
    return found


def all_metrics(refresh: bool = False) -> dict[str, Metric]:
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = _discover()
    return dict(_CACHE)


def get_metric(metric_id: str) -> Metric:
    metrics = all_metrics()
    if metric_id not in metrics:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"unknown metric '{metric_id}'; available: {available}")
    return metrics[metric_id]


def metrics_for(
    category: str | None = None,
    evaluation_method: str | None = None,
) -> list[Metric]:
    """Every registered metric matching both filters, id-sorted."""
    selected = []
    for metric in all_metrics().values():
        if category and metric.spec.category != category:
            continue
        if evaluation_method and metric.spec.evaluation_method != evaluation_method:
            continue
        selected.append(metric)
    return sorted(selected, key=lambda metric: metric.id)


def selected_metrics(metric_ids: list[str]) -> list[Metric]:
    """Resolve configured ids, failing loudly on one that does not exist."""
    return [get_metric(metric_id) for metric_id in metric_ids]
