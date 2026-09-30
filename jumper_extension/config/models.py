"""Pydantic v2 models for plot-subset descriptors.

``MetricConfig`` is a discriminated union on the ``type`` field; Pydantic
selects the correct concrete model automatically during ``model_validate()``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class SeriesStyle(BaseModel):
    column: str
    label: str
    color: str = "steelblue"
    width: float = 2.0
    y_axis: Literal["left", "right"] = "left"


class SingleSeriesConfig(BaseModel):
    type: Literal["single_series"]
    column: str
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class SummarySeriesConfig(BaseModel):
    type: Literal["summary_series"]
    columns: list[str]
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class MultiSeriesConfig(BaseModel):
    type: Literal["multi_series"]
    prefix: str
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class CompositeSeriesConfig(BaseModel):
    type: Literal["composite_series"]
    series: list[SeriesStyle]
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


MetricConfig = Annotated[
    Union[
        SingleSeriesConfig,
        SummarySeriesConfig,
        MultiSeriesConfig,
        CompositeSeriesConfig,
    ],
    Field(discriminator="type"),
]

def validate_metric_config(data: dict):
    from pydantic import TypeAdapter
    return TypeAdapter(MetricConfig).validate_python(data)


class PerfReportsDefaults(BaseModel):
    level: str = "process"
    text: bool = False


class MonitoringDefaults(BaseModel):
    default_interval: float = 1.0
    live_update_interval: float = 2.0
    live_window_seconds: float = 120.0


class ExportVarsConfig(BaseModel):
    perfdata: str = "perfdata_df"
    cell_history: str = "cell_history_df"


class LoadedVarsConfig(BaseModel):
    perfdata: str = "loaded_perfdata_df"
    cell_history: str = "loaded_cell_history_df"


class SettingsConfig(BaseModel):
    perfreports: PerfReportsDefaults = Field(default_factory=PerfReportsDefaults)
    monitoring: MonitoringDefaults = Field(default_factory=MonitoringDefaults)
    export_vars: ExportVarsConfig = Field(default_factory=ExportVarsConfig)
    loaded_vars: LoadedVarsConfig = Field(default_factory=LoadedVarsConfig)
    visualizer_backend: str = "plotly"


class PlotsConfig(BaseModel):
    default_subsets: list[str] = Field(default_factory=lambda: ["cpu", "mem", "io"])
    subsets: dict[str, dict[str, MetricConfig]]


class AILLMConfig(BaseModel):
    """LLM client parameters for the AI reviewer."""
    base_url: str = "https://llm.scads.ai/v1"
    model: str = "MiniMaxAI/MiniMax-M2.7"
    max_tokens: int = 8000
    timeout: float = 120.0
    max_retries: int = 2
    streaming: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    extra_body: dict = Field(default_factory=dict)
    api_key_env: str = "JUMPER_AI_API_KEY"


class AIContextConfig(BaseModel):
    """Context-gathering parameters: default strategy and known packages."""
    strategy: str = "faster"
    # An extra strategies file merged over the built-in ones, so a preset can
    # be added without editing the package. Overridden by the environment
    # variable JUMPER_AI_STRATEGIES_PATH.
    strategies_path: str | None = None
    known_packages: list[str] = Field(default_factory=list)


class AIBenchmarkChecksConfig(BaseModel):
    """Which benchmark steps run. Each is also gated by adapter capability: a
    step enabled here but unsupported for the cell's language is skipped with a
    warning; one turned off here is skipped silently.
    """
    # Static parse check gating a suggestion before any replay.
    validate_syntax: bool = True
    # The timed replay, plus fingerprinting and comparing each variant's results
    # against the baseline's. The two are one step: there is nothing to
    # fingerprint without an execution, and capturing it is cheap next to the run.
    run: bool = True


class AIBenchmarkReplayConfig(BaseModel):
    """How the state a timed cell needs is rebuilt between measurements.

    ``full`` re-runs every preceding cell per measurement: always correct, and
    the reason a benchmark is slow. The others rebuild that state once and reuse
    it, and each is gated by what it can actually serve - a mode that is not
    built, or cannot handle the cell's language, degrades to ``full`` with a
    warning rather than failing the benchmark.
    """
    mode: Literal["full", "fork", "dill"] = "full"
    # Measure the baseline once through the full replay as well, and compare.
    # Worth the extra prefix run: a mode that rebuilt the wrong state would
    # otherwise pass unnoticed, because every variant is compared against a
    # baseline that went through the same broken rebuild, so the two agree and
    # the divergence check reports a match. Results that differ send the whole
    # benchmark back to the full replay; durations that differ grossly warn.
    cross_check: bool = True
    # Ceiling on a dill checkpoint. A prefix holding tens of gigabytes would
    # otherwise fill a shared filesystem before anything could fall back.
    dill_max_checkpoint_gb: float = 4.0


class AIBenchmarkConfig(BaseModel):
    """Parameters for replaying and timing the suggestions of a review."""
    runs: int = 3
    fix_attempts: int = 3
    # Finer than the live monitor's: a successful optimization is often too
    # short to be sampled at 1s, and would come back with no metrics at all.
    interval: float = 0.05
    # Kill a variant once it exceeds this multiple of the baseline duration.
    timeout_factor: float = 10.0
    checks: AIBenchmarkChecksConfig = Field(default_factory=AIBenchmarkChecksConfig)
    replay: AIBenchmarkReplayConfig = Field(default_factory=AIBenchmarkReplayConfig)


class AIConfig(BaseModel):
    llm: AILLMConfig = Field(default_factory=AILLMConfig)
    context: AIContextConfig = Field(default_factory=AIContextConfig)
    benchmark: AIBenchmarkConfig = Field(default_factory=AIBenchmarkConfig)


class PythonCollectorsConfig(BaseModel):
    collectors: dict[str, dict]


class CCollectorsConfig(BaseModel):
    collectors: list[str]


class CollectorsConfig(BaseModel):
    python: PythonCollectorsConfig
    c: CCollectorsConfig


class AppConfig(BaseModel):
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    plots: PlotsConfig
    ai: AIConfig = Field(default_factory=AIConfig)
    collectors: CollectorsConfig
