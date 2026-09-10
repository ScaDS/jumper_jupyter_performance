"""Typed view over the Hydra-composed config.

Hydra composes and records; these models are what the rest of the harness
actually reads, so a typo in a YAML file fails at startup with a field name
rather than four minutes into a benchmark with an AttributeError.
"""
from __future__ import annotations

from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

CONTEXT_FAMILY = "context"
PROMPT_FAMILY = "prompt"


class AblationEffect(BaseModel):
    """What a preset switches, in the reviewer's own two buckets.

    ``context`` toggles ids the collector knows (data really disappears);
    ``overrides`` toggles prompt items (the wording changes, the data does
    not). They are merged into one flat map downstream, exactly as the
    reviewer's own strategy loader does it.
    """

    context: dict[str, bool] = Field(default_factory=dict)
    overrides: dict[str, bool] = Field(default_factory=dict)


class AblationConfig(BaseModel):
    """One ablation, which is also literally one reviewer strategy."""

    id: str
    name: str
    description: str = ""
    family: Literal["context", "prompt"] = CONTEXT_FAMILY
    require_note: bool = False
    note: str = ""
    effect: AblationEffect = Field(default_factory=AblationEffect)

    def flat_overrides(self) -> dict[str, bool]:
        """The ``id -> enabled`` map the reviewer applies.

        Prompt overrides win on a collision, matching
        ``strategy/loader.py`` - the two buckets share an id namespace on
        purpose, so one toggle can drop both a payload and the sentence
        promising it.
        """
        return {**self.effect.context, **self.effect.overrides}


class SuiteConfig(BaseModel):
    """The grid: which usecases, crossed with which ablations.

    This is the declarative half of a run. Everything below it says how hard
    each cell of the grid is hit, not which cells exist.
    """

    id: str
    description: str = ""
    usecases: list[str]
    ablations: list[str]


class BenchmarkProtocol(BaseModel):
    """How the suggestions get measured.

    ``mode`` decides which of the two shapes of the command runs. With
    ``separate`` the review is asked for on its own and the benchmark follows
    as ``--resume <id> --benchmark``, which is how the two commands are meant
    to be used together: look at the options, then spend the machine time. It
    also records the suggestions as the model wrote them, before the repair
    loop overwrites them, which is the only way to see what a repair changed.
    With ``inline`` the notebook's own ``--benchmark`` review runs as written.
    """

    enabled: bool = True
    mode: Literal["separate", "inline"] = "separate"
    runs: int = 7
    fix_attempts: int = 3
    extra_replay_modes: list[str] = Field(default_factory=list)


class KernelProtocol(BaseModel):
    name: str = "python3"
    startup_timeout: int = 180
    cell_timeout: int = 14400


class ProtocolConfig(BaseModel):
    generations_per_target: int = 10
    repetitions: int = 1
    pin_target_cell: bool = True
    level: str = "process"
    benchmark: BenchmarkProtocol = Field(default_factory=BenchmarkProtocol)
    kernel: KernelProtocol = Field(default_factory=KernelProtocol)


class MetricItem(BaseModel):
    """The on/off switch for one metric, plus whatever it needs to know."""

    id: str
    enabled: bool = True
    parameters: dict = Field(default_factory=dict)


class MetricGroupConfig(BaseModel):
    metrics: list[MetricItem] = Field(default_factory=list)

    def enabled_ids(self) -> list[str]:
        return [item.id for item in self.metrics if item.enabled]

    def parameters_for(self, metric_id: str) -> dict:
        for item in self.metrics:
            if item.id == metric_id:
                return dict(item.parameters)
        return {}


class MetricsConfig(BaseModel):
    """The two categories, kept apart because they answer different questions.

    Both are read through the same registry and scored through the same
    evaluation methods; only the lists are separate.
    """

    analysis: MetricGroupConfig = Field(default_factory=MetricGroupConfig)
    suggestions: MetricGroupConfig = Field(default_factory=MetricGroupConfig)


class JudgeConfig(BaseModel):
    blind: bool = True
    units_per_packet: int = 1
    sample_per_cell: int | None = None
    sources: list[str] = Field(default_factory=list)


class BootstrapConfig(BaseModel):
    samples: int = 10000
    confidence: float = 0.95
    seed: int = 0


class ReportingConfig(BaseModel):
    baseline_ablation: str = "base"
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    figures: bool = True


class ExperimentConfig(BaseModel):
    """Everything one run of the experiment was told to do."""

    run_id: str
    results_root: str
    workspace_root: str
    usecases_root: str
    target_run: str | None = None
    suite: SuiteConfig
    protocol: ProtocolConfig
    judge: JudgeConfig
    metrics: MetricsConfig
    reporting: ReportingConfig


def load_experiment_config(config: DictConfig) -> ExperimentConfig:
    """Validate the Hydra-composed config into :class:`ExperimentConfig`."""
    raw = OmegaConf.to_container(config, resolve=True)
    return ExperimentConfig.model_validate(raw)


def load_ablation_config(config: DictConfig) -> AblationConfig:
    """Validate one composed ``configs/ablation/<id>.yaml`` entry."""
    raw = OmegaConf.to_container(config, resolve=True)
    return AblationConfig.model_validate(raw)
