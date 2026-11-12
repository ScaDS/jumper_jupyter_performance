# Configuration

JUmPER is configured through a single, global `AppConfig` object, composed at
import time from the YAML files under `jumper_extension/config/` and exposed via
`load_config()`. Every default mentioned elsewhere in this documentation — the
initial `--level`, the sampling interval, which plot subsets are shown, which
collectors run, which LLM the AI reviewer talks to — comes from this config.

```python
from jumper_extension.config.loader import load_config

config = load_config()
config.settings.perfreports.level   # "process"
config.plots.default_subsets         # ["cpu", "mem", "io"]
config.ai.llm.model                  # "MiniMaxAI/MiniMax-M3-MXFP8"
```

## How it's put together

`jumper_extension/config/config.yaml` lists which option file to load for each
config group, Hydra-style:

```yaml
defaults:
  - settings: default
  - plots: default
  - ai: default
  - collectors/python: default
  - collectors/c: default
```

`load_config()` reads this list, loads `jumper_extension/config/<group>/<option>.yaml`
for each entry, validates the merged result with Pydantic (`AppConfig` in
`jumper_extension/config/models.py`), and caches it for the lifetime of the
process.

!!! note
    Config files are shipped as package data and read via `importlib.resources` —
    there is currently no user-level override file (e.g. `~/.jumper/config.yaml`).
    To change a default, edit the corresponding `default.yaml` in your
    installed/checked-out copy of `jumper_extension/config/`.

### Adding your own variant of a config group

Each entry in `defaults:` is `<group>: <option>`, which resolves to
`config/<group>/<option>.yaml` — `default` is just the name of the option that
ships out of the box, not a special value. To make a custom variant the new
default for the whole application:

1. Copy `config/<group>/default.yaml` to `config/<group>/<your-name>.yaml`
   (e.g. `config/settings/lab.yaml`) and edit it as needed.
2. In `config/config.yaml`, change that group's entry to point at it, e.g.
   `- settings: lab`.
3. `load_config()` will now load `settings/lab.yaml` everywhere — no other code
   changes needed. `pyproject.toml`'s package-data entries already glob `*.yaml`
   per group, so the new file is picked up automatically.

This works for any group (`settings`, `plots`, `ai`, `collectors/python`,
`collectors/c`).

## `settings` — `config/settings/default.yaml`

```yaml
perfreports:
  level: process
  text: false

monitoring:
  default_interval: 1.0
  live_update_interval: 2.0
  live_window_seconds: 120.0

export_vars:
  perfdata: perfdata_df
  cell_history: cell_history_df

loaded_vars:
  perfdata: loaded_perfdata_df
  cell_history: loaded_cell_history_df

visualizer_backend: plotly
```

| Key | Affects |
|---|---|
| `perfreports.level` | Default `--level` for `%perfmonitor_perfreport`, `%perfmonitor_ai_review`, `%perfmonitor_export_perfdata`, and the initial level used by automatic per-cell reports and `fast_setup()`. |
| `perfreports.text` | Default `--text` for `%perfmonitor_perfreport` and `fast_setup()`. |
| `monitoring.default_interval` | Sampling interval used by `%perfmonitor_start` and `fast_setup()` when no interval is given, default `--interval` for `%perfmonitor_auto_perfreports`, and the fallback interval recorded by `start_script_recording`. |
| `monitoring.live_update_interval` / `monitoring.live_window_seconds` | Default `INTERVAL`/`WINDOW` for `%perfmonitor_plot --live` when not given explicitly. |
| `export_vars.perfdata` / `export_vars.cell_history` | Default variable names created by `%export_perfdata` / `%export_cell_history` when `--name` is omitted. |
| `loaded_vars.perfdata` / `loaded_vars.cell_history` | Default variable names created by `%import_perfdata` / `%import_cell_history`. |
| `visualizer_backend` | Default plotting backend (`matplotlib` or `plotly`), used when `build_perfmonitor_service()` / `%load_ext jumper_extension` is not given an explicit backend. |

These values seed `jumper_extension.core.state.State` via `State.from_config()`
and are then mutated at runtime by magic commands (e.g. `%perfmonitor_perfreport
--level user` changes `state.perfreports.level` for subsequent auto-reports,
but does not change the config default).

## `plots` — `config/plots/default.yaml`

Controls which metric subsets exist and which ones `%perfmonitor_plot` shows
when called without `--metrics`. See the
[Visualizing Custom Collector Metrics guide](visualizing-custom-collector-metrics.md)
for the full reference on `default_subsets`, plot types, and adding new subsets.

## `collectors/python` and `collectors/c`

Define which metric collectors are active for the Python-based monitors
(`thread`, `subprocess_python`) and the `native_c` binary, respectively. See:

- [Custom Python Collectors](python-custom-collector.md) — registering a
  collector in `config/collectors/python/default.yaml`.
- [Custom C Collectors](c-custom-collector.md) — registering a collector in
  `config/collectors/c/default.yaml`.

## `ai` — `config/ai/default.yaml`

Everything `%perfmonitor_ai_review` needs, in four groups: `llm` (which model
answers), `suggestions` (how many options it proposes), `benchmark` (how
suggestions are measured) and `context` (what the model is told). Usage is covered in the [AI Review](ai-review/index.md) guides.

### `ai.llm` — the model

```yaml
llm:
  base_url: https://llm.scads.ai/v1
  model: MiniMaxAI/MiniMax-M3-MXFP8
  max_tokens: 16000
  timeout: 520.0
  api_key_env: JUMPER_AI_API_KEY
```

| Key | Description |
|---|---|
| `base_url` | OpenAI-compatible API endpoint. |
| `model` | Model identifier passed to the endpoint. |
| `max_tokens` / `timeout` / `max_retries` | Per-completion token budget, request timeout in seconds, retries. |
| `streaming` | Stream the reply instead of waiting for it whole. |
| `temperature` / `top_p` / `seed` | Sampling controls; `null` leaves the endpoint's own default. |
| `enable_thinking` | Ask a reasoning model for its chain of thought — shown collapsed on the review card. |
| `extra_body` | Arbitrary passthrough to a vLLM endpoint (`top_k`, `min_p`, `guided_*`, …). |
| `api_key_env` | Name of the **environment variable** holding the API key. |

All model parameters live in this file — none of them are read from environment
variables. The only secret is the API key itself, which is never stored in the
config: set the environment variable named by `api_key_env` (default
`JUMPER_AI_API_KEY`) before starting your notebook or IPython session. To use a
different secret name (e.g. on a shared cluster), change `api_key_env` here and
export that variable instead.

### `ai.suggestions` — how many options

```yaml
suggestions:
  count: 3
```

| Key | Description |
|---|---|
| `count` | Optimization options asked for per review. The suggest prompt names this number, and the structured-output schema accepts exactly that many: an endpoint that honors JSON-schema array bounds cannot return another number, and a reply that does anyway fails to parse. |

### `ai.benchmark` — measuring suggestions

```yaml
benchmark:
  runs: 3
  fix_attempts: 3
  interval: 0.05
  timeout_factor: 10.0
  checks:
    validate_syntax: true
    run: true
  replay:
    mode: full
    cross_check: true
    dill_max_checkpoint_gb: 4.0
```

| Key | Description |
|---|---|
| `runs` | Timed replays per suggestion; the first is dropped as warm-up. Per run: `--benchmark-runs`. |
| `fix_attempts` | Repair rounds before a failing suggestion is reported as failed. Per run: `--fix-attempts`. |
| `interval` | Sampling interval for replays, finer than the live monitor's so a fast variant still yields metrics. |
| `timeout_factor` | A variant is killed past this multiple of the baseline duration. |
| `checks.validate_syntax` / `checks.run` | Which [benchmark steps](ai-review/benchmark.md#running-fewer-steps) run. Per run: `--check` / `--skip-check`. |
| `replay.mode` | How prefix state is rebuilt: `full`, `fork`, `dill`. Per run: `--replay-mode`. See [Replay modes](ai-review/replay-modes.md). |
| `replay.cross_check` | With a fast mode active, also measure the baseline once through `full` and compare. Differing results send the whole benchmark back to `full`; differing durations warn. Costs one extra prefix replay. |
| `replay.dill_max_checkpoint_gb` | Ceiling on a `dill` checkpoint; passing it abandons the checkpoint and falls back to `full`. |

### `ai.context` — what the model is told

```yaml
context:
  strategy: faster
  known_packages:
    - numpy
    - torch
    # ...
```

| Key | Description |
|---|---|
| `strategy` | Default `--strategy`; see [Steering the review](ai-review/strategies.md). |
| `known_packages` | Packages whose installed versions are reported to the model, so it only suggests what is available. Absent ones are omitted. |

!!! note
    Strategies themselves are not part of this config. They live in
    `adapters/ai_reviewer/strategy/strategies.yaml`, next to the prompt specs
    they toggle.
