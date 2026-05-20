"""Built-in plot renderers.

Each function is registered under its plot-type key via @register.
To add a custom renderer, import register and decorate your function:

    from jumper_extension.adapters.visualizer.render import RENDERERS, PlotResult, SeriesItem, register

    @register("my_type")
    def render_my_type(df, config, level, hardware, io_window):
        series = df[config.column]
        return PlotResult(
            series=[SeriesItem(label=config.label, data=series)],
            title=config.title,
            ylim=config.ylim,
        )
"""

import re

from jumper_extension.adapters.visualizer.render import RENDERERS, PlotResult, SeriesItem, register  # noqa: F401

_IO_BYTE_COLUMNS = ("io_read", "io_write")
_IO_COLUMNS = ("io_read", "io_write", "io_read_count", "io_write_count")


@register("single_series")
def render_single_series(df, config, level, hardware, io_window):
    column = config.column
    if column not in df.columns:
        return None

    series = df[column].astype(float).clip(lower=0)
    if column in _IO_BYTE_COLUMNS:
        series = series / (1024 ** 2)
    if column in _IO_COLUMNS and io_window > 1:
        series = series.rolling(window=io_window, min_periods=1).mean()

    ylim = config.ylim
    if column == "memory" and ylim is None:
        ylim = (0.0, float(hardware.memory_limits.get(level, 0.0)))

    return PlotResult(
        series=[
            SeriesItem(
                label=config.label,
                data=series,
                color="blue",
                width=2.0,
                opacity=1.0,
                linestyle="solid",
            )
        ],
        title=config.title,
        ylim=ylim,
    )


@register("summary_series")
def render_summary_series(df, config, level, hardware, io_window):
    available = [col for col in config.columns if col in df.columns]
    if not available:
        return None

    title = config.title
    if level == "system":
        title = re.sub(r"\d+", str(hardware.num_system_cpus), title)

    linestyles = ["dotted", "solid", "dashed"]
    opacities = [0.35, 1.0, 0.35]
    labels = ["Min", "Average", "Max"]

    series_items = []
    for index, column in enumerate(config.columns):
        if column not in df.columns:
            continue
        series_items.append(SeriesItem(
            label=labels[index % len(labels)],
            data=df[column],
            color="blue",
            width=2.0,
            opacity=opacities[index % len(opacities)],
            linestyle=linestyles[index % len(linestyles)],
        ))

    return PlotResult(series=series_items, title=title, ylim=config.ylim)


@register("multi_series")
def render_multi_series(df, config, level, hardware, io_window):
    prefix = config.prefix
    per_device_columns = [
        col for col in df.columns
        if prefix and col.startswith(prefix) and not col.endswith("avg")
    ]
    avg_column = f"{prefix}avg" if prefix else None
    has_avg = avg_column and avg_column in df.columns

    if not per_device_columns and not has_avg:
        return None

    series_items = [
        SeriesItem(
            label=column,
            data=df[column],
            color="blue",
            width=1.0,
            opacity=0.5,
            linestyle="solid",
        )
        for column in per_device_columns
    ]
    if has_avg:
        series_items.append(SeriesItem(
            label="Mean",
            data=df[avg_column],
            color="blue",
            width=2.0,
            opacity=1.0,
            linestyle="solid",
        ))

    return PlotResult(series=series_items, title=config.title, ylim=config.ylim)


@register("composite_series")
def render_composite_series(df, config, level, hardware, io_window):
    series_items = [
        SeriesItem(
            label=style.label,
            data=df[style.column],
            color=style.color,
            width=style.width,
            opacity=1.0,
            linestyle="solid",
        )
        for style in config.series
        if style.column in df.columns
    ]

    if not series_items:
        return None

    return PlotResult(series=series_items, title=config.title, ylim=config.ylim)


# === User-defined renderers ===
# Add your custom renderers below using @register("<type>").
#
# Example:
#
#   @register("my_type")
#   def render_my_type(df, config, level, hardware, io_window):
#       series = df[config.column]
#       return PlotResult(
#           series=[SeriesItem(label=config.label, data=series)],
#           title=config.title,
#           ylim=config.ylim,
#       )
