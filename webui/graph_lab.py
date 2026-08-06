"""Configurable Matplotlib/Seaborn graph builder for recorded experiment data."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid

_cache = Path(os.environ.get("SEALTORCH_RESULTS", Path(__file__).resolve().parents[1] / "results")) / ".matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SOURCES = {"summary": "results.csv", "samples": "samples.csv", "logits": "logits.csv",
           "activations": "activation_samples.csv", "figure": "figure_data.csv",
           "events": "events.csv"}
PLOT_TYPES = ("line", "scatter", "bar", "point", "box", "violin", "strip", "swarm",
              "histogram", "kde", "ecdf", "regression", "area", "heatmap")
AGGREGATIONS = ("none", "mean", "median", "min", "max", "sum", "count")
ERROR_BARS = ("none", "sd", "se", "ci90", "ci95", "ci99")
THEMES = ("darkgrid", "whitegrid", "dark", "white", "ticks")
CONTEXTS = ("paper", "notebook", "talk", "poster")
PALETTES = ("deep", "muted", "pastel", "bright", "dark", "colorblind", "viridis",
            "magma", "rocket", "crest", "flare", "tab10", "Set2")


def _run_directory(results: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("_"):
        raise ValueError("invalid run id")
    directory = (results / run_id).resolve()
    if directory.parent != results.resolve() or not directory.is_dir():
        raise ValueError(f"unknown run: {run_id}")
    return directory


def available_runs(results: Path) -> list[dict]:
    runs = []
    for manifest_path in sorted(results.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text())
            sources = [name for name, filename in SOURCES.items()
                       if (manifest_path.parent / filename).is_file()]
            if not sources:
                continue
            model = manifest.get("model", "unknown")
            runs.append({"id": manifest["run_id"], "model": model, "sources": sources,
                         "label": f'{manifest["run_id"]} · {model}'})
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return runs


def load_data(results: Path, run_ids: list[str], source: str) -> pd.DataFrame:
    if source not in SOURCES:
        raise ValueError("unknown graph data source")
    if not run_ids:
        raise ValueError("select at least one run")
    frames = []
    for run_id in dict.fromkeys(run_ids):
        directory = _run_directory(results, run_id)
        path = directory / SOURCES[source]
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        frame["run_id"] = run_id
        frame["run_label"] = f'{manifest.get("model", "unknown")} · {run_id}'
        frames.append(frame)
    if not frames:
        raise ValueError(f"selected runs have no {source} data yet")
    return pd.concat(frames, ignore_index=True, sort=False)


def schema(results: Path, request: dict) -> dict:
    frame = load_data(results, list(request.get("runs", [])), request.get("source", "summary"))
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    values = {}
    for column in categorical:
        unique = frame[column].dropna().astype(str).unique()
        if len(unique) <= 100:
            values[column] = sorted(unique.tolist())
    return {"rows": len(frame), "columns": list(frame.columns), "numeric": numeric,
            "categorical": categorical, "values": values, "plot_types": PLOT_TYPES,
            "aggregations": AGGREGATIONS, "error_bars": ERROR_BARS,
            "themes": THEMES, "contexts": CONTEXTS, "palettes": PALETTES}


def _filtered(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = frame.copy()
    for column, condition in filters.items():
        if column not in result.columns:
            raise ValueError(f"unknown filter column: {column}")
        if isinstance(condition, list):
            result = result[result[column].astype(str).isin(map(str, condition))]
        elif isinstance(condition, dict):
            numeric = pd.to_numeric(result[column], errors="coerce")
            if condition.get("min") is not None:
                result = result[numeric >= float(condition["min"])]
            if condition.get("max") is not None:
                result = result[numeric <= float(condition["max"])]
        else:
            result = result[result[column].astype(str) == str(condition)]
    if result.empty:
        raise ValueError("filters removed every row")
    return result


def _error_bar(value: str):
    return {"none": None, "sd": "sd", "se": "se", "ci90": ("ci", 90),
            "ci95": ("ci", 95), "ci99": ("ci", 99)}[value]


def _long_data(frame: pd.DataFrame, x: str, metrics: list[str], hue: str | None,
               style: str | None) -> pd.DataFrame:
    identifiers = list(dict.fromkeys(column for column in (x, hue, style) if column))
    result = frame.melt(id_vars=identifiers, value_vars=metrics,
                        var_name="__graph_metric", value_name="__graph_value")
    result["__graph_value"] = pd.to_numeric(result["__graph_value"], errors="coerce")
    result = result.dropna(subset=["__graph_value"])
    if hue:
        result["__graph_series"] = result["__graph_metric"].astype(str) + " · " + result[hue].astype(str)
    else:
        result["__graph_series"] = result["__graph_metric"].astype(str)
    return result


def _draw(axis, frame: pd.DataFrame, config: dict, metrics: list[str], plot_type: str,
          color_offset: int = 0) -> None:
    if not metrics:
        return
    x = config["x"]; hue = config.get("hue") or None; style = config.get("style") or None
    for column in [x, *metrics, *(item for item in (hue, style) if item)]:
        if column not in frame.columns:
            raise ValueError(f"unknown graph column: {column}")
    long = _long_data(frame, x, metrics, hue, style)
    if long.empty:
        raise ValueError("selected metrics contain no numeric values")
    palette = config.get("palette", "colorblind")
    common = {"data": long, "hue": "__graph_series", "palette": palette, "ax": axis,
              "alpha": float(config.get("alpha", 0.9))}
    estimator_name = config.get("aggregation", "none")
    estimator = None if estimator_name == "none" else estimator_name
    errorbar = _error_bar(config.get("error_bar", "none"))
    if estimator is None:
        errorbar = None
    if plot_type == "line":
        sns.lineplot(x=x, y="__graph_value", style=style if style else None,
                     markers=bool(config.get("markers", True)), dashes=True,
                     estimator=estimator, errorbar=errorbar, linewidth=float(config.get("line_width", 2)),
                     **common)
    elif plot_type == "scatter":
        sns.scatterplot(x=x, y="__graph_value", style=style if style else None,
                        s=float(config.get("point_size", 55)), **common)
    elif plot_type == "bar":
        sns.barplot(x=x, y="__graph_value", estimator=estimator or "mean", errorbar=errorbar, **common)
    elif plot_type == "point":
        sns.pointplot(x=x, y="__graph_value", estimator=estimator or "mean", errorbar=errorbar,
                      markers="o" if config.get("markers", True) else "", **common)
    elif plot_type in ("box", "violin", "strip", "swarm"):
        function = {"box": sns.boxplot, "violin": sns.violinplot,
                    "strip": sns.stripplot, "swarm": sns.swarmplot}[plot_type]
        categorical = {key: value for key, value in common.items() if key != "alpha"}
        if plot_type in ("strip", "swarm"):
            categorical.update({"alpha": float(config.get("alpha", 0.9)),
                                "size": math.sqrt(float(config.get("point_size", 55)))})
        function(x=x, y="__graph_value", **categorical)
    elif plot_type in ("histogram", "kde", "ecdf"):
        function = {"histogram": sns.histplot, "kde": sns.kdeplot, "ecdf": sns.ecdfplot}[plot_type]
        options = {"x": "__graph_value", **common}
        if plot_type == "histogram":
            options.update({"bins": int(config.get("bins", 30)), "element": "step"})
        elif plot_type == "kde":
            options["fill"] = bool(config.get("fill", False))
        function(**options)
    elif plot_type == "regression":
        for series, values in long.groupby("__graph_series", dropna=False):
            sns.regplot(data=values, x=x, y="__graph_value", ax=axis, label=str(series),
                        scatter_kws={"s": float(config.get("point_size", 40)),
                                     "alpha": float(config.get("alpha", 0.9))})
    elif plot_type == "area":
        for series, values in long.groupby("__graph_series", dropna=False):
            values = values.sort_values(x)
            axis.fill_between(values[x], values["__graph_value"], alpha=float(config.get("alpha", 0.45)),
                              label=str(series))
    elif plot_type == "heatmap":
        if len(metrics) != 1 or not hue:
            raise ValueError("heatmap requires one metric and a categorical hue column")
        pivot = frame.pivot_table(index=hue, columns=x, values=metrics[0],
                                  aggfunc=estimator or "mean")
        cmap = palette if palette in plt.colormaps() else "viridis"
        sns.heatmap(pivot, cmap=cmap, annot=bool(config.get("annotations", False)), ax=axis)
    else:
        raise ValueError(f"unsupported plot type: {plot_type}")


def _bounds(axis, config: dict, suffix: str) -> None:
    axis.set_yscale(config.get(f"y_{suffix}_scale", "linear"))
    lower, upper = config.get(f"y_{suffix}_min"), config.get(f"y_{suffix}_max")
    if lower not in (None, "") or upper not in (None, ""):
        axis.set_ylim(None if lower in (None, "") else float(lower),
                      None if upper in (None, "") else float(upper))


def _render_frame(frame: pd.DataFrame, config: dict, output: Path) -> None:
    x = config.get("x")
    if not x or x not in frame.columns:
        raise ValueError("select a valid x-axis column")
    left = list(config.get("left_y", [])); right = list(config.get("right_y", []))
    if not left and not right:
        raise ValueError("select at least one left or right y-axis metric")
    if any(metric not in frame.columns for metric in (*left, *right)):
        raise ValueError("selected y-axis metric does not exist")
    left_type = config.get("left_type", "line"); right_type = config.get("right_type", "line")
    if left_type not in PLOT_TYPES or right_type not in PLOT_TYPES:
        raise ValueError("unknown plot type")
    theme = config.get("theme", "whitegrid"); context = config.get("context", "notebook")
    if theme not in THEMES or context not in CONTEXTS:
        raise ValueError("unknown Seaborn theme or context")
    sns.set_theme(style=theme, context=context, font_scale=float(config.get("font_scale", 1.0)))
    facet = config.get("facet") or None
    facets = [(None, frame)] if not facet else list(frame.groupby(facet, dropna=False))
    if len(facets) > 12:
        raise ValueError("facet selection creates more than 12 panels")
    columns = min(int(config.get("facet_columns", 2)), max(1, len(facets)))
    rows = math.ceil(len(facets) / columns)
    width, height = float(config.get("width", 10)), float(config.get("height", 6))
    dpi = int(config.get("dpi", 180))
    if not 3 <= width <= 40 or not 3 <= height <= 40 or not 72 <= dpi <= 600:
        raise ValueError("width/height must be 3–40 inches and DPI must be 72–600")
    figure, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False)
    flat_axes = axes.reshape(-1)
    for position, (facet_value, subset) in enumerate(facets):
        left_axis = flat_axes[position]
        right_axis = left_axis.twinx() if right else None
        _draw(left_axis, subset, config, left, left_type)
        if right_axis is not None:
            _draw(right_axis, subset, config, right, right_type, len(left))
        left_axis.set_xlabel(config.get("x_label") or x)
        left_axis.set_ylabel(config.get("left_label") or ", ".join(left))
        left_axis.set_xscale(config.get("x_scale", "linear"))
        _bounds(left_axis, config, "left")
        if right_axis is not None:
            right_axis.set_ylabel(config.get("right_label") or ", ".join(right))
            _bounds(right_axis, config, "right")
        if facet:
            left_axis.set_title(f"{facet} = {facet_value}")
        if config.get("grid", True):
            left_axis.grid(True, alpha=0.25)
        if config.get("legend", True):
            handles, labels = left_axis.get_legend_handles_labels()
            if left_axis.legend_:
                left_axis.legend_.remove()
            if right_axis is not None:
                right_handles, right_labels = right_axis.get_legend_handles_labels()
                handles += right_handles; labels += right_labels
                if right_axis.legend_:
                    right_axis.legend_.remove()
            if handles:
                left_axis.legend(handles, labels, loc=config.get("legend_location", "best"),
                                 fontsize=float(config.get("legend_font_size", 8)))
    for axis in flat_axes[len(facets):]:
        axis.remove()
    figure.suptitle(config.get("title", ""), fontsize=float(config.get("title_size", 14)))
    figure.tight_layout()
    figure.savefig(output / "graph.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output / "graph.svg", bbox_inches="tight")
    plt.close(figure)


def render(results: Path, request: dict) -> dict:
    source = request.get("source", "summary")
    frame = load_data(results, list(request.get("runs", [])), source)
    frame = _filtered(frame, request.get("filters", {}))
    graph_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    output = results / "_graph_lab" / graph_id; output.mkdir(parents=True)
    config = {**request, "graph_id": graph_id, "created_at": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    frame.to_csv(output / "data.csv", index=False)
    _render_frame(frame, config, output)
    metadata = {"graph_id": graph_id, "rows": len(frame), "columns": list(frame.columns),
                "matplotlib": matplotlib.__version__, "seaborn": sns.__version__,
                "pandas": pd.__version__, "numpy": np.__version__}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "reproduce.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "directory = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(directory.parents[2]))\n"
        "from webui.graph_lab import render_saved\n"
        "render_saved(directory)\n", encoding="utf-8")
    return {**metadata, "base": f"_graph_lab/{graph_id}",
            "png": f"_graph_lab/{graph_id}/graph.png", "svg": f"_graph_lab/{graph_id}/graph.svg",
            "data": f"_graph_lab/{graph_id}/data.csv", "config": f"_graph_lab/{graph_id}/config.json"}


def render_saved(directory: Path) -> None:
    config = json.loads((directory / "config.json").read_text())
    frame = pd.read_csv(directory / "data.csv")
    _render_frame(frame, config, directory)


def saved_graphs(results: Path) -> list[dict]:
    graphs = []
    for config_path in sorted((results / "_graph_lab").glob("*/config.json"), reverse=True):
        try:
            if not (config_path.parent / "graph.png").is_file():
                continue
            config = json.loads(config_path.read_text())
            graphs.append({"id": config["graph_id"], "title": config.get("title") or "Untitled graph",
                           "created_at": config.get("created_at"),
                           "png": f'_graph_lab/{config["graph_id"]}/graph.png',
                           "svg": f'_graph_lab/{config["graph_id"]}/graph.svg',
                           "data": f'_graph_lab/{config["graph_id"]}/data.csv',
                           "config": f'_graph_lab/{config["graph_id"]}/config.json'})
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return graphs
