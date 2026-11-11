"""Draw the comparison figures from the Brooks results.

Uses the loaders in compare.py and writes one PNG per study to
analysis/figures/. Points and bars are medians of the trials, and the
error bars show the lowest and highest trial.

    pip install -r analysis/requirements.txt
    python analysis/figures.py
"""

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from compare import FRAMEWORKS, REPO, load_runs, select, spread, values

NAMES = {"pytorch": "PyTorch", "tensorflow": "TensorFlow"}
COLORS = {"pytorch": "#2a78d6", "tensorflow": "#eb6834"}
SURFACE, INK, INK_SOFT = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_SOFT,
    "axes.edgecolor": AXIS, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def short(value, _=None):
    # compact number for tick and bar labels: 23535 -> 23.5k
    v = abs(value)
    if v >= 1e6:
        return f"{value / 1e6:.3g}M"
    if v >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:.3g}"


def point(rows, metric, scale=1.0):
    # median plus the distance down to the min and up to the max
    median, low, high = spread(rows, metric)
    return median * scale, (median - low) * scale, (high - median) * scale


def line_panel(ax, rows, xcol, metric, phase, title, scale=1.0, log_y=False,
               ticks=None):
    # one line per framework over a swept setting (threads or batch size)
    xs = sorted({r[xcol] for r in rows})
    for fw in FRAMEWORKS:
        pts = [point(select(rows, framework=fw, phase=phase, **{xcol: x}),
                     metric, scale) for x in xs]
        ax.errorbar(range(len(xs)), [p[0] for p in pts],
                    yerr=[[p[1] for p in pts], [p[2] for p in pts]],
                    color=COLORS[fw], marker="o", capsize=2.5, elinewidth=1,
                    markeredgecolor=SURFACE, markeredgewidth=1,
                    label=NAMES[fw])
    finish_lines(ax, xs, title, log_y, ticks)


def finish_lines(ax, xs, title, log_y, ticks=None):
    # evenly spaced ticks, since the sweeps double at every step
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if ticks:
        ax.set_yticks(ticks)
        ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(short))


def bar_panel(ax, rows, groups, metric, phase, title, scale=1.0, xcol=None):
    # grouped bars: one group per value of xcol, one bar per framework
    width = 0.38 if xcol else 0.6
    for i, fw in enumerate(FRAMEWORKS):
        for g, group in enumerate(groups):
            where = {xcol: group} if xcol else {}
            mid, down, up = point(
                select(rows, framework=fw, phase=phase, **where), metric, scale)
            x = g + (i - 0.5) * (width + 0.03) if xcol else i
            ax.bar(x, mid, width, color=COLORS[fw], yerr=[[down], [up]],
                   error_kw={"elinewidth": 1, "ecolor": INK_SOFT, "capsize": 2},
                   label=NAMES[fw] if g == 0 else None)
            ax.annotate(short(mid), (x, mid + up), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)
    if xcol:
        ax.set_xticks(range(len(groups)), groups)
    else:
        ax.set_xticks(range(len(FRAMEWORKS)), [NAMES[f] for f in FRAMEWORKS])
    ax.set_title(title)
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(short))
    ax.tick_params(axis="x", length=0)


def save(fig, out, name, title, note, legend_from=None):
    # shared heading, one legend for the whole figure, then write the PNG
    # heading offsets are in inches so short and tall figures match
    height = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.12 / height, ha="left", va="top",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.48 / height, note, ha="left", va="top",
             fontsize=9.5, color=INK_SOFT)
    if legend_from is not None:
        handles, labels = legend_from.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncols=2,
                   bbox_to_anchor=(0.995, 1 - 0.05 / height))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print(f"wrote {out / name}")


def baseline_figure(rows, out):
    # headline comparison at the baseline settings, one metric per panel
    rows = select(rows, study="baseline")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    panels = [
        ("elapsed_seconds", "train", "Training time (s)", 1),
        ("elapsed_seconds", "inference", "Inference time (ms)", 1000),
        ("cycles_per_sample", "train", "Training cycles per image", 1),
        ("cycles_per_sample", "inference", "Inference cycles per image", 1),
        ("ipc", "train", "Training IPC", 1),
        ("ipc", "inference", "Inference IPC", 1),
        ("effective_ghz", "train", "Training clock speed (GHz)", 1),
        ("context-switches", "train", "Training context switches", 1),
    ]
    for ax, (metric, phase, title, scale) in zip(axes.flat, panels):
        bar_panel(ax, rows, [None], metric, phase, title, scale)
    save(fig, out, "baseline.png", "Baseline: 784-128-10, 1 thread, batch 64",
         "Median of 3 trials, error bars show min and max. "
         "5 training epochs (300,000 images), 10,000 test images.")


def speedup_panel(ax, rows, title):
    # training speedup over the same framework's 1-thread median
    xs = sorted({r["threads"] for r in rows})
    for fw in FRAMEWORKS:
        mine = select(rows, framework=fw, phase="train")
        one = statistics.median(values(select(mine, threads=1),
                                       "elapsed_seconds"))
        mid, down, up = [], [], []
        for x in xs:
            median, low, high = spread(select(mine, threads=x),
                                       "elapsed_seconds")
            mid.append(one / median)
            down.append(one / median - one / high)
            up.append(one / low - one / median)
        ax.errorbar(range(len(xs)), mid, yerr=[down, up], color=COLORS[fw],
                    marker="o", capsize=2.5, elinewidth=1,
                    markeredgecolor=SURFACE, markeredgewidth=1,
                    label=NAMES[fw])
    ax.axhline(1, color=AXIS, linewidth=1)
    finish_lines(ax, xs, title, log_y=False)


def threads_figure(rows, out):
    # what adding threads does to time and to the work the CPU performs
    rows = select(rows, study="threads")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    a = axes.flat
    line_panel(a[0], rows, "threads", "elapsed_seconds", "train",
               "Training time (s)")
    speedup_panel(a[1], rows, "Training speedup over 1 thread")
    line_panel(a[2], rows, "threads", "elapsed_seconds", "inference",
               "Inference time (ms, log scale)", scale=1000, log_y=True,
               ticks=[10, 20, 50, 100, 200, 400])
    line_panel(a[3], rows, "threads", "cpu_utilization", "train",
               "Cores kept busy in training (log scale)", log_y=True)
    line_panel(a[4], rows, "threads", "cycles_per_sample", "train",
               "Training cycles per image (log scale)", log_y=True)
    line_panel(a[5], rows, "threads", "context-switches", "train",
               "Training context switches (log scale)", log_y=True)
    for ax in a[3:]:
        ax.set_xlabel("threads")
    save(fig, out, "threads.png", "Thread scaling: 1 to 128 threads",
         "Baseline model, batch 64. Median of 3 trials, error bars show "
         "min and max.", legend_from=a[0])


def batch_figure(rows, out):
    # throughput and per-image cost as the batch size grows
    rows = select(rows, study="batch")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    a = axes.flat
    line_panel(a[0], rows, "batch_size", "images_per_second", "train",
               "Training throughput (images/s)")
    line_panel(a[1], rows, "batch_size", "images_per_second", "inference",
               "Inference throughput (images/s)")
    line_panel(a[2], rows, "batch_size", "test_accuracy", "train",
               "Test accuracy after 5 epochs (%)", scale=100)
    a[2].set_ylim(88, 95)
    line_panel(a[3], rows, "batch_size", "instructions_per_sample", "train",
               "Training instructions per image")
    line_panel(a[4], rows, "batch_size", "ipc", "train", "Training IPC")
    line_panel(a[5], rows, "batch_size", "effective_ghz", "train",
               "Training clock speed (GHz)")
    for ax in a[3:]:
        ax.set_xlabel("batch size")
    save(fig, out, "batch.png", "Batch size: 32 to 256",
         "Baseline model, 1 thread. Median of 3 trials, error bars show "
         "min and max.", legend_from=a[0])


def models_figure(rows, out):
    # the three network shapes side by side
    rows = select(rows, study="models")
    names = ["baseline", "wide", "deep"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    panels = [
        ("elapsed_seconds", "train", "Training time (s)", 1),
        ("elapsed_seconds", "inference", "Inference time (ms)", 1000),
        ("ipc", "train", "Training IPC", 1),
        ("cache_miss_rate", "train", "Training cache miss rate (%)", 100),
    ]
    for ax, (metric, phase, title, scale) in zip(axes, panels):
        bar_panel(ax, rows, names, metric, phase, title, scale, xcol="model")
    save(fig, out, "models.png",
         "Model shape: baseline 784-128-10, wide 784-256-10, deep 784-128-64-10",
         "1 thread, batch 64. Median of 3 trials, error bars show min and max.",
         legend_from=axes[0])


def clock_figure(rows, out):
    # every 1-thread, batch-64 baseline-model training run, from all studies
    rows = select(rows, model="baseline", threads=1, batch_size=64,
                  phase="train")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [("elapsed_seconds", "Training time (s)"),
              ("cycles_per_sample", "Cycles per image")]
    for ax, (metric, title) in zip(axes, panels):
        for fw in FRAMEWORKS:
            mine = select(rows, framework=fw)
            ax.scatter(values(mine, "effective_ghz"), values(mine, metric),
                       s=42, color=COLORS[fw], edgecolor=SURFACE,
                       linewidth=1, label=NAMES[fw], zorder=3)
        ax.set_title(title)
        ax.set_xlabel("clock speed during the run (GHz)")
        ax.set_ylim(bottom=0)
        ax.margins(y=0.15)
        ax.yaxis.set_major_formatter(FuncFormatter(short))
    save(fig, out, "clock.png", "Clock speed explains part of the time gap",
         "Each dot is one training run with the baseline settings "
         "(12 per framework).", legend_from=axes[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path,
                        default=REPO / "analysis" / "figures",
                        help="folder for the PNG files")
    a = parser.parse_args()

    rows, _ = load_runs()
    for figure in (baseline_figure, threads_figure, batch_figure,
                   models_figure, clock_figure):
        figure(rows, a.output)


if __name__ == "__main__":
    main()
