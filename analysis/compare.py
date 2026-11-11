"""Compare the PyTorch and TensorFlow results from Brooks.

Reads the two runs CSVs under results/, merges them into one table and
writes a per-condition summary plus a side-by-side framework comparison.

    python analysis/compare.py
"""

import argparse
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_FILES = {
    "tensorflow": REPO / "results" / "tensorflow" / "tensorflow_runs.csv",
    "pytorch": REPO / "results" / "pytorch" / "pytorch_runs.csv",
}
FRAMEWORKS = ["pytorch", "tensorflow"]
PHASES = ["train", "inference"]

# columns that identify one experimental condition
CONDITION = ["study", "model", "threads", "batch_size", "phase"]

# metrics that get a median/min/max in the summary tables
METRICS = [
    "elapsed_seconds", "images_per_second", "test_accuracy",
    "ipc", "cache_miss_rate", "branch_miss_rate",
    "cpu_utilization", "effective_ghz",
    "instructions_per_sample", "cycles_per_sample",
    "cache_misses_per_sample", "branch_misses_per_sample",
    "context-switches", "cpu-migrations", "page-faults",
]

# metrics shown side by side in the framework comparison
COMPARED = [
    "elapsed_seconds", "images_per_second", "test_accuracy", "ipc",
    "cycles_per_sample", "instructions_per_sample", "effective_ghz",
    "cpu_utilization", "context-switches",
]

TEXT_COLUMNS = {
    "framework", "framework_version", "python_version", "machine", "study",
    "model", "phase", "perf_scope", "perf_problem",
}
INT_COLUMNS = {"threads", "batch_size", "trial", "run_order", "epochs",
               "parameters", "samples", "steps"}


def convert(column, text):
    # CSV cells are strings, turn the numeric ones into numbers
    if column in TEXT_COLUMNS:
        return text
    if text == "":
        return None
    return int(float(text)) if column in INT_COLUMNS else float(text)


def load_runs(run_files=RUN_FILES):
    # read both frameworks into one list, keeping the shared column order
    rows, fields = [], None
    for framework in FRAMEWORKS:
        with open(run_files[framework], newline="") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise SystemExit(f"{run_files[framework]} has different columns")
            for row in reader:
                rows.append({k: convert(k, v) for k, v in row.items()})
    return rows, fields


def select(rows, **wanted):
    # rows whose columns match every keyword, e.g. study="threads"
    return [r for r in rows if all(r[k] == v for k, v in wanted.items())]


def values(rows, metric):
    return [r[metric] for r in rows if r[metric] is not None]


def spread(rows, metric):
    # median, lowest and highest value of a metric across trials
    v = values(rows, metric)
    if not v:
        return None, None, None
    return statistics.median(v), min(v), max(v)


def conditions(rows):
    # unique conditions in the order they first appear in the data
    seen = []
    for r in rows:
        key = tuple(r[c] for c in CONDITION)
        if key not in seen:
            seen.append(key)
    return seen


def summarize(rows):
    # one line per framework and condition with median/min/max per metric
    out = []
    for framework in FRAMEWORKS:
        mine = select(rows, framework=framework)
        for key in conditions(mine):
            group = select(mine, **dict(zip(CONDITION, key)))
            line = {"framework": framework, **dict(zip(CONDITION, key)),
                    "trials": len(group)}
            for metric in METRICS:
                median, low, high = spread(group, metric)
                line[f"{metric}_median"] = median
                line[f"{metric}_min"] = low
                line[f"{metric}_max"] = high
            out.append(line)
    return out


def compare(rows):
    # put the two frameworks' medians next to each other per condition
    out = []
    for key in conditions(rows):
        where = dict(zip(CONDITION, key))
        line = dict(where)
        for metric in COMPARED:
            medians = {}
            for framework in FRAMEWORKS:
                group = select(rows, framework=framework, **where)
                medians[framework] = spread(group, metric)[0]
                line[f"{framework}_{metric}"] = medians[framework]
            if metric == "elapsed_seconds" and all(medians.values()):
                line["tensorflow_over_pytorch_time"] = (
                    medians["tensorflow"] / medians["pytorch"])
        out.append(line)
    return out


def tidy(value, digits):
    # blanks for missing values, floats cut to the wanted precision
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return value


def write_csv(path, rows, fields=None, digits=6):
    fields = fields or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: tidy(row.get(k), digits) for k in fields})
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=REPO / "results",
                        help="folder for the merged and summary CSVs")
    a = parser.parse_args()

    rows, fields = load_runs()
    # the merged table keeps full precision, the summaries are rounded
    write_csv(a.output / "combined_runs.csv", rows, fields, digits=12)
    write_csv(a.output / "comparison_summary.csv", summarize(rows))
    write_csv(a.output / "framework_comparison.csv", compare(rows))


if __name__ == "__main__":
    main()
