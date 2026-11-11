"""Benchmarks MNIST training and inference in PyTorch on the CPU.
Runs the baseline, thread, batch size and model studies, each run in its own process."""

import argparse
import copy
import csv
import gzip
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from statistics import mean, median, stdev

# Hidden layer sizes for each model; all take 784 pixels in and give 10 digit scores.
MODELS = {
    "baseline": (128,),     # 784-128-10
    "wide": (256,),         # 784-256-10
    "deep": (128, 64),      # 784-128-64-10
}
THREAD_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
BATCH_SWEEP = (32, 64, 128, 256)
STUDIES = ("baseline", "threads", "batch", "models")

# Fixed seed shared with the TensorFlow script, so every trial starts from the same weights.
SEED = 42

# Original MNIST files (the same 60,000 / 10,000 split Keras uses).
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

# Columns written for every run (same as the TensorFlow script), and the ones averaged
# in the summary file.
FIELDS = [
    "framework", "framework_version", "python_version", "machine",
    "study", "model", "parameters", "threads", "batch_size", "epochs",
    "learning_rate", "trial", "run_order", "phase",
    "elapsed_seconds", "warmup_seconds", "seconds_per_epoch",
    "samples", "steps", "images_per_second", "test_accuracy", "load_avg_1min",
]
NUMERIC_FIELDS = [
    "elapsed_seconds", "warmup_seconds", "seconds_per_epoch",
    "images_per_second", "test_accuracy",
]


def build_parser():
    """Defines the command line options; the hidden ones are only passed to worker runs."""
    p = argparse.ArgumentParser(description="MNIST PyTorch CPU experiments")
    p.add_argument("--study", choices=[*STUDIES, "all"], default="baseline")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--train-samples", type=int, default=60000,
                   help="train on fewer images (only for quick tests)")
    p.add_argument("--output", default="torch_results")
    p.add_argument("--data-dir", default=str(Path.home() / "mnist-project" / "data"),
                   help="where the MNIST files are stored (downloaded if missing)")
    p.add_argument("--thread-list", default=",".join(map(str, THREAD_SWEEP)),
                   help="thread counts for the threads study")
    p.add_argument("--batch-list", default=",".join(map(str, BATCH_SWEEP)),
                   help="batch sizes for the batch study")
    p.add_argument("--model-list", default=",".join(MODELS),
                   help="models for the models study: " + ", ".join(MODELS))

    hidden = argparse.SUPPRESS
    p.add_argument("--worker", action="store_true", help=hidden)
    p.add_argument("--model", choices=list(MODELS), default="baseline", help=hidden)
    p.add_argument("--phase", choices=["train", "inference"], default="train", help=hidden)
    p.add_argument("--trial", type=int, default=1, help=hidden)
    p.add_argument("--study-name", default="baseline", help=hidden)
    p.add_argument("--row-file", default="row.csv", help=hidden)
    p.add_argument("--weights-file", default="model.pt", help=hidden)
    return p


def int_list(text):
    """Turns "1,2,4" into [1, 2, 4]."""
    return [int(x) for x in text.split(",") if x.strip()]


def check_args(p, a):
    """Rejects bad values and converts the sweep options from text into lists."""
    if min(a.trials, a.epochs, a.batch_size, a.threads, a.train_samples) <= 0:
        p.error("trials, epochs, batch-size, threads and train-samples must be positive")
    if a.train_samples > 60000:
        p.error("MNIST only has 60000 training images")
    if a.learning_rate <= 0:
        p.error("learning-rate must be positive")
    try:
        a.thread_list = int_list(a.thread_list)
        a.batch_list = int_list(a.batch_list)
    except ValueError:
        p.error("--thread-list and --batch-list take comma-separated integers")
    a.model_list = [m.strip() for m in a.model_list.split(",") if m.strip()]
    unknown = [m for m in a.model_list if m not in MODELS]
    if unknown:
        p.error(f"unknown model(s) {unknown}, pick from: {', '.join(MODELS)}")
    if not (a.thread_list and a.batch_list and a.model_list):
        p.error("thread, batch and model lists can't be empty")
    if min(a.thread_list + a.batch_list) <= 0:
        p.error("thread counts and batch sizes must be positive")
    a.data_dir = Path(a.data_dir).expanduser()
    return a


def parse_args(parser=None):
    p = parser or build_parser()
    return check_args(p, p.parse_args())


def download_mnist(data_dir):
    """Downloads any missing MNIST file, so no download ever happens inside a run."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = data_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(MNIST_URL + filename, path)


def setup_torch(threads, seed):
    """Limits PyTorch and its math libraries to `threads` CPU threads, then imports it.
    The environment variables only work if they are set before the import."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)

    import numpy as np
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch


def read_idx(path, magic_number, header_size):
    """Reads one gzipped MNIST file and returns its data as a NumPy array.
    Based on Breylan's loader from the team's first PyTorch script."""
    import numpy as np

    with gzip.open(path, "rb") as f:
        header = struct.unpack(">" + "I" * (header_size // 4), f.read(header_size))
        if header[0] != magic_number:
            raise ValueError(f"unexpected file format: {path}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(torch, data_dir, train_samples):
    """Loads MNIST with each image flattened to 784 float32 pixels scaled to [0, 1],
    the same preprocessing as the TensorFlow script."""
    import numpy as np

    images = {k: read_idx(data_dir / MNIST_FILES[k], 2051, 16).reshape(-1, 784)
              for k in ("x_train", "x_test")}
    labels = {k: read_idx(data_dir / MNIST_FILES[k], 2049, 8).astype(np.int64)
              for k in ("y_train", "y_test")}

    x_train = torch.from_numpy(images["x_train"][:train_samples].astype(np.float32) / 255.0)
    y_train = torch.from_numpy(labels["y_train"][:train_samples])
    x_test = torch.from_numpy(images["x_test"].astype(np.float32) / 255.0)
    y_test = labels["y_test"]
    return x_train, y_train, x_test, y_test


def build_model(torch, name):
    """Builds a fully connected ReLU network with the same starting weights as Keras:
    Glorot (Xavier) uniform weights and zero biases."""
    from torch import nn

    layers, width_in = [], 784
    for width in MODELS[name]:
        layers += [nn.Linear(width_in, width), nn.ReLU()]
        width_in = width
    layers.append(nn.Linear(width_in, 10))
    model = nn.Sequential(*layers)

    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


def train(torch, model, x, y, batch_size, learning_rate, epochs, max_batches=None):
    """Runs SGD with cross-entropy loss over the training data in order (no shuffling),
    taking each batch as a slice of the data tensors."""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for i, start in enumerate(range(0, len(x), batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[start:start + batch_size]), y[start:start + batch_size])
            loss.backward()
            optimizer.step()


def predict(torch, model, x, batch_size):
    """Runs the model over `x` one batch at a time and returns the predicted digits,
    converting each batch's output to NumPy like the TensorFlow script does."""
    import numpy as np

    model.eval()
    labels = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            scores = model(x[start:start + batch_size]).numpy()
            labels.append(np.argmax(scores, axis=1))
    return np.concatenate(labels)


class Stopwatch:
    """Measures wall-clock time for the code inside a `with` block."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def run_worker(a, stopwatch=Stopwatch):
    """Times one phase (training or inference) and writes its results to a.row_file.
    Training saves its weights so the matching inference run uses the same model."""
    torch = setup_torch(a.threads, seed=SEED)
    import numpy as np

    x_train, y_train, x_test, y_test = load_mnist(torch, a.data_dir, a.train_samples)
    model = build_model(torch, a.model)

    if a.phase == "train":
        # Warm-up: run 2 batches untimed to match the TensorFlow script, then restore the
        # starting weights so the timed run begins from scratch.
        start_weights = copy.deepcopy(model.state_dict())
        t0 = time.perf_counter()
        train(torch, model, x_train, y_train, a.batch_size, a.learning_rate, 1, max_batches=2)
        warmup = time.perf_counter() - t0
        model.load_state_dict(start_weights)

        # Timed training over all epochs, then accuracy is checked outside the timer.
        with stopwatch() as timer:
            train(torch, model, x_train, y_train, a.batch_size, a.learning_rate, a.epochs)
        torch.save(model.state_dict(), a.weights_file)

        predictions = predict(torch, model, x_test, a.batch_size)
        samples = a.train_samples * a.epochs
        steps = math.ceil(a.train_samples / a.batch_size) * a.epochs
    else:
        model.load_state_dict(torch.load(a.weights_file))

        # Warm-up with one untimed batch, then time predictions on all 10,000 test images.
        t0 = time.perf_counter()
        predict(torch, model, x_test[:a.batch_size], a.batch_size)
        warmup = time.perf_counter() - t0

        with stopwatch() as timer:
            predictions = predict(torch, model, x_test, a.batch_size)
        samples = len(x_test)
        steps = math.ceil(len(x_test) / a.batch_size)

    # Result row: settings plus timing, throughput and accuracy for this phase.
    elapsed = timer.seconds
    accuracy = float(np.mean(predictions == y_test))
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "study": a.study_name,
        "model": a.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "threads": a.threads,
        "batch_size": a.batch_size,
        "epochs": a.epochs,
        "learning_rate": a.learning_rate,
        "trial": a.trial,
        "phase": a.phase,
        "elapsed_seconds": round(elapsed, 6),
        "warmup_seconds": round(warmup, 6),
        "seconds_per_epoch": round(elapsed / a.epochs, 6) if a.phase == "train" else "",
        "samples": samples,    # images processed while timed
        "steps": steps,        # weight updates (train) or batches predicted (inference)
        "images_per_second": round(samples / elapsed, 2),
        "test_accuracy": round(accuracy, 6),
    })
    with open(a.row_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    print(f"{a.phase}: {elapsed:.3f}s (warm-up {warmup:.3f}s not counted), "
          f"{samples / elapsed:.1f} images/s, accuracy {accuracy:.2%}")


def usable_cpus():
    """Counts the CPUs this process is allowed to use (respects taskset/cgroup limits)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def plan_runs(a, cpus):
    """Lists the (study, settings) pairs to run; each study changes one setting from the
    baseline, and thread counts above the available CPUs are left out."""
    base = {"threads": a.threads, "batch_size": a.batch_size, "model": "baseline"}
    studies = STUDIES if a.study == "all" else [a.study]
    plan = []
    for study in studies:
        if study == "baseline":
            changes = [{}]
        elif study == "threads":
            changes = [{"threads": n} for n in a.thread_list if n <= cpus]
        elif study == "batch":
            changes = [{"batch_size": n} for n in a.batch_list]
        else:
            changes = [{"model": m} for m in a.model_list]
        plan += [(study, {**base, **c}) for c in changes]
    return plan


def read_row(path):
    with open(path, newline="") as f:
        return next(csv.DictReader(f))


def run_in_new_process(worker_args, tag, script=None, wrapper=()):
    """Starts a fresh Python process for one run and returns the result row it writes.
    `wrapper` is an optional command placed in front of python, such as a profiler."""
    script = Path(script or __file__).resolve()
    with tempfile.TemporaryDirectory(prefix="torch_run_") as temp:
        row_file = Path(temp) / "row.csv"
        cmd = [*wrapper, sys.executable, str(script), "--worker",
               *worker_args, "--row-file", str(row_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"{tag} failed, see the error output above")
        print(result.stdout.strip())
        return read_row(row_file)


def read_text(path):
    """Returns a file's contents, or "" if it can't be read."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def write_system_info(out, cpus, extra=""):
    """Saves the machine, OS, CPU and library details to system_info.txt for the report."""
    lines = [
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Machine: {platform.machine()}",
        f"Logical CPUs usable by this process: {cpus} (os.cpu_count() = {os.cpu_count()})",
    ]

    # Library versions and PyTorch's math-library setup, read in a separate process.
    versions = subprocess.run(
        [sys.executable, "-c",
         "import numpy, torch; "
         "print('torch', torch.__version__); "
         "print('numpy', numpy.__version__); "
         "print(torch.__config__.parallel_info())"],
        capture_output=True, text=True,
    )
    lines += ["", "== library versions ==", (versions.stdout or versions.stderr).strip()]

    # OS release plus CPU frequency governor and turbo settings, which affect timing.
    for path in ("/etc/os-release",
                 "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                 "/sys/devices/system/cpu/intel_pstate/no_turbo",
                 "/sys/devices/system/cpu/cpufreq/boost"):
        text = read_text(path)
        if text:
            lines += ["", f"== {path} ==", text]

    # Kernel, CPU layout and memory size.
    for cmd in (["uname", "-a"], ["lscpu"], ["free", "-h"]):
        if shutil.which(cmd[0]):
            result = subprocess.run(cmd, capture_output=True, text=True)
            lines += ["", "$ " + " ".join(cmd), (result.stdout or result.stderr).strip()]

    if extra:
        lines += ["", extra]
    (out / "system_info.txt").write_text("\n".join(lines) + "\n")


def save_runs(path, rows, fields):
    """Writes one CSV row per run."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path, rows, numeric_fields):
    """Groups runs by condition and writes the mean, std, median, min and max of each
    numeric column over the trials."""
    group_by = ("study", "model", "threads", "batch_size", "epochs", "phase")
    stats = ("mean", "std", "median", "min", "max")
    groups = {}
    for row in rows:
        key = tuple(str(row[k]) for k in group_by)
        groups.setdefault(key, []).append(row)

    columns = [*group_by, "trials"]
    columns += [f"{field}_{s}" for field in numeric_fields for s in stats]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for key, group in groups.items():
            entry = dict(zip(group_by, key))
            entry["trials"] = len(group)
            for field in numeric_fields:
                values = [float(r[field]) for r in group if r.get(field) not in ("", None)]
                if not values:
                    continue    # column is blank for this phase, e.g. seconds_per_epoch
                entry[f"{field}_mean"] = round(mean(values), 6)
                entry[f"{field}_std"] = round(stdev(values), 6) if len(values) > 1 else ""
                entry[f"{field}_median"] = round(median(values), 6)
                entry[f"{field}_min"] = round(min(values), 6)
                entry[f"{field}_max"] = round(max(values), 6)
            writer.writerow(entry)


def run_study(a, run_once=run_in_new_process, fields=FIELDS, numeric_fields=NUMERIC_FIELDS):
    """Runs every planned condition for each trial (training, then inference) and
    saves the results after each run."""
    out = Path(a.output).resolve()
    (out / "weights").mkdir(parents=True, exist_ok=True)
    cpus = usable_cpus()
    if a.threads > cpus:
        sys.exit(f"--threads {a.threads} is more than the {cpus} CPUs available")
    download_mnist(a.data_dir)
    write_system_info(out, cpus)
    print("System details written to", out / "system_info.txt")

    plan = plan_runs(a, cpus)
    too_many = [n for n in a.thread_list if n > cpus]
    if too_many and a.study in ("threads", "all"):
        print(f"Skipping thread counts {too_many}, only {cpus} CPUs available")

    total = len(plan) * a.trials * 2
    rows = []
    order = 0
    for trial in range(1, a.trials + 1):
        # Odd trials go through the plan forwards and even trials backwards,
        # so slow drift like CPU heat or server load doesn't always hit the same settings.
        for study, s in (plan if trial % 2 else plan[::-1]):
            name = f"{study}_{s['model']}_t{s['threads']}_b{s['batch_size']}_r{trial}"
            weights = out / "weights" / f"{name}.pt"
            for phase in ("train", "inference"):
                order += 1
                tag = f"{name}_{phase}"
                load = os.getloadavg()[0]
                print(f"[{order}/{total}] {tag} (load avg {load:.1f})", flush=True)
                row = run_once([
                    "--model", s["model"],
                    "--threads", str(s["threads"]),
                    "--batch-size", str(s["batch_size"]),
                    "--epochs", str(a.epochs),
                    "--learning-rate", str(a.learning_rate),
                    "--train-samples", str(a.train_samples),
                    "--data-dir", str(a.data_dir),
                    "--phase", phase,
                    "--study-name", study,
                    "--trial", str(trial),
                    "--weights-file", str(weights),
                ], tag)
                row["run_order"] = order
                row["load_avg_1min"] = round(load, 2)
                rows.append(row)

                # Both CSVs are rewritten after each run, so finished results survive a crash.
                save_runs(out / "pytorch_runs.csv", rows, fields)
                save_summary(out / "pytorch_summary.csv", rows, numeric_fields)

    print("\nDone. Results are in", out)
    print("  pytorch_runs.csv     one row per run and phase")
    print("  pytorch_summary.csv  mean/std/median/min/max per condition")
    print("  system_info.txt      machine, OS and library versions")
    return rows


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_study(args)
