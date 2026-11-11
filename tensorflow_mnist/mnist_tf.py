"""MNIST training/inference benchmark in TensorFlow (CS3339, Team 6).

Runs the experiments from section 1.3 of our proposal: a baseline run plus
thread, batch size and model sweeps, all on the CPU. Every measurement runs
in a new Python process so the thread settings really apply to it.

    python mnist_tf.py --study baseline --trials 1     # quick pilot
    python mnist_tf.py --study all --trials 3          # everything
"""

import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median, stdev

# Hidden layer widths. Every model takes 784 inputs and ends in 10 logits.
MODELS = {
    "baseline": (128,),     # 784-128-10
    "wide": (256,),         # 784-256-10
    "deep": (128, 64),      # 784-128-64-10
}
THREAD_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
BATCH_SWEEP = (32, 64, 128, 256)
STUDIES = ("baseline", "threads", "batch", "models")

# Same seed for every run, in both frameworks, so all trials of a condition
# start from the same weights and only the timing/counters vary.
SEED = 42

FIELDS = [
    "framework", "framework_version", "python_version", "machine",
    "study", "model", "parameters", "threads", "batch_size", "epochs",
    "learning_rate", "trial", "run_order", "phase",
    "elapsed_seconds", "warmup_seconds", "seconds_per_epoch",
    "samples", "steps", "images_per_second", "test_accuracy", "load_avg_1min",
]
# columns that get mean/std/median/min/max in the summary file
NUMERIC_FIELDS = [
    "elapsed_seconds", "warmup_seconds", "seconds_per_epoch",
    "images_per_second", "test_accuracy",
]


def build_parser():
    p = argparse.ArgumentParser(description="MNIST TensorFlow CPU experiments")
    p.add_argument("--study", choices=[*STUDIES, "all"], default="baseline")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--train-samples", type=int, default=60000,
                   help="train on fewer images (only for quick tests)")
    p.add_argument("--output", default="tf_results")
    p.add_argument("--thread-list", default=",".join(map(str, THREAD_SWEEP)),
                   help="thread counts for the threads study")
    p.add_argument("--batch-list", default=",".join(map(str, BATCH_SWEEP)),
                   help="batch sizes for the batch study")
    p.add_argument("--model-list", default=",".join(MODELS),
                   help="models for the models study: " + ", ".join(MODELS))

    # Only used when the script starts itself as a worker for a single run.
    hidden = argparse.SUPPRESS
    p.add_argument("--worker", action="store_true", help=hidden)
    p.add_argument("--model", choices=list(MODELS), default="baseline", help=hidden)
    p.add_argument("--phase", choices=["train", "inference"], default="train", help=hidden)
    p.add_argument("--trial", type=int, default=1, help=hidden)
    p.add_argument("--study-name", default="baseline", help=hidden)
    p.add_argument("--row-file", default="row.csv", help=hidden)
    p.add_argument("--weights-file", default="model.weights.h5", help=hidden)
    return p


def int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def check_args(p, a):
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
    return a


def parse_args(parser=None):
    p = parser or build_parser()
    return check_args(p, p.parse_args())


def setup_tensorflow(threads, seed):
    # The thread variables have to be set before TensorFlow (and NumPy) start
    # their thread pools, so TensorFlow is only imported here.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    return tf


def load_mnist(tf, train_samples):
    """Flattened float32 images scaled to [0, 1], same as the PyTorch version."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:train_samples].reshape(-1, 784).astype("float32") / 255.0
    y_train = y_train[:train_samples]
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def build_model(tf, name, learning_rate):
    layers = [tf.keras.Input(shape=(784,))]
    for width in MODELS[name]:
        layers.append(tf.keras.layers.Dense(width, activation="relu"))
    layers.append(tf.keras.layers.Dense(10))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def training_dataset(tf, x, y, batch_size, threads):
    # If fit() is given NumPy arrays, Keras builds its own tf.data pipeline
    # with AUTOTUNE, and that uses extra threads sized to the whole machine.
    # Building it ourselves keeps the 1-thread runs actually single threaded.
    data = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = threads
    options.threading.max_intra_op_parallelism = 1
    return data.with_options(options)


def predict(model, x, batch_size):
    import numpy as np

    labels = []
    for start in range(0, len(x), batch_size):
        scores = model(x[start:start + batch_size], training=False).numpy()
        labels.append(np.argmax(scores, axis=1))
    return np.concatenate(labels)


class Stopwatch:
    """Wall-clock timer for the part of a run we actually measure."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def run_worker(a, stopwatch=Stopwatch):
    """One measurement: either training or inference, never both.

    Start-up, data loading and warm-up happen outside the stopwatch.
    The train run saves its weights so the inference run of the same
    trial measures the same trained model.
    """
    tf = setup_tensorflow(a.threads, seed=SEED)
    import numpy as np

    x_train, y_train, x_test, y_test = load_mnist(tf, a.train_samples)
    model = build_model(tf, a.model, a.learning_rate)

    if a.phase == "train":
        data = training_dataset(tf, x_train, y_train, a.batch_size, a.threads)

        # The first fit() traces and compiles the training step, which is a
        # one-time cost PyTorch's eager loop doesn't have. Run two batches to
        # get that out of the way, then put the seeded weights back so the
        # timed run still starts from scratch.
        start_weights = model.get_weights()
        t0 = time.perf_counter()
        model.fit(data.take(2), epochs=1, shuffle=False, verbose=0)
        warmup = time.perf_counter() - t0
        model.set_weights(start_weights)
        model.optimizer.iterations.assign(0)

        with stopwatch() as timer:
            model.fit(data, epochs=a.epochs, shuffle=False, verbose=0)
        model.save_weights(a.weights_file)

        predictions = predict(model, x_test, a.batch_size)
        samples = a.train_samples * a.epochs
        steps = math.ceil(a.train_samples / a.batch_size) * a.epochs
    else:
        model.load_weights(a.weights_file)

        t0 = time.perf_counter()
        model(x_test[:a.batch_size], training=False).numpy()
        warmup = time.perf_counter() - t0

        with stopwatch() as timer:
            predictions = predict(model, x_test, a.batch_size)
        samples = len(x_test)
        steps = math.ceil(len(x_test) / a.batch_size)

    elapsed = timer.seconds
    accuracy = float(np.mean(predictions == y_test))
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "framework": "tensorflow",
        "framework_version": tf.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "study": a.study_name,
        "model": a.model,
        "parameters": model.count_params(),
        "threads": a.threads,
        "batch_size": a.batch_size,
        "epochs": a.epochs,
        "learning_rate": a.learning_rate,
        "trial": a.trial,
        "phase": a.phase,
        "elapsed_seconds": round(elapsed, 6),
        "warmup_seconds": round(warmup, 6),
        "seconds_per_epoch": round(elapsed / a.epochs, 6) if a.phase == "train" else "",
        "samples": samples,    # images processed in the timed part
        "steps": steps,        # optimizer updates (train) or forward batches (inference)
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
    """Logical CPUs we're allowed to run on (respects taskset/cgroup limits)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def plan_runs(a, cpus):
    """List of (study, settings). Only the studied setting changes from the baseline."""
    base = {"threads": a.threads, "batch_size": a.batch_size, "model": "baseline"}
    studies = STUDIES if a.study == "all" else [a.study]
    plan = []
    for study in studies:
        if study == "baseline":
            changes = [{}]
        elif study == "threads":
            # never ask for more threads than the machine has
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
    """Run this script (or `script`) as a worker and return the row it writes.

    `wrapper` is put in front of the python command, e.g. a profiler.
    """
    script = Path(script or __file__).resolve()
    with tempfile.TemporaryDirectory(prefix="tf_run_") as temp:
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
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def write_system_info(out, cpus, extra=""):
    """Machine, OS and library details for the report (proposal 1.2)."""
    lines = [
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Machine: {platform.machine()}",
        f"Logical CPUs usable by this process: {cpus} (os.cpu_count() = {os.cpu_count()})",
        "TF_ENABLE_ONEDNN_OPTS: "
        + os.environ.get("TF_ENABLE_ONEDNN_OPTS", "unset (TensorFlow default)"),
    ]

    # ask a separate python for the versions so TensorFlow isn't loaded here
    versions = subprocess.run(
        [sys.executable, "-c",
         "import numpy, tensorflow as tf; "
         "print('tensorflow', tf.__version__); "
         "print('keras', getattr(tf.keras, '__version__', 'unknown')); "
         "print('numpy', numpy.__version__)"],
        capture_output=True, text=True,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "CUDA_VISIBLE_DEVICES": "-1"},
    )
    lines += ["", "== library versions ==", (versions.stdout or versions.stderr).strip()]

    # frequency scaling / turbo settings affect timing, so keep them too
    for path in ("/etc/os-release",
                 "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                 "/sys/devices/system/cpu/intel_pstate/no_turbo",
                 "/sys/devices/system/cpu/cpufreq/boost"):
        text = read_text(path)
        if text:
            lines += ["", f"== {path} ==", text]

    for cmd in (["uname", "-a"], ["lscpu"], ["free", "-h"]):
        if shutil.which(cmd[0]):
            result = subprocess.run(cmd, capture_output=True, text=True)
            lines += ["", "$ " + " ".join(cmd), (result.stdout or result.stderr).strip()]

    if extra:
        lines += ["", extra]
    (out / "system_info.txt").write_text("\n".join(lines) + "\n")


def save_runs(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path, rows, numeric_fields):
    """Mean, std, median, min and max of every condition over its trials."""
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
                    continue    # e.g. seconds_per_epoch for inference rows
                entry[f"{field}_mean"] = round(mean(values), 6)
                entry[f"{field}_std"] = round(stdev(values), 6) if len(values) > 1 else ""
                entry[f"{field}_median"] = round(median(values), 6)
                entry[f"{field}_min"] = round(min(values), 6)
                entry[f"{field}_max"] = round(max(values), 6)
            writer.writerow(entry)


def run_study(a, run_once=run_in_new_process, fields=FIELDS, numeric_fields=NUMERIC_FIELDS):
    out = Path(a.output).resolve()
    (out / "weights").mkdir(parents=True, exist_ok=True)
    cpus = usable_cpus()
    if a.threads > cpus:
        sys.exit(f"--threads {a.threads} is more than the {cpus} CPUs available")
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
        # Flip the order every other trial so slow drift (the CPU heating up,
        # other people's jobs on the server) doesn't always land on the same
        # settings.
        for study, s in (plan if trial % 2 else plan[::-1]):
            name = f"{study}_{s['model']}_t{s['threads']}_b{s['batch_size']}_r{trial}"
            weights = out / "weights" / f"{name}.weights.h5"
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
                    "--phase", phase,
                    "--study-name", study,
                    "--trial", str(trial),
                    "--weights-file", str(weights),
                ], tag)
                row["run_order"] = order
                row["load_avg_1min"] = round(load, 2)
                rows.append(row)
                # rewrite both files after every run so a crash halfway
                # through doesn't lose what's already been measured
                save_runs(out / "tensorflow_runs.csv", rows, fields)
                save_summary(out / "tensorflow_summary.csv", rows, numeric_fields)

    print("\nDone. Results are in", out)
    print("  tensorflow_runs.csv     one row per run and phase")
    print("  tensorflow_summary.csv  mean/std/median/min/max per condition")
    print("  system_info.txt         machine, OS and library versions")
    return rows


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_study(args)
