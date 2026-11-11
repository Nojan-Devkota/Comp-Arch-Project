"""MNIST training/inference benchmark in TensorFlow (CS3339, Team 6).

Runs the experiments from section 1.3 of our proposal: a baseline run plus
thread, batch size and model sweeps, all on the CPU.
"""

import argparse
import csv
import math
import os
import platform
import time

# Hidden layer widths. Every model takes 784 inputs and ends in 10 logits.
MODELS = {
    "baseline": (128,),     # 784-128-10
    "wide": (256,),         # 784-256-10
    "deep": (128, 64),      # 784-128-64-10
}
THREAD_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
BATCH_SWEEP = (32, 64, 128, 256)
STUDIES = ("baseline", "threads", "batch", "models")

FIELDS = [
    "framework", "framework_version", "python_version", "machine",
    "study", "model", "parameters", "threads", "batch_size", "epochs",
    "learning_rate", "trial", "run_order", "phase",
    "elapsed_seconds", "warmup_seconds", "seconds_per_epoch",
    "samples", "steps", "images_per_second", "test_accuracy", "load_avg_1min",
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
    tf = setup_tensorflow(a.threads, seed=1000 + a.trial)
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


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args)
