"""MNIST training/inference benchmark in TensorFlow (CS3339, Team 6).

Runs the experiments from section 1.3 of our proposal: a baseline run plus
thread, batch size and model sweeps, all on the CPU.
"""

import argparse
import os

# Hidden layer widths. Every model takes 784 inputs and ends in 10 logits.
MODELS = {
    "baseline": (128,),     # 784-128-10
    "wide": (256,),         # 784-256-10
    "deep": (128, 64),      # 784-128-64-10
}
THREAD_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
BATCH_SWEEP = (32, 64, 128, 256)
STUDIES = ("baseline", "threads", "batch", "models")


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
