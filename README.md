# Comp-Arch-Project

CS3339 Fall 2026, Team 6: performance comparison and hardware profiling of
PyTorch and TensorFlow on the CPU, using MNIST.

## TensorFlow benchmark

`tensorflow_mnist/mnist_tf.py` runs the experiments from section 1.3 of the
proposal. All settings default to the baseline: 5 epochs, batch size 64,
SGD with learning rate 0.01, float32, 1 intra-op thread, 784-128-10 model,
seed 42 for every run and no shuffling (same as the PyTorch side).

| Study      | What changes                                  |
|------------|-----------------------------------------------|
| `baseline` | nothing, baseline settings                    |
| `threads`  | 1, 2, 4, 8, 16, 32, 64, 128 threads           |
| `batch`    | batch size 32, 64, 128, 256                   |
| `models`   | 784-128-10, 784-256-10 (wide), 784-128-64-10 (deep) |

Thread counts higher than the CPUs available are skipped.

```bash
pip install tensorflow-cpu

# download MNIST once so it isn't part of any measured run
python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"

python tensorflow_mnist/mnist_tf.py --study baseline --trials 1   # pilot
python tensorflow_mnist/mnist_tf.py --study all --trials 3        # full run
```

Each measurement (one condition, one trial, train or inference) runs in its
own Python process. Warm-up and data loading are not timed. Results go to
`tf_results/` (change with `--output`):

- `tensorflow_runs.csv`: one row per run
- `tensorflow_summary.csv`: mean, std, median, min, max per condition
- `system_info.txt`: CPU, OS, kernel and library versions

`--thread-list`, `--batch-list` and `--model-list` override the sweep
values, and `--train-samples` / `--epochs` can shrink runs for quick tests.

## PyTorch benchmark

`pytorch_mnist/mnist_torch.py` runs the same four studies with the same
settings, options and CSV columns as the TensorFlow script, so the two result
files can be merged directly. To keep the comparison fair it also matches:

- starting weights: Glorot/Xavier uniform weights and zero biases (the Keras default)
- seed 42, no shuffling, batches taken in order as slices of the data
- the same warm-up (2 untimed training batches, 1 untimed inference batch)
- the same inference loop (forward pass, `.numpy()` and `argmax` per batch)
- 1 inter-op thread, with OMP/MKL threads set before PyTorch is imported

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

python pytorch_mnist/mnist_torch.py --study baseline --trials 1   # pilot
python pytorch_mnist/mnist_torch.py --study all --trials 3        # full run
```

MNIST is downloaded once to `~/mnist-project/data` (change with
`--data-dir`) before any run starts. Results go to `torch_results/`:
`pytorch_runs.csv`, `pytorch_summary.csv` and `system_info.txt`.
