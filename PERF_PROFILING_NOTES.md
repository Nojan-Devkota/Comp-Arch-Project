# TensorFlow MNIST CPU Hardware Profiling

This document explains the CPU performance measurements collected by the project's `mnist_tf_perf.py` wrapper around `mnist_tf.py`.

## Purpose

Wall-clock time measures overall speed, while Linux `perf stat` counters can help explain *why* configurations differ. The profiler enables counters during the measured workload rather than during startup or warm-up.

## Recorded counters

- `task-clock`: CPU time (milliseconds)
- `cycles`: processor cycles
- `instructions`: retired instructions
- `cache-references` and `cache-misses`: CPU cache activity
- `branches` and `branch-misses`: branch prediction activity
- `context-switches`, `cpu-migrations`, and `page-faults`: OS/runtime behavior

## Derived performance metrics

| Metric | Formula | Meaning |
| --- | --- | --- |
| IPC | instructions / cycles | Instructions retired per CPU cycle |
| CPI | cycles / instructions | Cycles per retired instruction |
| Cache miss rate | cache-misses / cache-references | Fraction of counted cache references that miss |
| Branch miss rate | branch-misses / branches | Fraction of counted branches mispredicted |
| Instructions per sample | instructions / samples | Instruction work per processed image |
| Cycles per sample | cycles / samples | CPU cycles per processed image |
| Average busy CPUs | task-clock (seconds) / elapsed_seconds | Average CPU parallelism; may exceed 1 |
| Effective GHz | cycles / (task-clock in milliseconds × 1,000,000) | Approximate average clock frequency |

The wrapper currently reports IPC but not CPI. CPI can be calculated later from the raw counters.

## How to check profiling support

From the repository root on a Linux machine with `perf` installed:

```bash
python tensorflow_mnist/mnist_tf_perf.py --check
```

The script writes `perf_check.txt` to its configured output directory. Check the script's help for supported study and trial options:

```bash
python tensorflow_mnist/mnist_tf_perf.py --help
```

A full profiling run uses the same experiment arguments accepted by the underlying MNIST benchmark. Hardware counter access may be disabled by Linux `perf_event_paranoid` settings.

## Fair experiment design

1. Record the CPU model, operating system, Python version, TensorFlow version, and thread configuration.
2. Keep the dataset, preprocessing, model, optimizer, and random seed consistent when comparing configurations.
3. Exclude initialization and warm-up from the timed interval.
4. Repeat each configuration and report both central tendency and variation.
5. Compare elapsed time alongside IPC, cache/branch miss rates, and instructions/cycles *per image*.
6. Clearly label unavailable hardware counters rather than treating missing measurements as zero.
7. When comparing frameworks, verify equivalent model architecture and workloads.

## Interpretation and limitations

- More IPC does **not** necessarily mean faster end-to-end execution; total instructions, parallelism, and elapsed time also matter.
- Cache events are hardware/PMU dependent, so their meaning can differ across machines.
- Linux `perf` may multiplex counters. Review the running percentages before interpreting scaled readings.
- Hybrid Intel P-core/E-core systems can report separate hardware events. Aggregation should be validated on the target machine.
- The profiler's parsing and formula implementation should be tested against actual `perf stat -x ';'` output before treating the results as validated.
- Profiling overhead, background processes, thermal behavior, and CPU frequency scaling can affect measured values.

## Suggested plots for the report

- Elapsed time vs. CPU thread count
- Elapsed time vs. batch size
- IPC and cycles per sample across model configurations
- Cache and branch miss rates vs. runtime

Use the results to make evidence-based observations about CPU utilization, cache behavior, and instruction execution instead of assuming a cause from timing alone.
