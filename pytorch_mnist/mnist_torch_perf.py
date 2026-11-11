"""Runs the mnist_torch.py studies under `perf stat` to collect CPU hardware counters.
Counters are switched on only while a run is being timed, not during start-up or warm-up."""

import argparse
import os
import select
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mnist_torch as bench

# perf events recorded for every run.
EVENTS = (
    "task-clock", "cycles", "instructions", "cache-references",
    "cache-misses", "branches", "branch-misses", "context-switches",
    "cpu-migrations", "page-faults",
)
# Events also reported per image, so runs that process different amounts of data compare fairly.
PER_SAMPLE_EVENTS = ("instructions", "cycles", "cache-misses", "branch-misses")
DERIVED = (
    "ipc", "cache_miss_rate", "branch_miss_rate", "cpu_utilization", "effective_ghz",
    *[e.replace("-", "_") + "_per_sample" for e in PER_SAMPLE_EVENTS],
)
# The 1.3 result columns plus the counter columns, and the ones averaged in the summary.
PERF_FIELDS = [
    *bench.FIELDS, *EVENTS, *DERIVED,
    "min_counter_running_percent", "hybrid_core_share", "perf_scope", "perf_problem",
]
PERF_NUMERIC = [*bench.NUMERIC_FIELDS, *EVENTS, *DERIVED, "min_counter_running_percent"]


class PerfStopwatch(bench.Stopwatch):
    """Stopwatch that also turns the perf counters on when the block starts and off when
    it ends, by writing to perf's control pipe."""

    def __init__(self, control, ack):
        self.control = control
        self.ack = ack

    def send(self, word):
        """Sends "enable" or "disable" to perf and waits up to 15 s for its reply."""
        os.write(self.ctl_fd, (word + "\n").encode())
        ready, _, _ = select.select([self.ack_fd], [], [], 15)
        if not ready or not os.read(self.ack_fd, 100).startswith(b"ack"):
            raise RuntimeError(f"perf didn't acknowledge '{word}'")

    def __enter__(self):
        self.ctl_fd = os.open(self.control, os.O_RDWR)
        self.ack_fd = os.open(self.ack, os.O_RDWR)
        self.send("enable")
        return super().__enter__()

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self.send("disable")
        os.close(self.ctl_fd)
        os.close(self.ack_fd)


@contextmanager
def perf_fifos():
    """Creates the control and reply pipes that let a run switch perf on and off.
    Both are held open here so neither perf nor the run blocks when opening them."""
    with tempfile.TemporaryDirectory(prefix="torch_perf_") as temp:
        control, ack = Path(temp) / "control.fifo", Path(temp) / "ack.fifo"
        os.mkfifo(control)
        os.mkfifo(ack)
        fds = [os.open(control, os.O_RDWR | os.O_NONBLOCK),
               os.open(ack, os.O_RDWR | os.O_NONBLOCK)]
        try:
            yield control, ack
        finally:
            for fd in fds:
                os.close(fd)


def perf_command(control, ack, perf_file):
    """Builds the `perf stat` command that wraps a run; --delay=-1 starts with the
    counters off until the run enables them."""
    return ["perf", "stat", "--delay=-1", f"--control=fifo:{control},{ack}",
            "-x", ";", "--no-big-num", "-e", ",".join(EVENTS),
            "-o", str(perf_file), "--"]


def split_event(field):
    """Splits perf's event label, e.g. 'cpu_core/cycles/' into ('cycles', 'cpu_core')."""
    name, pmu = field.strip(), ""
    if "/" in name:
        pmu, name = name.split("/")[:2]
    return name.split(":")[0], pmu


def read_perf(path):
    """Reads perf's output file into {event: count}, skipping unsupported events.
    On hybrid Intel CPUs the per-core-type lines are unscaled and added together."""
    if not path.exists():
        return {}, "perf produced no output"
    found = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split(";")
        if len(parts) < 3:
            continue
        event, pmu = split_event(parts[2])
        if event not in EVENTS:
            continue
        try:
            value = float(parts[0].strip().replace(",", ""))
        except ValueError:          # "<not supported>" or "<not counted>"
            continue
        try:
            running = float(parts[4].strip().rstrip("%"))
        except (IndexError, ValueError):
            running = 100.0
        found.setdefault(event, []).append((pmu, value, running))

    values, running_pcts = {}, []
    for event, entries in found.items():
        if any(pmu for pmu, _, _ in entries):
            values[event] = round(sum(v * pct / 100 for _, v, pct in entries))
        else:
            values[event] = sum(v for _, v, _ in entries)
            running_pcts += [pct for _, _, pct in entries]

    # Lowest share of time a counter actually ran; below 100 means perf scaled (estimated) it.
    if running_pcts:
        values["min_counter_running_percent"] = round(min(running_pcts), 2)
    if any(pmu for pmu, _, _ in found.get("cycles", [])):
        values["hybrid_core_share"] = ", ".join(
            f"{pmu} {pct:.1f}%" for pmu, _, pct in found["cycles"])
    return values, "" if values else "no supported perf events returned"


def derived_metrics(row):
    """Computes IPC, miss rates, CPU utilization, average clock speed and per-image
    counts from the raw counters in a result row."""
    def num(key):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return None

    def ratio(top, bottom, digits):
        return round(top / bottom, digits) if top is not None and bottom else ""

    elapsed, samples = num("elapsed_seconds"), num("samples")
    task_ms, cycles = num("task-clock"), num("cycles")
    result = {
        "ipc": ratio(num("instructions"), cycles, 4),
        "cache_miss_rate": ratio(num("cache-misses"), num("cache-references"), 6),
        "branch_miss_rate": ratio(num("branch-misses"), num("branches"), 6),
        # CPU time divided by wall time = average number of busy CPUs.
        "cpu_utilization": ratio(None if task_ms is None else task_ms / 1000, elapsed, 3),
        # Cycles divided by CPU time = average clock speed while running.
        "effective_ghz": ratio(cycles, None if task_ms is None else task_ms * 1e6, 3),
    }
    for event in PER_SAMPLE_EVENTS:
        result[event.replace("-", "_") + "_per_sample"] = ratio(num(event), samples, 1)
    return result


def perf_runner(out):
    """Returns a run function for bench.run_study that starts each run under perf,
    then adds the counters and derived metrics to its result row."""
    def run_once(worker_args, tag):
        perf_file = out / "perf_raw" / f"{tag}.txt"
        with perf_fifos() as (control, ack):
            row = bench.run_in_new_process(
                [*worker_args, "--control", str(control), "--ack", str(ack)], tag,
                script=__file__, wrapper=perf_command(control, ack, perf_file))
        values, problem = read_perf(perf_file)
        row.update(values)
        row.update(derived_metrics(row))
        row["perf_scope"] = "measured phase only"
        row["perf_problem"] = problem
        if problem:
            print("perf warning:", problem)
        return row
    return run_once


def perf_probe(a):
    """Small test workload for --check that uses the same on/off switching as real runs."""
    with PerfStopwatch(a.control, a.ack):
        total = sum(i * i for i in range(2_000_000))
    print("probe finished", total > 0)


def check_perf(out):
    """Measures the test workload under perf and reports which events are available."""
    if not shutil.which("perf"):
        return False, "perf not found (install linux-tools for your kernel)"
    report = [
        "perf_event_paranoid: "
        + (bench.read_text("/proc/sys/kernel/perf_event_paranoid") or "unknown"),
        subprocess.run(["perf", "--version"], capture_output=True, text=True).stdout.strip(),
    ]
    probe_file = out / "perf_raw" / "_perf_check.txt"
    with perf_fifos() as (control, ack):
        result = subprocess.run(
            [*perf_command(control, ack, probe_file), sys.executable, __file__,
             "--probe", "--control", str(control), "--ack", str(ack)],
            capture_output=True, text=True)
    if result.returncode:
        report += ["perf couldn't measure a test process:",
                   (result.stderr or result.stdout).strip()]
        return False, "\n".join(report)

    values, _ = read_perf(probe_file)
    have = [e for e in EVENTS if e in values]
    missing = [e for e in EVENTS if e not in values]
    report.append("events available: " + (", ".join(have) or "none"))
    if missing:
        report.append("events unavailable (left blank in results): " + ", ".join(missing))
    if "hybrid_core_share" in values:
        report.append("hybrid P-core/E-core CPU: per-core-type counts are added together")
    return bool(have), "\n".join(report)


def main():
    """Adds the perf options to the 1.3 parser, then runs a probe, one worker run,
    or the full study depending on the options given."""
    p = bench.build_parser()
    p.add_argument("--check", action="store_true",
                   help="only check that perf works, then exit")
    p.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--control", default="", help=argparse.SUPPRESS)
    p.add_argument("--ack", default="", help=argparse.SUPPRESS)
    a = bench.parse_args(p)

    if a.probe:
        perf_probe(a)
        return
    if a.worker:
        bench.run_worker(a, stopwatch=lambda: PerfStopwatch(a.control, a.ack))
        return

    # Check perf first and stop before the long study if it can't count anything.
    out = Path(a.output).resolve()
    (out / "perf_raw").mkdir(parents=True, exist_ok=True)
    ok, report = check_perf(out)
    (out / "perf_check.txt").write_text(report + "\n")
    print(report)
    if a.check:
        return
    if not ok:
        sys.exit("\nperf isn't usable here, so nothing was run. Ask the admin about "
                 "perf_event_paranoid, or run mnist_torch.py for timing only.")
    bench.run_study(a, run_once=perf_runner(out),
                    fields=PERF_FIELDS, numeric_fields=PERF_NUMERIC)


if __name__ == "__main__":
    main()
