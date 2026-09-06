# CrashSurgery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make excision (`CrashSurgery`) the pipeline's crash handling, with per-epoch DTW in Julia, while keeping interpolation (`CrashRepair`) reachable behind `--crash_mode interp`.

**Architecture:** Stage 1 (`reproc_cpCST.py`) annotates crashes on the original clock instead of interpolating and resampling. Stage 2 (`compute_irt_parallel.jl`) aligns each contiguous `is_valid` run separately, masking 3 samples at each run edge. Interp mode is the old path untouched, filename-tagged `_interp`.

**Tech Stack:** Python 3 (pandas, numpy, scipy, matplotlib, pytest), Julia (CSV, DataFrames, DynamicAxisWarping, Test).

**Spec:** `docs/superpowers/specs/2026-09-06-crash-surgery-design.md`

## Global Constraints

- Never renumber, resample or drop `flip_time` in surgery mode.
- `CrashRepair.py` is not modified. Interp-mode output must equal the golden file byte-for-byte after `read_csv`.
- Sampling rate is 30 Hz; derive intervals from `flip_time`.
- No `Pkg.add()` in Julia scripts; `Test` is a stdlib and needs no Project change.
- Tests run with `python3 -m pytest tests -q` and `julia --threads=auto tests/test_irt.jl` from the repo root.
- Commit after every task with the trailer:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_011yjju7ZCvr32NUX68yNUKZ`.

## File map

- Create `tests/conftest.py` — puts repo root on `sys.path`, forces Agg backend.
- Create `tests/synth.py` — synthetic recordings from the plant identity.
- Create `tests/data/synth_crash.csv`, `tests/data/golden_interp.csv` — fixed input and interp-mode golden output.
- Create `tests/test_crash_surgery.py`, `tests/test_reproc.py`, `tests/test_irt.jl`.
- Modify `CrashSurgery.py` — remove `align_epochs`, add `plot_excision`.
- Modify `reproc_cpCST.py` — `--crash_mode`, gap-aware velocity, masked detrend/z-score, event table, excision plot, `_interp` tag.
- Modify `compute_irt_parallel.jl` — per-run alignment, `EDGE_MASK`, `n_epochs_aligned`.
- Modify `CLAUDE.md`, `README.md`, `.gitignore`.

---

### Task 1: Synthetic recordings and the interp golden file

Do this at the current HEAD, before any pipeline change, so the golden captures today's interp behaviour.

**Files:**
- Create: `tests/conftest.py`, `tests/synth.py`, `tests/data/synth_crash.csv`, `tests/data/golden_interp.csv`

**Interfaces:**
- Produces: `synth.make_recording(duration_s=60.0, crash_times=(), seed=0, t0=3.5) -> pd.DataFrame` with raw-file columns `flip_time, stim_pos, user_pos, crash_count, did_crash, expected_time, lambda_val`, `user_pos` in the raw (unflipped) sign, ending with the corpus's duplicated last `flip_time`. Constants `FS, DT, LAMBDA, GAIN, RESET_GAP_S, BOUNDARY`.

- [ ] **Step 1: conftest**

```python
# tests/conftest.py
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: generator**

```python
# tests/synth.py
"""Synthetic cpCST recordings built from the plant identity.

d(stim)/dt = GAIN * LAMBDA * e,  e = stim + user_raw
A crash is ~1.9 s of error sharing the stimulus's sign (runaway to BOUNDARY),
then RESET_GAP_S with no rows logged, then a reset to stim = 0 and 0.5 s of
one-signed error (settle). Matches the corpus's crash shape.
"""
import numpy as np
import pandas as pd

FS = 30.0
DT = 1.0 / FS
LAMBDA = 0.15
GAIN = 3.0
RESET_GAP_S = 2.583
BOUNDARY = 0.5


def make_recording(duration_s=60.0, crash_times=(), seed=0, t0=3.5):
    rng = np.random.default_rng(seed)
    n_target = int(round(duration_s * FS))
    pending = sorted(crash_times)
    rec = []
    state = {"t": t0, "stim": 0.0, "cc": 0, "did_crash": False}

    def push(e):
        rec.append((state["t"], state["stim"], e - state["stim"],
                    state["cc"], state["did_crash"]))
        state["did_crash"] = False
        state["stim"] += DT * GAIN * LAMBDA * e
        state["t"] += DT

    settle_left = 0
    while len(rec) < n_target:
        if pending and state["t"] - t0 >= pending[0]:
            pending.pop(0)
            sgn = 1.0 if state["stim"] >= 0 else -1.0
            while abs(state["stim"]) < BOUNDARY:
                push(sgn * 0.6)
            state["t"] += RESET_GAP_S - DT
            state["cc"] += 1
            state["stim"] = 0.0
            state["did_crash"] = True
            settle_left = 15
            continue
        if settle_left:
            settle_left -= 1
            push(0.05)
        else:
            push(rng.normal(0.0, 0.01) - 0.3 * state["stim"])

    df = pd.DataFrame(rec, columns=["flip_time", "stim_pos", "user_pos",
                                    "crash_count", "did_crash"])
    df["expected_time"] = df["flip_time"] - 0.015
    df["lambda_val"] = LAMBDA
    # every file in the corpus ends with a duplicated flip_time
    return pd.concat([df, df.iloc[[-1]]], ignore_index=True)


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    make_recording(60.0, crash_times=(20.0, 40.0)).to_csv(
        out / "synth_crash.csv", index=False)
```

- [ ] **Step 3: write the fixed input and the golden**

```bash
python3 tests/synth.py
python3 - <<'EOF'
import sys; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import os; os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import reproc_cpCST as r
Path("tests/data/_golden").mkdir(parents=True, exist_ok=True)
r.process_file(Path("tests/data/synth_crash.csv"), Path("tests/data/_golden"), False, False)
os.replace("tests/data/_golden/synth_crash.csv", "tests/data/golden_interp.csv")
import shutil; shutil.rmtree("tests/data/_golden")
EOF
rm -f tests/data/*_repaired.png
head -c 300 tests/data/golden_interp.csv
```
Expected: `golden_interp.csv` has columns `flip_time,stim_pos,user_pos,crash_count,expected_time,lambda_val,was_repaired,did_crash,tracking,...`. Note `process_file` appends a line to `./crash_count.csv`; delete that line or the file (`git checkout crash_count.csv` is not possible, it is untracked; just leave it).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/synth.py tests/data/synth_crash.csv tests/data/golden_interp.csv
git commit -m "Add synthetic recordings and the interp-mode golden output"
```

---

### Task 2: CrashSurgery — tests, remove `align_epochs`, add `plot_excision`

**Files:**
- Modify: `CrashSurgery.py` (remove `align_epochs` lines 176-199; add `plot_excision` after `epoch_segments`)
- Test: `tests/test_crash_surgery.py`

**Interfaces:**
- Consumes: `synth.make_recording`.
- Produces: `CrashSurgery.plot_excision(annotated, events, zoom_s=6.0) -> matplotlib.figure.Figure`.

- [ ] **Step 1: failing tests**

```python
# tests/test_crash_surgery.py
import numpy as np
import pandas as pd
import pytest

import CrashSurgery as cs
from synth import make_recording, RESET_GAP_S


def load(crash_times=(20.0,)):
    df = make_recording(60.0, crash_times=crash_times)
    df["user_pos"] = df["user_pos"] * -1        # pipeline sign flip
    return df


def test_plant_fit_is_exact_on_synthetic():
    ann, _ = cs.prepare(load())
    assert cs.check_plant_fit(ann) > 0.999


def test_detects_one_event_with_sensible_geometry():
    df = load()
    ann, ev = cs.prepare(df)
    assert len(ev) == 1
    e = ev[0]
    t = ann["flip_time"].values
    assert e["reset"] == int(np.argmax(ann["crash_count"].values == 1))
    assert e["onset"] < e["reset"] < e["settle"]
    assert 1.5 < e["runaway_s"] < 2.5
    assert abs(e["reset_gap_s"] - RESET_GAP_S) < 1e-6
    assert 0.4 <= e["settle_s"] <= 2.0
    assert e["onset_time"] == t[e["onset"]]


def test_annotation_columns():
    ann, ev = cs.prepare(load())
    e = ev[0]
    valid = ann["is_valid"].values
    assert not valid[e["onset"]:e["settle"] + 1].any()
    assert valid[:e["onset"]].all() and valid[e["settle"] + 1:].all()
    phase = ann["crash_phase"].values
    assert (phase[e["onset"]:e["reset"]] == cs.PHASE_RUNAWAY).all()
    assert (phase[e["reset"]:e["settle"] + 1] == cs.PHASE_REACQUIRE).all()
    assert (ann["epoch"].values[:e["reset"]] == 0).all()
    assert (ann["epoch"].values[e["reset"]:] == 1).all()
    t = ann["flip_time"].values
    assert np.isnan(ann["time_since_crash"].values[:e["reset"]]).all()
    assert ann["time_since_crash"].values[e["reset"]] == 0.0
    assert np.isnan(ann["time_to_crash"].values[e["reset"]:]).all()
    assert np.isclose(ann["time_to_crash"].values[0], t[e["reset"]] - t[0])


def test_flip_time_is_untouched():
    df = load()
    ann, _ = cs.prepare(df)
    raw = df["flip_time"].values[:-1]           # prepare drops the duplicate
    assert len(ann) == len(raw)
    assert np.array_equal(ann["flip_time"].values, raw)


def test_crash_free_file_is_all_valid():
    ann, ev = cs.prepare(load(crash_times=()))
    assert ev == []
    assert ann["is_valid"].all()
    assert ann["time_since_crash"].isna().all()


def test_two_crashes_give_three_epochs():
    ann, ev = cs.prepare(load(crash_times=(20.0, 40.0)))
    assert len(ev) == 2
    segs = list(cs.epoch_segments(ann))
    assert [k for k, _ in segs] == [0, 1, 2]


def test_align_epochs_removed():
    assert not hasattr(cs, "align_epochs")


def test_plot_excision_returns_figure(tmp_path):
    ann, ev = cs.prepare(load(crash_times=(20.0, 40.0)))
    fig = cs.plot_excision(ann, ev)
    assert len(fig.axes) == 3                    # overview + one per crash
    fig.savefig(tmp_path / "x.png")
    assert (tmp_path / "x.png").stat().st_size > 0
```

- [ ] **Step 2: run, expect the last two to fail**

Run: `python3 -m pytest tests/test_crash_surgery.py -q`
Expected: `test_align_epochs_removed` and `test_plot_excision_returns_figure` FAIL; the others PASS (they exercise the existing prototype). If any of the others fail, the prototype has a bug — investigate before continuing, do not loosen the test.

- [ ] **Step 3: edit `CrashSurgery.py`**

Delete `align_epochs` (the whole function). In its place add:

```python
def plot_excision(annotated, events, zoom_s=6.0):
    """Overview of the recording with every excised span shaded, plus one
    zoomed panel per crash marking onset, reset and settle."""
    import matplotlib.pyplot as plt

    t = annotated["flip_time"].values.astype(float)
    s = annotated["stim_pos"].values.astype(float)
    u = annotated["user_pos"].values.astype(float)
    n = len(events)
    fig, axes = plt.subplots(1 + n, 1, figsize=(12, 3 * (1 + n)), squeeze=False)
    axes = axes[:, 0]

    ax = axes[0]
    ax.plot(t, s, "r-", lw=0.6, label="stim_pos")
    ax.plot(t, u, "b-", lw=0.6, label="user_pos (flipped)")
    for ev in events:
        ax.axvspan(ev["onset_time"], ev["settle_time"], color="gray", alpha=0.3)
    ax.set_ylabel("position")
    ax.set_title(f"{n} crash(es); shaded spans are excised (is_valid == False)")
    ax.legend(loc="upper right")

    marks = (("onset_time", "k", "onset"), ("reset_time", "m", "reset"),
             ("settle_time", "g", "settle"))
    for ax, ev in zip(axes[1:], events):
        m = (t >= ev["onset_time"] - zoom_s) & (t <= ev["settle_time"] + zoom_s)
        ax.plot(t[m], s[m], "r.-", ms=2, lw=0.6, label="stim_pos")
        ax.plot(t[m], u[m], "b.-", ms=2, lw=0.6, label="user_pos (flipped)")
        ax.axvspan(ev["onset_time"], ev["settle_time"], color="gray", alpha=0.3)
        for key, color, label in marks:
            ax.axvline(ev[key], color=color, ls=":", label=label)
        ax.set_ylabel("position")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("flip_time (s)")
    fig.tight_layout()
    return fig
```

Move the edge-mask paragraph from the deleted docstring into a comment at the top of `epoch_segments` noting that alignment now lives in `compute_irt_parallel.jl` (`EDGE_MASK = 3`).

- [ ] **Step 4: run, expect all pass**

Run: `python3 -m pytest tests/test_crash_surgery.py -q` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add CrashSurgery.py tests/test_crash_surgery.py
git commit -m "CrashSurgery: cover detection with tests, add plot_excision, drop align_epochs"
```

---

### Task 3: `reproc_cpCST.py` — `--crash_mode`, gap-aware velocity, masked scaling, events, plot

**Files:**
- Modify: `reproc_cpCST.py`
- Test: `tests/test_reproc.py`

**Interfaces:**
- Consumes: `CrashSurgery.prepare`, `CrashSurgery.plot_excision`.
- Produces: `process_file(file_path, output_path, detrend_vectors, zscale_vectors, max_seconds=None, max_samples=None, crash_mode="surgery")`; `record_events(events, file_name, ursi, output_path)`; `detrend_fit(series, mask)`; `zscale_fit(series, mask)`; `compute_velocity(df, target_col, fs=30.0)`; constants `EVENTS_FILE = "crash_events.csv"`, `EVENT_COLS`.

- [ ] **Step 1: failing tests**

```python
# tests/test_reproc.py
import numpy as np
import pandas as pd
import pytest

import reproc_cpCST as r
from synth import make_recording, FS

DATA = pytest.importorskip("pathlib").Path(__file__).parent / "data"
ANNOT = ["crash_phase", "is_valid", "epoch", "time_since_crash", "time_to_crash"]


def run(tmp_path, mode, **kw):
    out = tmp_path / mode
    out.mkdir()
    r.process_file(DATA / "synth_crash.csv", out, kw.pop("detrend", False),
                   kw.pop("zscale", False), crash_mode=mode, **kw)
    return out


def test_interp_matches_golden(tmp_path):
    out = run(tmp_path, "interp")
    got = pd.read_csv(out / "synth_crash_interp.csv")
    want = pd.read_csv(DATA / "golden_interp.csv")
    pd.testing.assert_frame_equal(got, want)
    assert (out / "synth_crash_interp_repaired.png").exists()


def test_surgery_keeps_rows_and_clock(tmp_path):
    out = run(tmp_path, "surgery")
    got = pd.read_csv(out / "synth_crash.csv")
    raw = pd.read_csv(DATA / "synth_crash.csv").iloc[:-1]     # dup dropped
    assert len(got) == len(raw)
    assert np.array_equal(got["flip_time"].values, raw["flip_time"].values)
    assert np.array_equal(got["user_pos"].values, raw["user_pos"].values)
    for c in ANNOT:
        assert c in got.columns
    assert "was_repaired" not in got.columns
    assert got["is_valid"].dtype == bool
    assert (~got["is_valid"]).sum() > 0


def test_surgery_velocity_nan_across_reset_gap(tmp_path):
    out = run(tmp_path, "surgery")
    got = pd.read_csv(out / "synth_crash.csv")
    gap = np.r_[False, np.diff(got["flip_time"].values) > 2.0 / FS]
    assert gap.sum() == 2
    for c in ("user_pos_vel", "stim_pos_vel", "tracking_vel"):
        assert got[c][gap].isna().all()
        assert got[c][~gap].notna().all()


def test_surgery_zscore_fits_on_valid_only(tmp_path):
    out = run(tmp_path, "surgery", zscale=True)
    got = pd.read_csv(out / "synth_crash.csv")
    v = got["is_valid"].values
    z = got["tracking"].values
    assert abs(z[v].mean()) < 1e-9
    assert abs(z[v].std(ddof=1) - 1.0) < 1e-9
    assert np.abs(z[~v]).max() > 3            # the excursion is off-scale by design


def test_surgery_detrend_fits_on_valid_only(tmp_path):
    out = run(tmp_path, "surgery", detrend=True)
    got = pd.read_csv(out / "synth_crash.csv")
    v = got["is_valid"].values
    y = got["stim_pos"].values
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x[v], y[v], 1)[0]
    assert abs(slope) < 1e-9


def test_surgery_writes_events_and_plot(tmp_path):
    out = run(tmp_path, "surgery")
    ev = pd.read_csv(out / r.EVENTS_FILE)
    assert list(ev.columns) == list(r.EVENT_COLS)
    assert len(ev) == 2 and list(ev["k"]) == [1, 2]
    assert (ev["file"] == "synth_crash.csv").all()
    assert (out / "synth_crash_excised.png").exists()


def test_events_file_is_fresh_per_run(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / r.EVENTS_FILE).write_text("stale\n")
    r.reset_events(out)
    r.process_file(DATA / "synth_crash.csv", out, False, False)
    ev = pd.read_csv(out / r.EVENTS_FILE)
    assert len(ev) == 2


def test_surgery_max_samples_counts_invalid_rows(tmp_path):
    out = run(tmp_path, "surgery", max_samples=1000)
    got = pd.read_csv(out / "synth_crash_trim1000samp.csv")
    assert len(got) == 1000
    assert "is_valid" in got.columns


def test_crash_free_surgery_has_all_valid(tmp_path):
    src = tmp_path / "clean.csv"
    make_recording(40.0).to_csv(src, index=False)
    out = tmp_path / "o"
    out.mkdir()
    r.process_file(src, out, False, False)
    got = pd.read_csv(out / "clean.csv")
    assert got["is_valid"].all()
    assert not (out / "clean_excised.png").exists()
    assert not (out / r.EVENTS_FILE).exists() or len(pd.read_csv(out / r.EVENTS_FILE)) == 0
```

- [ ] **Step 2: run, expect failures**

Run: `python3 -m pytest tests/test_reproc.py -q`
Expected: everything fails on `crash_mode` unexpected keyword or missing attributes.

- [ ] **Step 3: implement**

In `reproc_cpCST.py`:

Imports: add `import CrashSurgery`.

Add after `MIN_SAMPLES`:

```python
EVENTS_FILE = "crash_events.csv"
EVENT_COLS = ("file", "ursi", "k", "onset_time", "reset_time", "settle_time",
              "runaway_s", "reset_gap_s", "settle_s")
CRASH_MODES = ("surgery", "interp")


def reset_events(output_path):
    """Start a fresh event table for this run (it is overwritten, not appended)."""
    p = Path(output_path) / EVENTS_FILE
    if p.exists():
        p.unlink()


def record_events(events, file_name, ursi, output_path):
    p = Path(output_path) / EVENTS_FILE
    write_header = not p.exists()
    with open(p, "a") as f:
        if write_header:
            f.write(",".join(EVENT_COLS) + "\n")
        for k, ev in enumerate(events, start=1):
            f.write(f"{file_name},{ursi},{k},{ev['onset_time']},{ev['reset_time']},"
                    f"{ev['settle_time']},{ev['runaway_s']},{ev['reset_gap_s']},"
                    f"{ev['settle_s']}\n")


def detrend_fit(series, mask):
    """Linear detrend fitted on `mask` rows only, subtracted from every row."""
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    m = mask & np.isfinite(y)
    a, b = np.polyfit(x[m], y[m], 1)
    return pd.Series(y - (a * x + b), index=series.index)


def zscale_fit(series, mask):
    """Z-score with mean and SD taken from `mask` rows only, applied to every row."""
    y = series.values.astype(float)
    m = mask & np.isfinite(y)
    return (series - y[m].mean()) / y[m].std(ddof=1)
```

Replace `compute_velocity`:

```python
def compute_velocity(df, target_col, fs=30.0):
    """First derivative, in position units per second.

    NaN wherever the frame interval exceeds two frames: across a controller
    reset dt is ~2.58 s and the quotient is not a velocity of anything. On the
    interp path the grid is uniform, so this never fires there.
    """
    dt = df["flip_time"].diff()
    ok = (dt > 0) & (dt <= 2.0 / fs)
    vel = df[target_col].diff() / dt.where(ok)
    vel.iloc[0] = 0.0
    df[f"{target_col}_vel"] = vel
```

Extract the interp block of `process_file` into:

```python
def repair_by_interpolation(df, file_path, output_path):
    cr = CrashRepair(df)
    cr.set_target_max_position()
    repaired_df = cr.repair_tracking()
    if df.crash_count.max() > 0:
        fig = cr.plot_repair(repaired_df, segment_index=0)
        if fig is not None:
            fig.savefig(output_path / file_path.name.replace(".csv", "_interp_repaired.png"))
            plt.close()
        else:
            print("No crash report generated")
    return repaired_df


def excise(df, file_path, ursi, output_path):
    annotated, events = CrashSurgery.prepare(df)
    record_events(events, file_path.name, ursi, output_path)
    if events:
        fig = CrashSurgery.plot_excision(annotated, events)
        fig.savefig(output_path / file_path.name.replace(".csv", "_excised.png"))
        plt.close(fig)
    return annotated
```

Rewrite the body of `process_file` from the sign flip onward:

```python
        df.user_pos = df.user_pos * -1
        if crash_mode == "interp":
            df = repair_by_interpolation(df, file_path, output_path)
        elif crash_mode == "surgery":
            df = excise(df, file_path, ursi, output_path)
        else:
            raise ValueError(f"unknown crash_mode {crash_mode!r}; choose from {CRASH_MODES}")
        if max_samples is not None:
            ... (unchanged)
        df["tracking"] = ...  (unchanged through velocities)

        valid = df["is_valid"].values if "is_valid" in df.columns else np.ones(len(df), bool)
        signal_cols = [c for c in SIGNAL_COLS if c in df.columns]
        if detrend_vectors:
            for col in signal_cols:
                df[col] = detrend(df[col]) if crash_mode == "interp" else detrend_fit(df[col], valid)
        if zscale_vectors:
            for col in signal_cols:
                df[col] = zscale(df[col]) if crash_mode == "interp" else zscale_fit(df[col], valid)

        filename = file_path.name.replace(".csv", "")
        if crash_mode == "interp":
            filename += "_interp"
        ... (tags unchanged)
```

`parse_arguments`: add `parser.add_argument("--crash_mode", choices=CRASH_MODES, default="surgery", help="surgery: excise crashes and annotate on the original clock (default). interp: the legacy CrashRepair interpolation, kept for reference; outputs are tagged _interp.")`.

`main`: after `mkdir`, call `reset_events(output_path)`, print the mode, and pass `crash_mode=args.crash_mode` to `process_file`.

Wait: the golden test expects `synth_crash_interp_repaired.png`. The interp plot name is derived from `file_path.name`, and the tag was added to the CSV name only. Either name works as long as the test and code agree; use `_interp_repaired.png` as written above so the PNG is tagged too.

- [ ] **Step 4: run everything**

Run: `python3 -m pytest tests -q` → all PASS. If `test_interp_matches_golden` fails on `did_crash`/`was_repaired` dtype, check the golden was generated at the pre-change HEAD; regenerate it from `git stash` only if the failure is dtype-only and the values are identical.

- [ ] **Step 5: Commit**

```bash
git add reproc_cpCST.py tests/test_reproc.py
git commit -m "reproc_cpCST: excise crashes by default, keep interpolation behind --crash_mode interp"
```

---

### Task 4: Julia — align each valid run separately

**Files:**
- Modify: `compute_irt_parallel.jl` (`compute_irt!`, `process_files` summary)
- Test: `tests/test_irt.jl`

**Interfaces:**
- Consumes: `is_valid` column written by Task 3 (`True`/`False` strings from pandas).
- Produces: `valid_runs(valid::AbstractVector{Bool}) -> Vector{UnitRange{Int}}`, `compute_irt!(DF; radius) -> Int` (skipped short-epoch count), columns `irt`, `dtw_radius`, `n_epochs_aligned`; `const EDGE_MASK = 3`.

- [ ] **Step 1: failing test**

```julia
# tests/test_irt.jl
# Run from the repo root: julia --threads=auto tests/test_irt.jl
include(joinpath(@__DIR__, "..", "compute_irt_parallel.jl"))
using Test

const LAG = 5                      # user lags stimulus by 5 samples = 0.1667 s
function frame(n; lag=LAG)
    t = 3.5 .+ (0:n-1) ./ 30
    stim = sin.(2π .* t ./ 7)
    user = -circshift(stim, lag)   # written in the raw sign; loader flips it back
    DataFrame(flip_time=t, stim_pos=stim, user_pos=user)
end
pybool(v) = [x ? "True" : "False" for x in v]

mktempdir() do dir
    src = joinpath(dir, "in"); dst = joinpath(dir, "out"); mkpath(src)
    n = 600
    valid = trues(n); valid[200:260] .= false
    df = frame(n); df.is_valid = pybool(valid)
    CSV.write(joinpath(src, "gap.csv"), df)
    CSV.write(joinpath(src, "plain.csv"), frame(n))
    short = frame(n); sv = trues(n); sv[100:n] .= false; short.is_valid = pybool(sv)
    CSV.write(joinpath(src, "short.csv"), short)

    n_ok = process_files(src, dst)
    @test n_ok == 2

    g = CSV.read(joinpath(dst, "gap.csv"), DataFrame)
    @test all(isnan, g.irt[200:260])
    @test all(isnan, g.irt[1:3]) && all(isnan, g.irt[197:199])
    @test all(isnan, g.irt[261:263]) && all(isnan, g.irt[n-2:n])
    fin = filter(!isnan, g.irt)
    @test length(fin) == n - 61 - 12
    @test abs(median(fin) - LAG/30) < 0.02
    @test all(==(2), g.n_epochs_aligned)
    @test all(==(DTW_RADIUS), g.dtw_radius)

    p = CSV.read(joinpath(dst, "plain.csv"), DataFrame)
    @test all(==(1), p.n_epochs_aligned)
    @test count(isnan, p.irt) == 6
    @test abs(median(filter(!isnan, p.irt)) - LAG/30) < 0.02

    @test !isfile(joinpath(dst, "short.csv"))
end
println("test_irt.jl passed")
```

- [ ] **Step 2: run, expect failure**

Run: `julia --threads=auto tests/test_irt.jl`
Expected: fails on `n_epochs_aligned` missing and NaN assertions, or errors on `is_valid`.

- [ ] **Step 3: implement**

Add after `DTW_RADIUS`:

```julia
# Samples blanked at each end of every aligned run. DTW's endpoint constraint
# pins the warp path to the corners, so the estimates there are artefacts.
# Measured extent is a single sample (MAE 0.34 s at the boundary, 0.006 s at
# the next, 0.000 s thereafter), so 3 is already generous. Do not inflate it
# "to be safe": the cost lands directly on coverage.
const EDGE_MASK = 3

# pandas writes booleans as True/False; CSV.jl normally parses those as Bool,
# but be robust to a String column.
as_bool(col) = eltype(col) <: Bool ? Vector{Bool}(col) :
               [lowercase(string(x)) == "true" for x in col]

"Contiguous runs of `true`, as index ranges."
function valid_runs(valid::AbstractVector{Bool})
	runs = UnitRange{Int}[]
	start = 0
	for k in eachindex(valid)
		if valid[k] && start == 0
			start = k
		elseif !valid[k] && start != 0
			push!(runs, start:k-1)
			start = 0
		end
	end
	start != 0 && push!(runs, start:length(valid))
	return runs
end
```

Replace `compute_irt!` with a per-run version (docstring: keep points 1-3, add point 4 on epochs):

```julia
function align_run!(irt, stim, user, t, rng, radius)
	n = length(rng)
	s = collect(view(stim, rng)); u = collect(view(user, rng)); tt = view(t, rng)
	i2min, i2max = radiuslimits(radius, n, n)
	_, stim_idx, user_idx = dtw(s, u, SqEuclidean(1e-12), i2min, i2max)
	sums = zeros(Float64, n); counts = zeros(Int, n)
	@inbounds for k in eachindex(stim_idx)
		sums[stim_idx[k]] += tt[user_idx[k]]
		counts[stim_idx[k]] += 1
	end
	@inbounds for k in 1:n
		irt[rng[k]] = counts[k] == 0 ? NaN : sums[k] / counts[k] - tt[k]
	end
	for k in 1:min(EDGE_MASK, n)
		irt[rng[k]] = NaN
		irt[rng[end-k+1]] = NaN
	end
end

function compute_irt!(DF; radius::Int=DTW_RADIUS)
	n = nrow(DF)
	valid = hasproperty(DF, :is_valid) ? as_bool(DF.is_valid) : trues(n)
	irt = fill(NaN, n)
	aligned = 0; skipped = 0
	for rng in valid_runs(valid)
		# Below the band half-width the constraint is meaningless and the warp
		# collapses onto the diagonal, yielding iRT == 0 everywhere -- a value
		# that looks like a measurement and is not one. Leave it NaN instead.
		if length(rng) < radius
			skipped += 1
			continue
		end
		align_run!(irt, DF.stim_pos, DF.user_pos, DF.flip_time, rng, radius)
		aligned += 1
	end
	aligned > 0 || error("no crash-free epoch of at least $radius samples; " *
	                     "iRT is not defined for this recording")
	DF[!, :irt] = irt
	DF[!, :dtw_radius] = fill(radius, n)
	DF[!, :n_epochs_aligned] = fill(aligned, n)
	return skipped
end
```

In `process_files`: add `short_epochs = Threads.Atomic{Int}(0)`, accumulate the return of `compute_irt!`, and print `"processed $(n_ok)/$(length(csv_files)) files at radius $radius; $(short_epochs[]) epoch(s) shorter than the band left NaN"`.

- [ ] **Step 4: run**

Run: `julia --threads=auto tests/test_irt.jl` → `test_irt.jl passed`.
Also: `python3 -m pytest tests -q` still green.

- [ ] **Step 5: Commit**

```bash
git add compute_irt_parallel.jl tests/test_irt.jl
git commit -m "compute_irt: align each crash-free epoch separately, mask run edges"
```

---

### Task 5: Docs and .gitignore

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `.gitignore`

- [ ] **Step 1: `.gitignore`** — change `*_repaired.png` to `*_repaired.png` + `*_excised.png`, and add `tests/data/_golden/` is not needed (removed in Task 1). Add `.pytest_cache/`.

- [ ] **Step 2: `README.md`** — Quick start unchanged. Add a "Crash handling" rewrite: surgery is the default; what the annotation columns mean; `--crash_mode interp` and the `_interp` tag; `crash_events.csv` and `_excised.png`; per-epoch alignment with `EDGE_MASK = 3`, `n_epochs_aligned`, stub epochs shorter than the band stay NaN (three corpus files); detrend/z-score fit on valid rows; velocity NaN across the reset gap. Update the "What comes out" table: replace `was_repaired` with `is_valid`, `crash_phase`, `epoch`, `time_since_crash`, `time_to_crash`, `n_epochs_aligned`; note `was_repaired` appears only in interp outputs. Update the trimming section: `--max_samples` now applies "after annotation" in surgery mode and counts invalid rows; the "repair re-inserts ~78 samples" reasoning applies to interp mode only. Note that the first and last 3 iRT samples of every file are now NaN in both modes.

- [ ] **Step 3: `CLAUDE.md`** — same content in the "Running the Pipeline", "Architecture > File Roles" (CrashSurgery is now wired in; CrashRepair is the reference path), "Key Data Conventions" and "Crash handling" sections. Add the tests commands under a "Tests" heading. Keep the fastdtw and negative-iRT sections. Remove the sentence saying `CrashSurgery.py` is "not wired into the pipeline".

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md .gitignore
git commit -m "Document surgery as the default crash handling"
```

---

### Task 6: Corpus acceptance run

- [ ] **Step 1: run both stages on the raw corpus in both modes**

```bash
rm -f errs.log crash_count.csv
python3 reproc_cpCST.py --base_path /Users/stan/Data/retest_data/cpCST --output_path ./processed_data_surgery
julia --threads=auto compute_irt_parallel.jl ./processed_data_surgery ./irt_data_surgery
python3 reproc_cpCST.py --base_path /Users/stan/Data/retest_data/cpCST --output_path ./processed_data_interp --crash_mode interp
julia --threads=auto compute_irt_parallel.jl ./processed_data_interp ./irt_data_interp
```

- [ ] **Step 2: check the health criterion**

```bash
python3 - <<'EOF'
import glob, pandas as pd, numpy as np
rows = []
for f in sorted(glob.glob("irt_data_surgery/*task-CPT_*")) + sorted(glob.glob("irt_data_surgery/*task-CPTLITE_*")):
    d = pd.read_csv(f)
    rows.append((f.split("/")[-1][:40], len(d), int(d.n_epochs_aligned[0]), int((~d.is_valid).sum()),
                 int((d.irt < 0).sum()), float(np.nanmedian(d.irt))))
r = pd.DataFrame(rows, columns=["file", "n", "epochs", "invalid", "neg_irt", "median_irt"])
print(r[r.invalid > 0].to_string()); print("files:", len(r), "any negative:", (r.neg_irt > 0).sum())
EOF
```
Expected: 66 files, `any negative: 0`, 15 files with `invalid > 0`, `epochs` = crashes+1 except the three with a stub epoch. `errs.log` contains only the known aborted-session entry. Record the table in the PR description.

- [ ] **Step 3: clean up and commit nothing** (outputs are gitignored).

## Self-review

- Spec coverage: flag/default (T3), annotation columns (T2/T3), velocity NaN (T3), masked detrend/z-score (T3), trimming semantics (T3 test + docs T5), event table (T3), excision plot (T2/T3), gitignore (T5), Julia per-run + EDGE_MASK + n_epochs_aligned + skip summary (T4), `align_epochs` removed (T2), tests (T1-T4), corpus acceptance (T6), docs (T5). No gaps.
- Placeholders: none. "unchanged" markers in Task 3 refer to code already in the file at the named location.
- Names: `process_file(..., crash_mode=)`, `reset_events`, `record_events`, `EVENTS_FILE`, `EVENT_COLS`, `detrend_fit`, `zscale_fit`, `plot_excision`, `valid_runs`, `as_bool`, `align_run!`, `EDGE_MASK`, `n_epochs_aligned` are used consistently across tasks.
