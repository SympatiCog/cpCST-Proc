# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Continuous Phase** module of the cpCST (Continuous Phase Cognitive Stability Task) processing pipeline. It processes continuous motor tracking data where participants follow an unstable stimulus with a cursor, handles crash artifacts, and computes behavioural metrics — principally instantaneous reaction time (iRT).

Calibration Phase processing is a separate pipeline. It is **not present in this checkout** —
the `../Calibration Phase/` path the previous version of this file pointed at does not exist,
and `../README.md` is empty. Do not send anyone to either.

## Running the Pipeline

### Stage 1 — processing (Python)
```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data

# with optional signal processing
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data --detrend_vectors --zscale_vectors
```

### Stage 2 — iRT computation (Julia)
```bash
julia --threads=auto compute_irt_parallel.jl ./processed_data ./irt_data

# override the DTW band (default 120 samples = 4.0 s at 30 Hz)
julia --threads=auto compute_irt_parallel.jl ./processed_data ./irt_data --radius 60
```

**`--threads=auto` is required for the parallelism to happen at all.** Julia defaults to
`Threads.nthreads() == 1`, in which case the `Threads.@threads` loop runs serially.

Do **not** pass `--project`: the script calls `Pkg.activate(@__DIR__)` itself, so it picks up
the committed `Project.toml`/`Manifest.toml` regardless of the working directory. The first run
resolves packages and is slower.

## Architecture

**Flow:** raw CSV → `reproc_cpCST.py` → `CrashRepair` → derived metrics → processed CSV
→ `compute_irt_parallel.jl` (FastDTW) → CSV with `irt`

### File Roles

- **`reproc_cpCST.py`** — Entry point. Loads CSVs, invokes `CrashRepair`, computes
  tracking/covary/velocity columns, optionally detrends and z-scores, writes output. Skips files
  lacking `REQUIRED_COLS`. Appends to `errs.log` (with traceback) and `crash_count.csv`; both
  accumulate across runs.
- **`CrashRepair.py`** — Detects crash segments via `crash_count` diff, interpolates across gaps
  with PCHIP splines, applies tanh damping and Savitzky-Golay smoothing, resamples to 30 Hz
  carrying every column through. `plot_repair()` draws a diagnostic 3-panel plot, selecting the
  repaired frame **by time** (it is on a different grid; indexing it by row position from the
  original frame silently misaligns the traces).
- **`compute_irt_parallel.jl`** — CLI script. Banded DTW alignment of stimulus against user
  position; emits stimulus-anchored `irt` plus the `dtw_radius` used. Per-file error isolation, so
  one bad file warns rather than killing the run.
- **`DTW.jl`** — Pluto notebook, now a thin front end that `include`s the script above. It used to
  hold a second copy of the pipeline; that duplication is how a sampling-rate error came to live
  in two places at once. Do not reintroduce logic here.
- **`CrashSurgery.py`** — Exploratory prototype, **not wired into the pipeline**. Handles crashes
  by excision rather than interpolation. See "Crash handling" below.

## Key Data Conventions

- **Sampling rate is 30 Hz.** Median frame interval 0.03333 s in 131 of 133 files in the retest
  set. Derive it from `flip_time` rather than hard-coding; a hard-coded `1/60` in the Julia
  timestamp conversion previously halved every iRT value produced.
- **`user_pos` sign flip**: negated on load (`* -1`) in both languages. `reproc_cpCST.py` flips it
  back before writing. **Derived columns are not flipped back**, so in the written file
  `user_pos_vel` is the derivative of `-user_pos` and `tracking == -user_pos - stim_pos`.
  `stim_pos_vel` is unaffected. Pre-existing; changing it is a research decision.
- **Expected CSV columns**: `flip_time`, `stim_pos`, `user_pos`, `crash_count`, `did_crash`,
  `lambda_val`, `expected_time`. `lambda_val` is present in **all** files, not calibration-only.
- **Every file ends with a duplicated `flip_time`.** `reproc_cpCST.py` drops it on load. A zero dt
  makes velocity undefined and one NaN poisons a detrended column.
- **LSL marker files share the input folder** (`StimMarkers_alpha,lsl_timestamp,ext_time,hh_mm_ss`).
  Both stages skip them by schema. They are not corrupt — they are the physiological sync channel.
- **NaN handling**: Julia uses forward-fill (`ffill!`) on position columns; a leading NaN survives.
- **Velocity units**: position units per second. Columns written before Aug 2026 used a different
  (incorrect) operator and are ~33x off; old and new outputs are not comparable.

## The plant, and physiological sync

Two identities recovered from the data, verified across the retest set. Both are load-bearing.

```
d(stim_pos)/dt = 3 * lambda_val * (stim_pos + user_pos_raw)        R^2 >= 0.9999
lsl_timestamp  = onset_lsl + flip_time                             median residual 0.7 ms
```

The first means `|stim_pos|` grows exactly while `sign(e) == sign(stim_pos)`, which locates crash
onset with no tuned threshold. `CrashSurgery.check_plant_fit()` re-verifies it on new data.

The second means **`flip_time` is the join key to physiology**. Never renumber or resample it if
the output is destined for a physiological analysis. Note `CrashRepair.compute_transition()`
violates this: it reassigns `flip_time` inside each repair window, displacing timestamps by up to
1.276 s across ~14% of a crashy recording.

## Do not use `fastdtw`

`fastdtw`'s `radius` argument does **not** bound the warp. FastDTW coarsens the series, aligns at
low resolution, projects that path up and refines within `radius` cells *of the projected path*,
not of the diagonal — so a bad coarse alignment is inherited rather than corrected. Measured on one
continuous-phase file at radius 120: offsets of −1584 and +1818 samples (−53 s, +61 s), 61.6% of
the path outside the nominal radius, iRT values down to −52.8 s.

Use `dtw` with explicit limits from `radiuslimits`, which is what the script does. It caps the
offset at exactly `radius`, removes every whole-file failure in the corpus, leaves well-behaved
files unchanged to three decimals, and runs ~3.7x faster at this series length.

**Negative iRT is a useful health check.** It is physically impossible — the user cannot respond
before the stimulus. With the banded estimator, 100% of the residual negatives in the continuous
phase fall within 5 s of a repaired crash region, and no crash-free continuous file produces any.
A crash-free recording with negative iRT means something new is wrong.

## Crash handling

`CrashRepair` interpolates across crashes. On a ground-truth test (crash-free recordings with
synthetic crashes injected) this recovers iRT **worse than doing nothing**, because `smooth_dampen`
is not identity-preserving near zero and compresses the whole ±3 s window, and because the 2.583 s
controller-reset dead time is filled with roughly 78 fabricated samples per crash.

`CrashSurgery.py` is the alternative: excise `[onset .. re-acquisition]`, mark it NaN on the
original clock, and align each crash-free epoch independently. If working on either, note:

- `boundary_frac` defaults to **0.05**, not a larger value. At 0.35 the retained samples still
  inflate the signal SD by 2.7x, which wrecks anything scaled by SD — sample entropy's tolerance
  `r` above all (MSE comes out 50-60% low at every scale, in proportion to crash count).
- The iRT series decorrelates in ~1.0 s while crash spans run 3-5 s. Filling those gaps is not
  recoverable interpolation; no fill beats the series median, and a smooth fill injects fabricated
  structure exactly where the physiological response to the crash lives.
- `edge_mask` needs to be **3**, not the DTW radius. The endpoint artefact is one sample wide;
  inflating it costs coverage directly.

## Dependencies

Julia deps are pinned in `Project.toml`/`Manifest.toml` — do not add `Pkg.add()` calls to scripts.

Python: pandas, numpy, scipy, matplotlib. No requirements.txt:
```bash
pip install pandas numpy scipy matplotlib
```
