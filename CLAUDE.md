# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Continuous Phase** module of the cpCST (Continuous Phase Cognitive Stability Task) processing pipeline. It processes continuous motor tracking data where participants follow an unstable stimulus with a cursor, locates and excises crash artifacts, and computes behavioural metrics — principally instantaneous reaction time (iRT).

Calibration Phase processing is a separate pipeline. It is **not present in this checkout** —
the `../Calibration Phase/` path the previous version of this file pointed at does not exist,
and `../README.md` is empty. Do not send anyone to either.

## Running the Pipeline

### Stage 1 — processing (Python)
```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data

# serial, in one process (easier to debug); default is one worker per CPU
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data --jobs 1

# with optional signal processing
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data --detrend_vectors --zscale_vectors

# trim every recording to its first N seconds (for full-vs-LITE comparison)
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_seconds 285
```

`--max_seconds` trims at load, **before** crash handling and alignment, and tags the output filename
`_trim<N>s`. The ordering is deliberate: DTW aligns the series as a whole, so iRT computed on a
full-length run and then truncated is not the same as iRT from a run that was only ever that long.
The difference is modest (99%+ of samples identical, medians within 0.04 s) but individual samples
diverge by up to ~2.9 s, and the point of trimming is an exact comparison.

Durations in this corpus: `CPT` 596.5 s, `CPTLITE` 296.4 s — almost exactly 2:1. The shortest
`CPTLITE` raw recording is 291.32 s, which caps any usable target.

`--max_seconds` equalizes **duration**, which is what ICC wants — equal task exposure. It does
**not** equalize sample count: nothing is logged during a controller reset, so every crash inside
the window costs ~77 rows. Measured 2026-10-07 on the 66 CPT/CPTLITE files:

| target | rows per file |
| --- | --- |
| 291.3 s | 49 at 8739, 8 at 8740; 9 crashy files from 8509 to 8672 |
| 288 s | 49 at 8640, 9 at 8641; 8 crashy files from 8410 to 8564 |
| 285 s | 49 at 8550, 9 at 8551; 8 crashy files from 8320 to 8474 |

No duration gives equal N. When N matters, use `--max_samples`.

### `--max_samples`: an exact sample count

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_samples 8665
```

Guarantees N by construction, which is what N-sensitive measures (sample entropy above all) need.
It applies at a **different point in the pipeline** from `--max_seconds`:

| | applied | guarantees |
| --- | --- | --- |
| `--max_seconds` | on raw samples, before crash detection | equal **duration** |
| `--max_samples` | after crash detection, before the derived columns and detrend/z-scoring | equal **N** |

Nothing is resampled, so the retained N rows are the first N rows of the original clock, and the
count **includes** rows marked `is_valid == False`. A file with fewer than N rows is skipped before
crash detection, so it leaves no event rows or plot behind.

Because truncation precedes the derived columns and detrend/z-scoring, every written column is
computed on exactly the series that gets written — a CPT run z-scored under `--max_samples` is
standardised over its retained window, not over the full 10 minutes.

**8665 is the largest value every CPT/CPTLITE file can supply** (the binding file is a CPTLITE run
with three crashes; the next shortest has 8741). Anything larger and files start being skipped.
A value this large also skips almost every Calibrate run, which is correct — a ~182 s calibration
cannot supply 289 s of samples — so point `--base_path` at continuous-phase files, or expect the
skips. (The 8740 quoted before October 2026 came from the removed interpolation path, which
resampled every file onto a uniform grid.)

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

**Flow:** raw CSV → `reproc_cpCST.py` → `CrashSurgery.prepare` (annotate on the original clock)
→ derived metrics → processed CSV → `compute_irt_parallel.jl` (banded DTW per crash-free epoch)
→ CSV with `irt`

### File Roles

- **`reproc_cpCST.py`** — Entry point. Loads CSVs, runs crash detection (`CrashSurgery`), computes tracking/covary/velocity columns, optionally detrends and z-scores, writes
  output. Skips files lacking `REQUIRED_COLS`. Writes `crash_events.csv` and `<name>_excised.png`
  to the output folder (overwritten per run). Appends to `errs.log` (with traceback) and
  `crash_count.csv`; those two accumulate across runs. Runs files in a `spawn` process pool
  (`--jobs`, default one per CPU; `--jobs 1` is serial and in-process). The work is split so
  that only the parent touches shared files: `process_one` writes a file's own CSV and plot and
  returns a result; `record_result` writes the three shared tables in sorted-file order, so
  serial and parallel runs are byte-identical (verified on the full corpus, 24.5 s → 4.3 s on 14
  cores). `process_file` is the two together. Do not reintroduce a shared-file write inside
  `process_one`.
- **`CrashSurgery.py`** — Crash handling. `detect_events` locates onset from the plant
  identity, reset from the `crash_count` step, settle from the post-reset transient. `annotate`
  adds `crash_phase`, `is_valid`, `epoch`, `time_since_crash`, `time_to_crash` without touching
  a row or a timestamp. `plot_excision` draws the diagnostic. The LSL hand-off and binning helpers
  (`onset_lsl_time`, `crash_markers`, `add_lsl_time`, `bin_to`, `entropy_segments`) are library
  functions, not wired in, untested, and documented as experimental in the README and the v1.0.0 release; that work happens on the `physio-sync` branch.
- **`compute_irt_parallel.jl`** — CLI script. Banded DTW alignment of stimulus against user
  position, run separately over each contiguous `is_valid` run; emits stimulus-anchored `irt`,
  `dtw_radius`, `n_epochs_aligned`, and a tracking check (`track_corr`, `track_lag`,
  `tracking_ok = track_corr > 0.5`, computed on the flipped user position over valid rows only).
  No corpus file is flagged (CPT/CPTLITE minimum 0.655). Per-file error isolation, so one bad file warns rather than
  killing the run.
- **`DTW.jl`** — Pluto notebook, now a thin front end that `include`s the script above. It used to
  hold a second copy of the pipeline; that duplication is how a sampling-rate error came to live
  in two places at once. Do not reintroduce logic here.
- **`tests/`** — `python3 -m pytest tests -q` and `julia --threads=auto tests/test_irt.jl`.
  `synth.py` builds recordings from the plant identity with injected crashes.
  `data/golden_surgery.csv` pins stage 1's output end to end; change it only on purpose, after
  confirming the diff is exactly the intended one.

## Key Data Conventions

- **Sampling rate is 30 Hz.** Median frame interval 0.03333 s in 131 of 133 files in the retest
  set. Derive it from `flip_time` rather than hard-coding; a hard-coded `1/60` in the Julia
  timestamp conversion previously halved every iRT value produced.
- **Sign convention: every written column is in the raw frame.** The raw recording has
  `user_pos ≈ -stim_pos` while tracking. Both stages negate `user_pos` *internally* (crash
  detection, DTW alignment, the tracking check all assume "user moves with the stimulus"), and
  that flipped frame must never reach an output. Stage 1 restores the sign before computing any
  derived column; stage 2 flips a copy and writes `user_pos` untouched. So `user_pos_vel` is
  `d(user_pos)/dt` and `tracking == stim_pos + user_pos`, the plant's error term. Every output
  row carries `sign_convention == "raw"`. Files without that column were written before
  Oct 2026 and have `user_pos_vel`, `tracking` and `tracking_vel` **negated** (and stage-2
  `user_pos` negated too); `abs_*`, `covary`, `stim_*`, `irt` and `track_*` are unaffected.
  The `<name>_excised.png` diagnostic still plots the flipped user trace, labelled as such,
  because overlaying it on the stimulus is the point of the plot.
- **Expected CSV columns**: `flip_time`, `stim_pos`, `user_pos`, `crash_count`, `did_crash`,
  `lambda_val`, `expected_time`. `lambda_val` is present in **all** files, not calibration-only.
- **Every file ends with a duplicated `flip_time`.** `reproc_cpCST.py` drops it on load. A zero dt
  makes velocity undefined and one NaN poisons a detrended column.
- **LSL marker files share the input folder** (`StimMarkers_alpha,lsl_timestamp,ext_time,hh_mm_ss`).
  Both stages skip them by schema. They are not corrupt — they are the physiological sync channel.
- **NaN handling**: Julia uses forward-fill (`ffill!`) on position columns; a leading NaN survives.
  `irt` is NaN wherever `is_valid` is False, at the first and last `EDGE_MASK = 3`
  samples of every aligned epoch (so 6 per crash-free file), and throughout any epoch shorter
  than the band. Velocity columns are NaN across the reset gap (dt > 2 frames).
- **Scaling**: `--detrend_vectors` and `--zscale_vectors` fit on valid rows only and apply to
  every row.
- **Velocity units**: position units per second. Columns written before Aug 2026 used a different
  (incorrect) operator and are ~33x off; old and new outputs are not comparable.

## The plant, and physiological sync

Two identities recovered from the data, verified across the retest set. Both are load-bearing.

```
d(stim_pos)/dt = 3 * lambda_val * (stim_pos + user_pos_raw)        median R^2 0.99995, min 0.898
lsl_timestamp  = onset_lsl + flip_time                             median residual 0.7 ms
```

The first means `|stim_pos|` grows exactly while `sign(e) == sign(stim_pos)`, which locates crash
onset with no tuned threshold. `CrashSurgery.check_plant_fit()` re-verifies it on new data —
pass `user_is_flipped=False` on raw files; the default assumes `user_pos` is already negated.
R^2 measured 2026-10-06 across 131 recordings (two aborted ones excluded): 75 reach 0.9999,
98 reach 0.999, 5th percentile 0.986, minimum 0.898 (a Calibrate run).
Continuous phase alone: median 0.99988, minimum 0.978. A fit in the high 0.9s is normal; one far
below that is worth a look.

The second means **`flip_time` is the join key to physiology**. Never renumber or resample it if
the output is destined for a physiological analysis. Excision honours this by construction. The
removed interpolation path violated it, displacing timestamps by up to 1.276 s across ~14% of a
crashy recording.

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
before the stimulus. Verified 2026-09-06 on the 66 CPT/CPTLITE files: excision produces **no
negative iRT in any file** (the removed interpolation path produced 2192 negative samples across
11 of the 15 crashy files, all within 5 s of a repaired region). Any negative iRT means something
new is wrong.

## Crash handling

Crashes are excised: `[onset .. settle]` is marked `is_valid == False` on the original clock, and
the Julia stage aligns each crash-free run independently. Nothing is added, removed or retimed.

**Interpolation was removed in October 2026 — do not reintroduce it.** The old `CrashRepair`
path (`--crash_mode interp`) filled each crash with PCHIP splines, tanh damping and Savitzky-Golay
smoothing, then resampled to 30 Hz. On a ground-truth test (crash-free recordings with synthetic
crashes injected) it recovered iRT **worse than doing nothing**, because `smooth_dampen` is not
identity-preserving near zero and compresses the whole ±3 s window, and because the 2.583 s
controller-reset dead time was filled with ~78 fabricated samples per crash. It also retimed
`flip_time`, produced negative iRT, and had no validity mask, so the tracking check misfired on its
output. Excision is several times more accurate on the same test. It remains in git history.

Corpus numbers (66 continuous files, 15 crashy): excision marks 0.6–7.5% of a crashy file invalid;
three valid runs are shorter than the band and stay NaN — one at the very start of a file, one
between back-to-back crashes, one at the end (measured 2026-10-07). A crash very near the end
usually leaves no valid run after it at all, so `n_epochs_aligned` counts aligned valid runs and
need not equal crashes + 1.

If working on `CrashSurgery`, note:

- `boundary_frac` defaults to **0.05**, not a larger value. At 0.35 the retained samples still
  inflate the signal SD by 2.7x, which wrecks anything scaled by SD — sample entropy's tolerance
  `r` above all (MSE comes out 50-60% low at every scale, in proportion to crash count).
- The iRT series decorrelates in ~1.0 s while crash spans run 3-5 s. Filling those gaps is not
  recoverable interpolation; no fill beats the series median, and a smooth fill injects fabricated
  structure exactly where the physiological response to the crash lives.
- `EDGE_MASK` in the Julia script is **3**, not the DTW radius. The endpoint artefact is one sample
  wide; inflating it costs coverage directly.
- At the reset row both `time_since_crash` and `time_to_crash` are 0.

## Dependencies

Julia deps are pinned in `Project.toml`/`Manifest.toml` — do not add `Pkg.add()` calls to scripts.

Python deps are declared in `requirements.txt` (runtime) and `requirements-dev.txt` (adds pytest).
Each is bounded to the major version the results were verified under — do not widen a bound
without re-running the golden-file test (`tests/data/golden_surgery.csv`) against it:
```bash
pip install -r requirements-dev.txt
```
