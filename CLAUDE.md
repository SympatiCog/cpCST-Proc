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

# legacy interpolation path, for reference only; outputs are tagged _interp
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data --crash_mode interp

# trim every recording to its first N seconds (for full-vs-LITE comparison)
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_seconds 291.3
```

`--max_seconds` trims at load, **before** crash handling and alignment, and tags the output filename
`_trim<N>s`. The ordering is deliberate: DTW aligns the series as a whole, so iRT computed on a
full-length run and then truncated is not the same as iRT from a run that was only ever that long.
The difference is modest (99%+ of samples identical, medians within 0.04 s) but individual samples
diverge by up to ~2.9 s, and the point of trimming is an exact comparison.

Durations in this corpus: `CPT` 596.5 s, `CPTLITE` 296.4 s — almost exactly 2:1. The shortest
`CPTLITE` raw recording is 291.32 s, which caps any usable target.

`--max_seconds` equalizes **duration**, which is what ICC wants — equal task exposure. It does not
automatically equalize sample count. A cut that lands *inside* a crash gap keeps the pre-gap data,
trims away the post-gap resumption, and so leaves that gap unrepaired and the file short. Three
`CPTLITE` runs have a crash gap in their final 30 s, which is exactly where a sensible cut falls:

| target | result across the 66 CPT/CPTLITE files |
| --- | --- |
| 291.3 s | 65 files at 8739 samples, 1 at 8671 (<redacted>'s gap spans 289.02–291.60 s) |
| 288 s | 65 files at 8640, 1 at 8638 (<redacted>'s gap spans 287.92–290.50 s) |
| **285 s** | **all 66 at exactly 8550 samples** — verified |

Use **285** when equal N is wanted from a time-based cut.

### `--max_samples`: an exact sample count

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_samples 8740
```

Guarantees N by construction, which is what N-sensitive measures need. It applies at a **different
point in the pipeline** from `--max_seconds`:

| | applied | guarantees |
| --- | --- | --- |
| `--max_seconds` | on raw samples, before crash handling | equal **duration** |
| `--max_samples` | after crash handling, before the derived columns and detrend/z-scoring | equal **N** |

In surgery mode (the default) nothing is resampled, so the retained N rows are the first N rows of
the original clock and the count **includes** rows marked `is_valid == False`. In interp mode the
distinction is the whole reason `--max_samples` exists: trimming raw samples could not guarantee an
output count, because repair re-inserts ~78 samples per crash and the resample grid is rebuilt over
whatever span survives. Verified on the interp path: `--max_samples 8740` puts all 66 CPT/CPTLITE
files on exactly 8740 rows, including the 9 containing crashes.

Because truncation precedes the derived columns and detrend/z-scoring, every written column is
computed on exactly the series that gets written — a CPT run z-scored under `--max_samples` is
standardised over its retained window, not over the full 10 minutes.

**8740 is the largest value every CPT/CPTLITE file can supply** (the binding file is
`<redacted>_ses-MOBI2B_CPTLITE`). Anything larger and files start being skipped. Note that a value
this large skips almost every Calibrate run, which is correct — a ~182 s calibration cannot supply
291 s of samples — so point `--base_path` at continuous-phase files, or expect the skips.

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

- **`reproc_cpCST.py`** — Entry point. Loads CSVs, runs crash handling (`--crash_mode`, default
  `surgery`), computes tracking/covary/velocity columns, optionally detrends and z-scores, writes
  output. Skips files lacking `REQUIRED_COLS`. Writes `crash_events.csv` and `<name>_excised.png`
  to the output folder (overwritten per run). Appends to `errs.log` (with traceback) and
  `crash_count.csv`; those two accumulate across runs. Runs files in a `spawn` process pool
  (`--jobs`, default one per CPU; `--jobs 1` is serial and in-process). The work is split so
  that only the parent touches shared files: `process_one` writes a file's own CSV and plot and
  returns a result; `record_result` writes the three shared tables in sorted-file order, so
  serial and parallel runs are byte-identical (verified on the full corpus, 24.5 s → 4.3 s on 14
  cores). `process_file` is the two together. Do not reintroduce a shared-file write inside
  `process_one`.
- **`CrashSurgery.py`** — The default crash handling. `detect_events` locates onset from the plant
  identity, reset from the `crash_count` step, settle from the post-reset transient. `annotate`
  adds `crash_phase`, `is_valid`, `epoch`, `time_since_crash`, `time_to_crash` without touching
  a row or a timestamp. `plot_excision` draws the diagnostic. The LSL hand-off and binning helpers
  (`onset_lsl_time`, `crash_markers`, `add_lsl_time`, `bin_to`, `entropy_segments`) are library
  functions, not wired in.
- **`CrashRepair.py`** — The legacy path, reachable via `--crash_mode interp`, kept for reference
  and comparison. Interpolates across gaps with PCHIP splines, applies tanh damping and
  Savitzky-Golay smoothing, resamples to 30 Hz. **Do not modify**: `tests/data/golden_interp.csv`
  pins its output. `plot_repair()` selects the repaired frame **by time** (it is on a different
  grid; indexing it by row position from the original frame silently misaligns the traces).
- **`compute_irt_parallel.jl`** — CLI script. Banded DTW alignment of stimulus against user
  position, run separately over each contiguous `is_valid` run; emits stimulus-anchored `irt`,
  `dtw_radius` and `n_epochs_aligned`. Per-file error isolation, so one bad file warns rather than
  killing the run.
- **`DTW.jl`** — Pluto notebook, now a thin front end that `include`s the script above. It used to
  hold a second copy of the pipeline; that duplication is how a sampling-rate error came to live
  in two places at once. Do not reintroduce logic here.
- **`tests/`** — `python3 -m pytest tests -q` and `julia --threads=auto tests/test_irt.jl`.
  `synth.py` builds recordings from the plant identity with injected crashes.

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
  In surgery mode `irt` is NaN wherever `is_valid` is False, at the first and last `EDGE_MASK = 3`
  samples of every aligned epoch (so 6 per crash-free file), and throughout any epoch shorter
  than the band. Velocity columns are NaN across the reset gap (dt > 2 frames).
- **Scaling in surgery mode**: `--detrend_vectors` and `--zscale_vectors` fit on valid rows only
  and apply to every row. Interp mode keeps its whole-series fit.
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
the output is destined for a physiological analysis. Surgery mode honours this by construction.
`CrashRepair.compute_transition()` (interp mode) violates it: it reassigns `flip_time` inside each
repair window, displacing timestamps by up to 1.276 s across ~14% of a crashy recording.

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
before the stimulus. Verified 2026-09-06 on the 66 CPT/CPTLITE files: surgery mode produces **no
negative iRT in any file**; interp mode produces 2192 negative samples across 11 of the 15 crashy
files, all within 5 s of a repaired region. Any negative iRT under surgery mode means something
new is wrong.

## Crash handling

Surgery (default) excises `[onset .. settle]` by marking it `is_valid == False` on the original
clock; the Julia stage aligns each crash-free epoch independently. Interp (`--crash_mode interp`)
is `CrashRepair`, which interpolates across crashes. On a ground-truth test (crash-free recordings
with synthetic crashes injected) interpolation recovers iRT **worse than doing nothing**, because
`smooth_dampen` is not identity-preserving near zero and compresses the whole ±3 s window, and
because the 2.583 s controller-reset dead time is filled with roughly 78 fabricated samples per
crash. Excision is several times more accurate on the same test.

Corpus numbers (66 continuous files, 15 crashy): excision marks 0.6–7.5% of a crashy file invalid;
three files crash within 4 s of the end and leave a stub epoch shorter than the band, which stays
NaN (`n_epochs_aligned` is then one less than crashes + 1).

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
without re-running the golden-file test against it:
```bash
pip install -r requirements-dev.txt
```
