# cpCST — Continuous Phase

Processing pipeline for the **Continuous Phase Cognitive Stability Task**. Participants track an
unstable stimulus with a cursor; when they lose control the stimulus runs to the screen boundary,
the trial crashes, and the controller is reset. This repo turns the raw event logs into behavioural
metrics, principally **instantaneous reaction time (iRT)**.

Calibration Phase processing is a separate pipeline, not included in this repository.

## Quick start

```bash
# Stage 1 — Python: locate and excise crashes, derive metrics
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data

# Stage 2 — Julia: DTW alignment, iRT
julia --threads=auto compute_irt_parallel.jl ./processed_data ./irt_data
```

Stage 1 runs files in parallel, one worker per CPU by default; `--jobs N` sets the count and
`--jobs 1` runs serially in one process, which gives the most readable tracebacks. Output is
identical either way.
Add `--detrend_vectors --zscale_vectors` to stage 1 for optional signal processing.
Add `--max_seconds N` to keep only the first N seconds of each recording — see below.
Add `--radius N` to stage 2 to override the DTW band (default 120 samples, which bounds the warp
to 4.0 s at 30 Hz).

> **Pass `--threads=auto`.** Julia defaults to a single thread, and without it the parallel loop
> runs serially. You do *not* need `--project` — the script activates its own environment.

### Dependencies

Julia packages are pinned in `Project.toml` / `Manifest.toml`; the first run resolves them.
Python dependencies are declared in `requirements.txt`, and `requirements-dev.txt` adds `pytest`:

```bash
pip install -r requirements-dev.txt    # or: uv pip install -r requirements-dev.txt
```

Each package is bounded to the major version the current results were verified under (pandas 3,
numpy 2, matplotlib 3). The bounds are deliberate — the numbers are the deliverable — so
widen one only after the golden-file test passes against it.

### Tests

```bash
python3 -m pytest tests -q
julia --threads=auto tests/test_irt.jl
```

`tests/synth.py` builds recordings from the plant identity with crashes injected, so the
detection and annotation tests need no corpus. `tests/data/golden_surgery.csv` pins stage 1's
output end to end. Change it only on purpose, and only after confirming the diff is exactly the
intended one.

## Comparing the full task against a LITE session

`CPT` runs ~596 s and `CPTLITE` ~296 s, so a like-for-like comparison needs the full task cut down
to the LITE duration:

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_seconds 285
julia --threads=auto compute_irt_parallel.jl ./trimmed ./trimmed_irt
```

Trimming happens at load, before crash handling and before alignment, and the output filename is
tagged `_trim<N>s`. That ordering is deliberate — DTW aligns the whole series, so iRT from a
full-length run that is then truncated differs from iRT of a run that was only ever that long.
The shortest `CPTLITE` raw recording is 291.32 s, which caps any target; 285 leaves a margin.

`--max_seconds` equalizes **duration**, which is what ICC wants. It does **not** equalize sample
count: nothing is logged during a controller reset, so each crash inside the window costs ~77 rows.
Across the 66 CPT/CPTLITE files at 285 s, 58 have 8550 or 8551 rows and the 8 with a crash in the
window have 8320–8474. No duration gives equal N.

If you need an exact **sample count** instead — which is what sample entropy needs, since SampEn is
N-biased and unequal N between sessions becomes a systematic difference between the things being
compared — use `--max_samples`:

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_samples 8665
```

| | applied | guarantees |
| --- | --- | --- |
| `--max_seconds` | on raw samples, before crash detection | equal **duration** |
| `--max_samples` | after crash detection, before the derived columns | equal **N** |

`--max_samples` keeps the first N rows on the original clock, and that count **includes** rows
marked invalid. **8665 is the largest value every CPT/CPTLITE file can supply.** Larger values start
skipping files, and a value this large skips nearly every Calibrate run — correct, since a ~182 s
calibration cannot supply 289 s of samples.

## What comes out

**Stage 1** (`processed_data`) writes one CSV per recording with the source columns plus:

| Column | Meaning |
| --- | --- |
| `crash_phase` | `ok`, `runaway` (control lost, stimulus diverging) or `reacquire` (post-reset transient). |
| `is_valid` | False from crash onset through re-acquisition. The only rows the aligner sees are the True ones. |
| `epoch` | 0 before the first reset, +1 at each reset. |
| `time_since_crash` | Seconds since the most recent reset; NaN before the first. |
| `time_to_crash` | Seconds until the next reset; NaN after the last. |
| `tracking` | `stim_pos + user_pos`: the plant's error term. The stimulus diverges while `sign(tracking) == sign(stim_pos)`. |
| `abs_tracking` | `\|tracking\|`. |
| `covary` | `\|user_pos\| − \|stim_pos\|`. |
| `abs_covary` | `\|covary\|`. |
| `user_pos_vel`, `stim_pos_vel`, `tracking_vel` | First derivatives, position units per second. NaN across the reset gap. |
| `sign_convention` | `raw`: every user column has the sign of the recording (see "Things that will bite you"). Absent from older outputs. |

The source columns pass through unchanged, among them `crash_count` (cumulative crashes, stepping
at each reset) and `did_crash` (True on the first sample after a reset).

**Stage 2** (`irt_data`) writes one CSV per stage-1 file with all of the above plus:

| Column | Meaning |
| --- | --- |
| `irt` | Stimulus-anchored instantaneous reaction time, seconds. Positive = user lagged the stimulus. NaN where undefined (see below). |
| `time_secs` | `flip_time` minus the file's first `flip_time`. |
| `dtw_radius` | The DTW band actually used, recorded so results carry their own provenance. |
| `n_epochs_aligned` | How many crash-free runs were long enough to align. Constant per file. |
| `track_corr` | Peak correlation of stimulus against flipped user position over user lags 0–2 s, valid rows only. Positive = following the stimulus. Constant per file. |
| `track_lag` | The user lag, in seconds, at which `track_corr` peaks. |
| `tracking_ok` | `track_corr > 0.5`. When False, iRT is not interpretable: DTW returns plausible values whether or not anyone was tracking. No file in the current corpus is flagged. |

`irt` is NaN wherever `is_valid` is False, at the first and last 3 samples of every aligned
run (DTW's endpoint constraint pins the path there; the artefact is one sample wide), and
throughout any valid run shorter than the DTW band (120 samples), where the warp would collapse
onto the diagonal and report iRT = 0 for every sample. In the continuous phase that happens three
times — once at the very start of a file, once between back-to-back crashes, once at the end.

Stage 1 also writes `crash_events.csv` to the output folder — one row per detected crash with
onset, reset and settle times and durations — and a `<name>_excised.png` diagnostic for every
crashy file. Each run starts a fresh table. A file skipped by `--max_samples` contributes no rows and no plot, so every row in the
table has a matching output CSV.

## Things that will bite you

**The data are 30 Hz, not 60.** Until August 2026 the Julia stage converted frame indices to
seconds at `1/60`, so **every iRT value it had ever produced was exactly half its true value**.
Anything computed from pre-fix outputs needs recomputing, not rescaling by eye — and the sanity
range people had internalised was calibrated against half-scale numbers.

**Velocity columns changed scale.** `compute_velocity` used to multiply by dt where it should
divide. With dt near 1/30 the two land within 11% of each other, which is why it survived for so
long. Corrected `*_vel` columns are ~33x off from older ones; old and new outputs are not
comparable.

**Every output column is in the raw sign frame, and older outputs are not.** The raw recording
has `user_pos ≈ -stim_pos` while the participant is tracking. Both stages negate `user_pos`
internally, so the computations can treat tracking as "moving with the stimulus", but that never
reaches the output: `user_pos` is written as recorded, `user_pos_vel` is its derivative, and
`tracking == stim_pos + user_pos` — the plant's error term, so the stimulus diverges exactly while
`sign(tracking) == sign(stim_pos)`. Every row carries `sign_convention == "raw"`.
Outputs written before October 2026 lack that column and have `user_pos_vel`, `tracking` and
`tracking_vel` with the **opposite** sign (and stage-2 files had `user_pos` negated as well).
Absolute-value columns, `covary`, the stimulus columns, `irt` and the tracking check are unchanged.

**Negative iRT means something is wrong.** The user cannot respond before the stimulus, so a
negative value is a direct read on alignment failure. On the 66 continuous-phase files (15 crashy),
measured 2026-09-06, excision produced none; the removed interpolation path produced 2192 negative
samples across 11 files, every one within 5 s of a repaired crash. Any negative iRT is worth
investigating rather than filtering.

**The first and last 3 iRT samples of every epoch are NaN**, so every file has at
least 6 NaN samples. Downstream code that assumes a fully finite `irt` column needs a NaN-aware
read.

**`errs.log` and `crash_count.csv` append.** Clear them between runs if you want an accurate count.
Both are written to the working directory, not the output folder. A file that errors is logged
with its traceback and contributes no rows to `crash_events.csv`; the rest of the run carries on.

**LSL marker files share the input folder.** Files with the `StimMarkers_alpha,lsl_timestamp,...`
schema are skipped automatically by both stages. They are not corrupt — they are the sync channel
to physiology, and they carry a `Crashed [...]` marker for every crash.

## Two identities worth knowing

Both were recovered by measurement rather than from documentation, and both hold across the retest
set (133 files, 515 crashes).

The plant is deterministic — the stimulus is a pure integrator on the tracking error:

```
d(stim_pos)/dt = 3 · lambda_val · (stim_pos + user_pos_raw)          median R² 0.99995
```

So `|stim_pos|` grows exactly while `sign(error) == sign(stim_pos)`, which locates the moment
control was lost with no tuned threshold. `CrashSurgery.check_plant_fit()` re-verifies this on new
data (pass `user_is_flipped=False` on raw files). Across 131 recordings R² is above 0.986 in 95%
of them and never below 0.898, so a fit in the high 0.9s is normal and one far below that means
something is wrong with the file.

The task clock maps onto the physiological clock exactly:

```
lsl_timestamp = onset_lsl + flip_time      median residual 0.7 ms, max 3.1 ms (n = 1198)
```

**`flip_time` is therefore the join key to physiology, and must not be renumbered or resampled.**
Excision never touches it. The removed interpolation path did, displacing timestamps by up to
1.276 s across roughly 14% of a crashy recording — so outputs from before October 2026 tagged
`_interp` do not join cleanly to heart rate or EEG.

**The physiological sync is not part of the supported pipeline yet.** `CrashSurgery` contains
helpers for it (`onset_lsl_time`, `crash_markers`, `add_lsl_time`, `bin_to`, `entropy_segments`),
but no stage calls them and they are not covered by the tests. Treat them as experimental: their
names, arguments and output may change before they are wired in.

## Crash handling

When control is lost the stimulus runs to the screen boundary, the controller resets, and about
2.58 s pass with nothing logged. The pipeline treats that as what it is: a gap.

- **Onset** is located from the plant identity — the first sample of the terminal divergence
  where `|stim_pos|` has passed 5% of the boundary. No tuned threshold.
- **Reset** is where `crash_count` steps. **Settle** is the end of the post-reset transient.
  The onset search never walks back past the previous reset: the reset row places the stimulus
  at ±0.005 with the error in the same sign, so it reads as "diverging", and an unbounded walk
  would hand a back-to-back crash the previous crash's runaway as its onset.
- Every row from onset through settle is marked `is_valid = False`. No row is added, removed
  or retimed; `flip_time` is untouched, so the output still joins to physiology.
- The Julia stage aligns each contiguous valid run on its own, so no warp path crosses a crash.
- Velocity is NaN across the reset gap. Detrend and z-score fit on valid rows only and apply to
  every row: a single excursion inflates the whole-series SD by up to 2.7x, and fitting on it
  would put crash count into the scale of every scaled column.

On a ground-truth test (crash-free recordings with synthetic crashes injected) this is several
times more accurate than interpolation, and it perturbs nothing far from a crash.

The cost is coverage, and it is small. In a *crashy* recording excision marks a median 1.1% of
rows invalid (0.6–7.5%); counting the reset gap, a median 1.9% of recording time is missing
(1.0–9.9%). Only 15 of the 66 continuous-phase files crash at all, so across the continuous phase
the loss is 0.6% of recording time. About 40% of that is controller dead time during which no
samples were logged.
(Measured 2026-10-05 on the surgery outputs.)

### Why there is no interpolation

Until October 2026, `--crash_mode interp` ran `CrashRepair`: PCHIP interpolation across each crash,
tanh damping, Savitzky-Golay smoothing, and a resample to a uniform 30 Hz grid. It was removed:

- It fabricated ~78 samples per crash and reassigned `flip_time` inside each ±3 s repair window,
  so its outputs do not join cleanly to physiology.
- On the ground-truth test it recovered iRT **worse than doing nothing**, because `smooth_dampen`
  is not identity-preserving near zero and compresses the whole window.
- It produced negative iRT, and without a validity mask the tracking check misfired on its output.

It remains in git history. Older outputs it wrote are tagged `_interp` and carry `was_repaired`
instead of the crash annotation columns.
