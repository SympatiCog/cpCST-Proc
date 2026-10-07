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
Add `--crash_mode interp` to stage 1 to run the legacy interpolation path instead — see
"Crash handling".
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
numpy 2, scipy 1, matplotlib 3). The bounds are deliberate — the numbers are the deliverable — so
widen one only after the golden-file test passes against it.

### Tests

```bash
python3 -m pytest tests -q
julia --threads=auto tests/test_irt.jl
```

`tests/synth.py` builds recordings from the plant identity with crashes injected, so the
detection and annotation tests need no corpus. `tests/data/golden_interp.csv` pins the interp
path's output; it must not change.

## Comparing the full task against a LITE session

`CPT` runs ~596 s and `CPTLITE` ~296 s, so a like-for-like comparison needs the full task cut down
to the LITE duration:

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_seconds 291.3
julia --threads=auto compute_irt_parallel.jl ./trimmed ./trimmed_irt
```

Trimming happens at load, before crash handling and before alignment, and the output filename is tagged
`_trim<N>s`. That ordering is deliberate — DTW aligns the whole series, so iRT from a full-length
run that is then truncated differs from iRT of a run that was only ever that long.

**Use 285.** The shortest `CPTLITE` raw recording is 291.32 s, so that caps any target — but
291.3 is not the right choice. `--max_seconds` equalizes *duration*, not sample count, and a cut
landing inside a crash gap keeps the pre-gap data while trimming away the post-gap resumption,
leaving that gap unrepaired and the file short. Three `CPTLITE` runs have a crash gap in their
final 30 s:

| target | result across the 66 CPT/CPTLITE files |
| --- | --- |
| 291.3 s | 65 files at 8739 samples, 1 at 8671 |
| 288 s | 65 files at 8640, 1 at 8638 |
| **285 s** | **all 66 at exactly 8550** |

Equal duration is what ICC wants. If you need an exact **sample count** instead — which is what
sample entropy needs, since SampEn is N-biased and unequal N between sessions becomes a systematic
difference between the things being compared — use `--max_samples`:

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_samples 8740
```

| | applied | guarantees |
| --- | --- | --- |
| `--max_seconds` | on raw samples, before crash handling | equal **duration** |
| `--max_samples` | after crash handling, before the derived columns | equal **N** |

In surgery mode the two coincide more closely than they used to, because nothing is resampled:
`--max_samples` keeps the first N rows on the original clock, and that count **includes** rows
marked invalid. In interp mode the different insertion point is the point: trimming raw samples
cannot guarantee an output count, because repair re-inserts ~78 samples per crash and the resample
grid is rebuilt over whatever span survives. Truncating after the resample is immune to that:
`--max_samples 8740` puts all 66 CPT/CPTLITE files on exactly 8740 rows, including the 9 with
crashes.

**8740 is the largest value every CPT/CPTLITE file can supply.** Larger values start skipping files,
and a value this large skips nearly every Calibrate run — correct, since a ~182 s calibration cannot
supply 291 s of samples.

## What comes out

Stage 2 writes one CSV per input with the source columns plus:

| Column | Meaning |
| --- | --- |
| `irt` | Stimulus-anchored instantaneous reaction time, seconds. Positive = user lagged the stimulus. NaN where undefined (see below). |
| `dtw_radius` | The DTW band actually used, recorded so results carry their own provenance. |
| `n_epochs_aligned` | How many crash-free epochs were long enough to align. Constant per file. |
| `track_corr` | Peak correlation of stimulus against flipped user position over user lags 0–2 s, valid rows only. Positive = following the stimulus. Constant per file. |
| `track_lag` | The user lag, in seconds, at which `track_corr` peaks. |
| `tracking_ok` | `track_corr > 0.5`. When False, iRT is not interpretable: DTW returns plausible values whether or not anyone was tracking. |
| `crash_count` | Cumulative crashes, stepping at each reset. |
| `did_crash` | True on the first sample after a reset. |
| `crash_phase` | `ok`, `runaway` (control lost, stimulus diverging) or `reacquire` (post-reset transient). |
| `is_valid` | False from crash onset through re-acquisition. The only rows the aligner sees are the True ones. |
| `epoch` | 0 before the first reset, +1 at each reset. |
| `time_since_crash` | Seconds since the most recent reset; NaN before the first. |
| `time_to_crash` | Seconds until the next reset; NaN after the last. |

`irt` is NaN wherever `is_valid` is False, at the first and last 3 samples of every aligned
epoch (DTW's endpoint constraint pins the path there; the artefact is one sample wide), and
throughout any epoch shorter than the DTW band (120 samples), where the warp would collapse onto
the diagonal and report iRT = 0 for every sample. Three corpus files crash within 4 s of the
end and leave such a stub.

Interp-mode outputs (`--crash_mode interp`) carry `was_repaired` instead of the five annotation
columns, and are tagged `_interp` in the filename.

Stage 1 also writes `crash_events.csv` to the output folder — one row per detected crash with
onset, reset and settle times and durations — and a `<name>_excised.png` diagnostic for every
crashy file. A surgery run starts a fresh table; an interp run into the same folder leaves it
alone. A file skipped by `--max_samples` contributes no rows and no plot, so every row in the
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

**Derived columns use the flipped sign convention.** `user_pos` is negated on load and flipped back
before writing, but the derived columns are not. So in the output file `user_pos_vel` is the
derivative of `-user_pos`, and `tracking == -user_pos - stim_pos`. `stim_pos_vel` is unaffected.
This is long-standing behaviour, left alone deliberately — which convention should win is a
research decision, not a cleanup.

**Negative iRT means something is wrong.** The user cannot respond before the stimulus, so a
negative value is a direct read on alignment failure. On the 66 continuous-phase files
(15 crashy), measured 2026-09-06:

| | surgery | interp |
| --- | --- | --- |
| files with negative iRT | 0 | 11 |
| negative iRT samples | 0 | 2192 |

Every interp-mode negative falls within 5 s of a repaired crash. Under surgery mode any negative
iRT is worth investigating rather than filtering.

**The first and last 3 iRT samples of every epoch are NaN**, in both modes, so every file has at
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
Note that `CrashRepair` currently violates this: it reassigns `flip_time` inside each repair
window, displacing timestamps by up to 1.276 s across roughly 14% of a crashy recording. If you are
aligning behaviour to heart rate or EEG, that matters.

## Crash handling

When control is lost the stimulus runs to the screen boundary, the controller resets, and about
2.58 s pass with nothing logged. The default path, **surgery**, treats that as what it is: a gap.

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
samples were logged; the interpolation path reports that time as present by synthesising it.
(Measured 2026-10-05 on the surgery outputs.)

### The legacy path, kept for reference

`--crash_mode interp` runs `CrashRepair`: PCHIP interpolation across each crash, tanh damping,
Savitzky-Golay smoothing, and a resample to a uniform 30 Hz grid. It is unchanged and its output
is pinned by a golden-file test. Know what it does before using it:

- It fabricates ~78 samples per crash and reassigns `flip_time` inside each ±3 s repair window,
  displacing timestamps by up to 1.276 s across ~14% of a crashy recording. Its outputs do not
  join cleanly to physiology.
- On the ground-truth test it recovers iRT **worse than doing nothing**, because `smooth_dampen`
  is not identity-preserving near zero and compresses the whole window.
- Its outputs are tagged `_interp` so a folder can never hold an ambiguous mix.
