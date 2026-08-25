# cpCST — Continuous Phase

Processing pipeline for the **Continuous Phase Cognitive Stability Task**. Participants track an
unstable stimulus with a cursor; when they lose control the stimulus runs to the screen boundary,
the trial crashes, and the controller is reset. This repo turns the raw event logs into behavioural
metrics, principally **instantaneous reaction time (iRT)**.

Calibration Phase processing is a separate pipeline, not included in this repository.

## Quick start

```bash
# Stage 1 — Python: repair, derive metrics
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data

# Stage 2 — Julia: DTW alignment, iRT
julia --threads=auto compute_irt_parallel.jl ./processed_data ./irt_data
```

Add `--detrend_vectors --zscale_vectors` to stage 1 for optional signal processing.
Add `--max_seconds N` to keep only the first N seconds of each recording — see below.
Add `--radius N` to stage 2 to override the DTW band (default 120 samples, which bounds the warp
to 4.0 s at 30 Hz).

> **Pass `--threads=auto`.** Julia defaults to a single thread, and without it the parallel loop
> runs serially. You do *not* need `--project` — the script activates its own environment.

### Dependencies

Julia packages are pinned in `Project.toml` / `Manifest.toml`; the first run resolves them.
Python needs `pandas numpy scipy matplotlib`.

## Comparing the full task against a LITE session

`CPT` runs ~596 s and `CPTLITE` ~296 s, so a like-for-like comparison needs the full task cut down
to the LITE duration:

```bash
python3 reproc_cpCST.py --base_path ./raw_data --output_path ./trimmed --max_seconds 291.3
julia --threads=auto compute_irt_parallel.jl ./trimmed ./trimmed_irt
```

Trimming happens at load, before repair and before alignment, and the output filename is tagged
`_trim<N>s`. That ordering is deliberate — DTW aligns the whole series, so iRT from a full-length
run that is then truncated differs from iRT of a run that was only ever that long.

**Use 291.3, not 296.** Seven of the 33 `CPTLITE` runs are shorter than 296 s, bottoming out at
291.30 s, so a 296 s cut leaves them untrimmed and subject N unequal. 291.3 gives every subject in
both sessions exactly 8740 samples, which is what ICC — and anything N-sensitive like sample
entropy — wants.

## What comes out

Stage 2 writes one CSV per input with the source columns plus:

| Column | Meaning |
| --- | --- |
| `irt` | Stimulus-anchored instantaneous reaction time, seconds. Positive = user lagged the stimulus. |
| `dtw_radius` | The DTW band actually used, recorded so results carry their own provenance. |
| `crash_count` | Cumulative crashes, stepping at each reset. |
| `did_crash` | True on the first sample after a reset. |
| `was_repaired` | True where `CrashRepair` rewrote the signal. |

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
negative value is a direct read on alignment failure. In the continuous phase 100% of them fall
within 5 s of a repaired crash region, and no crash-free recording produces any — so a clean
recording with negative iRT is worth investigating rather than filtering.

**`errs.log` and `crash_count.csv` append.** Clear them between runs if you want an accurate count.

**LSL marker files share the input folder.** Files with the `StimMarkers_alpha,lsl_timestamp,...`
schema are skipped automatically by both stages. They are not corrupt — they are the sync channel
to physiology, and they carry a `Crashed [...]` marker for every crash.

## Two identities worth knowing

Both were recovered by measurement rather than from documentation, and both hold across the retest
set (133 files, 515 crashes).

The plant is deterministic — the stimulus is a pure integrator on the tracking error:

```
d(stim_pos)/dt = 3 · lambda_val · (stim_pos + user_pos_raw)          R² ≥ 0.9999
```

So `|stim_pos|` grows exactly while `sign(error) == sign(stim_pos)`, which locates the moment
control was lost with no tuned threshold. `CrashSurgery.check_plant_fit()` re-verifies this on new
data; a poor fit means something is wrong with the file.

The task clock maps onto the physiological clock exactly:

```
lsl_timestamp = onset_lsl + flip_time      median residual 0.7 ms, max 3.1 ms (n = 1198)
```

**`flip_time` is therefore the join key to physiology, and must not be renumbered or resampled.**
Note that `CrashRepair` currently violates this: it reassigns `flip_time` inside each repair
window, displacing timestamps by up to 1.276 s across roughly 14% of a crashy recording. If you are
aligning behaviour to heart rate or EEG, that matters.

## Crash handling, and an open question

`CrashRepair` heals crashes: it interpolates across them with PCHIP splines, damps the excursion,
and resamples. Tested against ground truth — crash-free recordings with synthetic crashes injected
between untouched segments — this recovers iRT **worse than not repairing at all**, and it is the
only approach tested that perturbs iRT far from any crash.

`CrashSurgery.py` is a prototype of the alternative: excise the crash region, mark it NaN on the
original clock, and align each crash-free epoch independently, so no warp path crosses a crash and
nothing is fabricated. On the same test it is several times more accurate. It is **not wired into
the pipeline**; adopting it is a decision, not a merge.

The trade is real and it is the open question here: excision marks a median 14% of a crashy
recording as missing (up to 35%), against a series with no invented values in it. Two thirds of
that loss is controller-reset dead time during which no samples were logged at all — data the
current pipeline reports as present by synthesising it.
