# Crash handling by excision (CrashSurgery) — design

Date: 2026-09-06. Status: approved in discussion, implementation to follow.

## Goal

Make `CrashSurgery` the pipeline's crash-handling path. Crashes are located from
the plant identity, the span from onset to re-acquisition is marked invalid on
the original clock, and the Julia stage aligns each crash-free epoch on its own
so no warp path ever crosses a crash. Nothing is fabricated and `flip_time` is
never renumbered or resampled.

The interpolation path (`CrashRepair`) stays in the repository, untouched and
reachable, for reference and for comparison.

## Non-goals

- Joining to the LSL physiology stream. `onset_lsl_time`, `crash_markers`,
  `add_lsl_time` and `bin_to` remain library functions and are not wired in.
- Changing the `user_pos` sign convention of the derived columns.
- Re-running or reproducing the ground-truth synthetic-crash study.

## Stage 1 — `reproc_cpCST.py`

### New flag

`--crash_mode {surgery,interp}`, default `surgery`.

- `surgery`: call `CrashSurgery.prepare(df)` after the sign flip. It returns the
  annotated frame (same rows, same `flip_time`, plus the covariate columns) and
  the event list. No `CrashRepair`, no resampling.
- `interp`: the current code path, unchanged: `CrashRepair` → `repair_tracking`
  → `plot_repair`. Output filenames gain the tag `_interp` immediately after the
  base name, before any other tag, so a folder can never hold an ambiguous mix.

### Annotation columns (surgery)

From `CrashSurgery.annotate`, present in every surgery-mode output, including
crash-free files:

| column | meaning |
| --- | --- |
| `crash_phase` | `ok`, `runaway`, `reacquire` |
| `is_valid` | False from onset through settle, inclusive |
| `epoch` | 0 before the first reset, +1 at each reset |
| `time_since_crash` | seconds since the most recent reset, NaN before the first |
| `time_to_crash` | seconds until the next reset, NaN after the last |

`is_valid` is written as `True`/`False` by pandas; CSV.jl parses both spellings
as `Bool` by default. A test covers the round trip.

### Derived columns (surgery)

`tracking`, `covary`, `abs_tracking`, `abs_covary` are computed on every row as
now. `compute_velocity` sets the derivative to NaN wherever the frame interval
exceeds `2 / fs` (0.0667 s): across the controller reset the interval is
2.58 s and the quotient is meaningless. This applies in both modes; in interp
mode the resampled grid has no such interval, so it is a no-op there.

### Detrend and z-score (surgery)

Both fit on valid, finite samples only and apply the fitted transform to every
row. Concretely: detrend fits the linear trend by least squares on
`is_valid & isfinite`, subtracts it from all rows; z-score uses mean and SD of
`is_valid & isfinite`, applied to all rows. Reason: a single crash excursion
inflates the whole-series SD by up to 2.7x, which would put crash count into the
scale of every z-scored column. In interp mode the current whole-series fit is
retained.

### Trimming

`--max_seconds` is applied at load as now, before detection. `--max_samples`
applies after annotation and before the derived columns; the retained count
includes invalid rows. `time_to_crash` may refer to a reset beyond the cut; that
is correct information and is left alone.

### Event table

`crash_events.csv` is written to the output folder, overwritten each run (not
appended), one row per detected crash:
`file, ursi, k, onset_time, reset_time, settle_time, runaway_s, reset_gap_s, settle_s`.
Crash-free files contribute no rows.

### Excision plot

For every crashy file in surgery mode, `CrashSurgery.plot_excision(annotated,
events)` draws stimulus and user position over the whole recording with each
excised span shaded, and one zoomed panel per crash (±6 s) showing onset, reset
and settle as vertical lines. Saved as `<name>_excised.png` alongside the CSV.
`.gitignore` gains `*_excised.png`.

### Logging

`crash_count.csv` and `errs.log` behave as now (append). The per-file
`crash_count` written to `crash_count.csv` is `crash_count.max()` as now.

## Stage 2 — `compute_irt_parallel.jl`

`compute_irt!` aligns each contiguous run of `is_valid` samples independently.

- If the frame has no `is_valid` column, every row is valid: one run, current
  behaviour.
- For each run of length `n_run`: if `n_run < radius`, leave NaN and count it as
  a skipped epoch. Otherwise run `dtw` with `radiuslimits(radius, n_run, n_run)`,
  compute stimulus-anchored iRT exactly as now using the run's own `flip_time`,
  and set the first and last `EDGE_MASK = 3` samples to NaN (DTW's endpoint
  constraint pins the path there; the measured artefact is one sample wide).
- Samples outside any run are NaN.
- If no run is long enough, the file errors as now ("iRT is not defined for
  this recording") and is skipped with a warning.
- New column `n_epochs_aligned` (constant per file) is written alongside
  `dtw_radius`. The run summary reports total skipped short epochs.

`ffill!` on positions is retained; it only matters for genuinely missing
samples, and positions inside invalid spans are never read by the aligner.

`align_epochs` is removed from `CrashSurgery.py`; its edge-mask findings move to
the Julia comment. `epoch_segments` stays (used by `entropy_segments` and tests).

## Testing

`tests/` (pytest), new:

- `test_crash_surgery.py`: synthetic series generated from the plant identity
  with injected crashes (runaway to boundary, 2.58 s gap, reset, settle).
  Asserts onset/reset/settle indices, `is_valid` span, `epoch` steps,
  `time_since_crash`/`time_to_crash`, and that `flip_time` is unchanged.
- `test_reproc.py`: runs `process_file` on the synthetic file in both modes.
  Surgery: annotation columns present, row count equals input, velocity NaN
  across the gap, z-score fit ignores invalid rows, `crash_events.csv` and
  `_excised.png` written. Interp: output identical to a run of the current code
  (golden file generated once from `main` before the change), filename tagged
  `_interp`.
- `test_irt.jl` (run via `julia --threads=auto tests/test_irt.jl`): writes a
  small CSV with an invalid span, runs `process_files`, asserts NaN across the
  span and its edges, finite elsewhere, `n_epochs_aligned` correct, and that a
  file without `is_valid` still produces the whole-file result.

Corpus acceptance, run by hand and recorded in the PR: surgery + iRT over the
66 CPT/CPTLITE files yields no negative iRT in any file, and the 15 crashy files
all align with the expected epoch counts (three leave a stub epoch shorter than
the band, which stays NaN).

## Documentation

CLAUDE.md and README: the flag and its default, the schema change (annotation
columns replace `was_repaired`), the `_interp` tag, the stub-epoch caveat, and
the changed detrend/z-score fit. The "Crash handling" section is rewritten to
describe surgery as the pipeline and interpolation as the reference path.
