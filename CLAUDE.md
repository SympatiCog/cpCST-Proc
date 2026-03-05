# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Continuous Phase** module of the cpCST (Continuous Phase Cognitive Stability Task) processing pipeline. It processes continuous motor tracking data where participants follow a stimulus with a cursor, repairing crash artifacts and computing behavioral metrics.

The sibling **Calibration Phase** (`../Calibration Phase/`) is a separate pipeline — see the parent `../README.md` for the full project context.

## Running the Pipeline

### Main processing (Python)
```bash
# Basic
python reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data

# With optional signal processing
python reproc_cpCST.py --base_path ./raw_data --output_path ./processed_data --detrend_vectors --zscale_vectors
```

### IRT computation (Julia, parallel)
```bash
julia compute_irt_parallel.jl <source_folder> <destination_folder>
```

Julia packages are installed inline via `Pkg.add()` at the top of the scripts. Key Julia deps: CSV, DataFrames, DynamicAxisWarping, Distances, Glob, Smoothers, Query, SentinelArrays, ArgParse.

## Architecture

**Processing flow:** Raw CSV -> `reproc_cpCST.py` (orchestrator) -> `CrashRepair` (repair crashes) -> derived metrics -> output CSV

### File Roles

- **`reproc_cpCST.py`** — Entry point. Loads CSVs, invokes CrashRepair, computes tracking/covary/velocity columns, optionally detrends and z-scores, writes output. Logs errors to `errs.log` and crash counts to `crash_count.csv`.
- **`CrashRepair.py`** — `CrashRepair` class. Detects crash segments via `crash_count` diff, interpolates across gaps with PCHIP splines, applies tanh damping and Savitzky-Golay smoothing, resamples to 30 Hz. Has `plot_repair()` for diagnostic 3-panel plots.
- **`compute_irt_parallel.jl`** — CLI script. Uses FastDTW to align stimulus/user position series, computes pointwise inter-response time (IRT). Processes files in parallel via `Threads.@threads`.
- **`DTW.jl`** — Pluto notebook version of the IRT pipeline. Adds adaptive radius logic: starts at radius=120, reduces by 10 (up to 6 iterations) if mean IRT is out of range (>3 or <0).

## Key Data Conventions

- **`user_pos` sign flip**: User position is negated on load (`* -1`) in both Python and Julia code. In `reproc_cpCST.py` it's flipped again before writing output.
- **Sampling rate**: Hardcoded 30 Hz throughout (CrashRepair default, resample target, IRT timestamp conversion uses `1/60` in Julia).
- **Expected CSV columns**: `flip_time`, `stim_pos`, `user_pos`, `crash_count`, `did_crash`, `lambda_val` (calibration only).
- **NaN handling**: Julia code uses forward-fill (`ffill!`) for NaN values in position columns.

## Python Dependencies

pandas, numpy, scipy (interpolate, signal, optimize), matplotlib. No requirements.txt — install manually:
```bash
pip install pandas numpy scipy matplotlib
```
