# Plots are only ever saved, never shown. Pin a non-interactive backend before
# anything imports pyplot: with --jobs > 1 every worker re-imports this module,
# and an interactive backend in a worker process hangs or warns by platform.
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import os
import sys
import traceback
import multiprocessing
from functools import partial
from scipy.signal import detrend
import argparse
from pathlib import Path
from CrashRepair import CrashRepair
import CrashSurgery
import matplotlib.pyplot as plt

def zscale(series):
    return (series - series.mean()) / series.std()

def get_ursi(filpath:str):
    fname = filpath.split('/')[-1]
    ursi = fname.split('_')[0].split('-')[-1]
    return(ursi)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--zscale_vectors", action="store_true", required=False)
    parser.add_argument("--detrend_vectors", action="store_true", required=False)
    parser.add_argument("--crash_mode", choices=CRASH_MODES, default="surgery",
                        help="surgery (default): locate each crash from the plant "
                             "identity, mark onset..settle invalid on the original "
                             "clock, never resample. interp: the legacy CrashRepair "
                             "interpolation, kept for reference; outputs are tagged "
                             "_interp.")
    parser.add_argument("--max_samples", type=int, default=None, required=False,
                        help="Keep only the first N samples of each processed "
                             "recording. Unlike --max_seconds this guarantees an "
                             "exact sample count, which is what N-sensitive "
                             "measures such as sample entropy need.")
    parser.add_argument("--max_seconds", type=float, default=None, required=False,
                        help="Keep only the first N seconds of each recording, "
                             "measured from its own first sample. Use to compare "
                             "the full-length task against a LITE session on equal "
                             "footing (CPT runs ~596 s, CPTLITE ~296 s).")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="Worker processes (default: one per CPU). Files are "
                             "independent; --jobs 1 runs serially in this process, "
                             "which is the easier way to debug.")
    return parser.parse_args()

# Only these are detrended / z-scored. The loop used to run over every column
# except flip_time, which now that the crash annotations survive would detrend
# crash_count and coerce did_crash.
SIGNAL_COLS = ("user_pos", "stim_pos", "tracking", "covary",
               "abs_tracking", "abs_covary",
               "user_pos_vel", "stim_pos_vel", "tracking_vel")

REQUIRED_COLS = {"flip_time", "stim_pos", "user_pos", "crash_count"}

# Minimum usable samples for a recording to be worth processing, at 30 Hz.
# Aborted sessions do occur: <redacted>'s entire MOBI1A session is two such
# files, of 0.000 s and 0.034 s. The corpus is cleanly bimodal -- the next
# shortest recording is 88.4 s -- so any threshold in that gap is unambiguous.
# One second is far below anything real and far above anything degenerate.
MIN_SAMPLES = 30

CRASH_MODES = ("surgery", "interp")

# One row per detected crash, written to the output folder. Unlike errs.log
# and crash_count.csv this is overwritten per run, not appended. All three are
# written only by record_result(), in the parent process, so parallel workers
# never append to the same file.
EVENTS_FILE = "crash_events.csv"
EVENT_COLS = ("file", "ursi", "k", "onset_time", "reset_time", "settle_time",
              "runaway_s", "reset_gap_s", "settle_s")


def reset_events(output_path):
    """Start a fresh event table for this run."""
    p = Path(output_path) / EVENTS_FILE
    if p.exists():
        p.unlink()


def event_rows(events, file_name, ursi):
    return [f"{file_name},{ursi},{k},{ev['onset_time']},{ev['reset_time']},"
            f"{ev['settle_time']},{ev['runaway_s']},{ev['reset_gap_s']},"
            f"{ev['settle_s']}"
            for k, ev in enumerate(events, start=1)]


def record_events(rows, output_path):
    p = Path(output_path) / EVENTS_FILE
    write_header = not p.exists()
    with open(p, "a") as f:
        if write_header:
            f.write(",".join(EVENT_COLS) + "\n")
        for row in rows:
            f.write(row + "\n")


def detrend_fit(series, mask):
    """Linear detrend fitted on `mask` rows only, subtracted from every row.

    The excised rows are left in the frame (the clock must survive), so a fit
    over the whole series would let the crash excursion tilt the trend.
    """
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    m = mask & np.isfinite(y)
    a, b = np.polyfit(x[m], y[m], 1)
    return pd.Series(y - (a * x + b), index=series.index)


def zscale_fit(series, mask):
    """Z-score with mean and SD from `mask` rows only, applied to every row.

    A single crash excursion inflates the whole-series SD by up to 2.7x, which
    would put crash count into the scale of every z-scored column.
    """
    y = series.values.astype(float)
    m = mask & np.isfinite(y)
    return (series - y[m].mean()) / y[m].std(ddof=1)


def compute_velocity(df, target_col, fs=30.0):
    """First derivative, in position units per second.

    NaN wherever the frame interval exceeds two frames: across a controller
    reset dt is ~2.58 s and the quotient is not a velocity of anything. On the
    interp path the grid is uniform, so this never fires there.

    This was `diff(pos) * diff(t) * 1000` -- a multiplication where a division
    belongs. With dt close to 1/30 the two land within 11% of each other
    (dt*1000 = 33.3 against the correct 1/dt = 30), which is why it survived;
    across a crash gap of dt = 2.58 s it is wrong by a factor of ~6600.

    NOTE: the corrected column is on a different scale to every *_vel column
    written by earlier runs. Old and new outputs are not comparable.
    """
    dt = df["flip_time"].diff()
    ok = (dt > 0) & (dt <= 2.0 / fs)
    vel = df[target_col].diff() / dt.where(ok)
    vel.iloc[0] = 0.0
    df[f"{target_col}_vel"] = vel

# def resample_data(data, target_frequency=30):
#     new_time_index = np.arange(data['flip_time'].iloc[0], data['flip_time'].iloc[-1], 1.0 / target_frequency)
#     resampled_data = pd.DataFrame({
#         'flip_time': new_time_index,
#         'stim_pos': np.interp(new_time_index, data['flip_time'], data['stim_pos']),
#         'user_pos': np.interp(new_time_index, data['flip_time'], data['user_pos'])
#     })
#     return resampled_data

def trim_to(df, max_seconds):
    """Keep the first `max_seconds` of a recording, measured from its own start.

    Applied at load, before repair and before alignment, so everything
    downstream sees only the retained window. That ordering matters: DTW
    aligns the series as a whole, so iRT computed on a full-length run and
    then truncated is not the same as iRT computed on a run that was only
    ever that long. The difference is small in aggregate -- 99%+ of samples
    are identical and medians move by <0.04 s -- but individual samples
    differ by up to ~2.9 s, and the point of trimming is to make the
    comparison exact rather than approximate.

    `flip_time` carries a task-onset offset of 1.0-11.2 s in this corpus, so
    the window is relative to the first sample, never to absolute flip_time.
    """
    t = df["flip_time"].values.astype(float)
    return df.loc[(t - t[0]) <= max_seconds].reset_index(drop=True)


def truncate_to(df, max_samples):
    """Keep the first `max_samples` rows of an already-processed recording.

    Applied AFTER repair and resampling but BEFORE the derived columns and any
    detrend/z-scoring, so every retained column is computed on exactly the
    series that gets written.

    This is deliberately a different insertion point from `trim_to`. Trimming
    raw samples cannot guarantee an output count, because repair re-inserts
    roughly 78 samples per crash and the resample grid is rebuilt over whatever
    span survives -- so N samples in can become more than N samples out.
    Truncating here is immune to that by construction.

    The cost is that repair may have drawn on context from just beyond the cut,
    bounded by its +/-3 s window. If that matters more than an exact N, use
    --max_seconds instead.
    """
    return df.iloc[:max_samples].reset_index(drop=True)


def repair_by_interpolation(df, file_path, output_path):
    """Legacy path: CrashRepair interpolates across each crash and resamples.

    Kept for reference and comparison. It fabricates ~78 samples per crash,
    reassigns flip_time inside each repair window, and recovers iRT worse than
    doing nothing on the ground-truth test; see CLAUDE.md.
    """
    cr = CrashRepair(df)
    cr.set_target_max_position()  # reset value from the user's own distribution
    repaired_df = cr.repair_tracking()
    if df.crash_count.max() > 0:
        fig = cr.plot_repair(repaired_df, segment_index=0)
        if fig is not None:
            fig.savefig(output_path / file_path.name.replace(".csv", "_interp_repaired.png"))
            plt.close()
        else:
            print("No crash report generated")
    return repaired_df


def excise(df, file_path, output_path):
    """Default path: annotate crashes on the original clock, fabricate nothing."""
    annotated, events = CrashSurgery.prepare(df)
    if events:
        fig = CrashSurgery.plot_excision(annotated, events)
        fig.savefig(output_path / file_path.name.replace(".csv", "_excised.png"))
        plt.close(fig)
    return annotated, events


def too_short(df, max_samples):
    return max_samples is not None and len(df) < max_samples


def short_message(df, max_samples):
    return f"only {len(df)} samples, fewer than the requested {max_samples}"


def process_one(file_path, output_path, detrend_vectors, zscale_vectors,
                max_seconds=None, max_samples=None, crash_mode="surgery"):
    """Process one recording; safe to run in a worker process.

    Writes only this file's own outputs (the processed CSV and its plot) and
    returns a result dict for record_result(), which owns the shared tables:
      status       "written", "skipped" or "error"
      message      skip reason, or the traceback on error
      crash_count  (ursi, max crash_count) once the recording is known usable,
                   else None -- recorded for every usable file, written or not
      events       crash_events.csv rows; only ever non-empty when written
    """
    res = {"file": str(file_path), "status": "error", "message": "",
           "crash_count": None, "events": []}

    def skipped(message):
        res.update(status="skipped", message=message, events=[])
        return res

    try:
        df = pd.read_csv(file_path)
        if not REQUIRED_COLS.issubset(df.columns):
            # LSL marker files live in the same folder and have a different schema
            return skipped("not an events file")
        # Every file in this set ends with a duplicated flip_time. A zero dt
        # makes velocity undefined and the resulting NaN poisons detrend.
        keep = np.r_[True, np.diff(df.flip_time.values.astype(float)) > 0]
        df = df.loc[keep].reset_index(drop=True)
        if max_seconds is not None and len(df) > 1:
            df = trim_to(df, max_seconds)
        if len(df) < MIN_SAMPLES:
            span = (df.flip_time.iloc[-1] - df.flip_time.iloc[0]) if len(df) > 1 else 0.0
            return skipped(f"aborted recording: {len(df)} usable sample(s), "
                           f"{span:.3f} s")
        ursi = get_ursi(str(file_path))
        res["crash_count"] = (ursi, df.crash_count.max())

        df.user_pos = df.user_pos * -1
        if crash_mode not in CRASH_MODES:
            raise ValueError(f"unknown crash_mode {crash_mode!r}; choose from {CRASH_MODES}")
        # Surgery adds no rows, so the --max_samples gate can run before it and
        # a skipped file leaves no event rows or plot behind. Interp gains
        # ~78 rows per crash, so its gate has to wait for the repaired length.
        if crash_mode == "surgery" and too_short(df, max_samples):
            return skipped(short_message(df, max_samples))
        events = []
        if crash_mode == "interp":
            df = repair_by_interpolation(df, file_path, output_path)
        else:
            df, events = excise(df, file_path, output_path)
        if max_samples is not None:
            if too_short(df, max_samples):
                return skipped(short_message(df, max_samples))
            df = truncate_to(df, max_samples)
        df["tracking"] = df.user_pos - df.stim_pos
        df["covary"] = np.abs(df.user_pos) - np.abs(df.stim_pos)
        df["abs_tracking"] = np.abs(df.tracking)
        df["abs_covary"] = np.abs(df.covary)

        for col in ["user_pos", "stim_pos", "tracking"]:
            compute_velocity(df, col)

        # Surgery leaves the excised rows in place, so fit any scaling on the
        # valid rows only. Interp has no such rows and keeps its whole-series fit.
        valid = (df["is_valid"].values if "is_valid" in df.columns
                 else np.ones(len(df), bool))
        signal_cols = [c for c in SIGNAL_COLS if c in df.columns]
        if detrend_vectors:
            for col in signal_cols:
                df[col] = (detrend(df[col]) if crash_mode == "interp"
                           else detrend_fit(df[col], valid))

        if zscale_vectors:
            for col in signal_cols:
                df[col] = (zscale(df[col]) if crash_mode == "interp"
                           else zscale_fit(df[col], valid))

        # df = resample_data(df)
        
        # Annotate filename with tags
        filename = file_path.name.replace(".csv", "")
        if crash_mode == "interp":
            filename += "_interp"
        if detrend_vectors:
            filename += "_detrend"
        if zscale_vectors:
            filename += "_zscale"
        if max_seconds is not None:
            filename += f"_trim{max_seconds:g}s"
        if max_samples is not None:
            filename += f"_trim{max_samples}samp"
        filename += ".csv"
        
        df.user_pos = df.user_pos * -1
        df.to_csv(output_path / filename, index=False)
        # Event rows only for files that were written, so every row in
        # crash_events.csv has a matching output CSV.
        res.update(status="written", message=filename,
                   events=event_rows(events, file_path.name, ursi))
    except Exception:
        # was a bare `except:`, which also swallowed KeyboardInterrupt and
        # logged a filename with no indication of what went wrong
        res.update(status="error", message=traceback.format_exc(), events=[])
    return res


def record_result(res, output_path):
    """Write one result's share of the shared tables. Parent process only."""
    if res["crash_count"] is not None:
        ursi, crash_count = res["crash_count"]
        with open("crash_count.csv", 'a') as f:
            f.write(f"{ursi},{crash_count}\n")
    if res["events"]:
        record_events(res["events"], output_path)
    if res["status"] == "written":
        print(res["file"])
    elif res["status"] == "skipped":
        print(f"skip ({res['message']}): {res['file']}")
    else:
        print(res["message"], end="", file=sys.stderr)
        print(f"err:{res['file']}")
        with open("errs.log", 'a') as f:
            f.write(f"{res['file']}\n{res['message']}\n")


def process_file(file_path, output_path, detrend_vectors, zscale_vectors,
                 max_seconds=None, max_samples=None, crash_mode="surgery"):
    """process_one() and record_result() together: the serial, single-file path."""
    res = process_one(file_path, output_path, detrend_vectors, zscale_vectors,
                      max_seconds, max_samples, crash_mode)
    record_result(res, output_path)
    return res

def main():
    args = parse_arguments()
    base_path = Path(args.base_path)
    output_path = Path(args.output_path)

    # Fail loudly on a bad invocation. Both of these used to exit 0 having
    # silently done nothing, which reads as success.
    if not base_path.is_dir():
        raise SystemExit(f"base_path is not a directory: {base_path}")
    csv_files = sorted(base_path.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no CSV files found in {base_path}")
    if args.jobs < 1:
        raise SystemExit(f"--jobs must be at least 1, got {args.jobs}")

    output_path.mkdir(parents=True, exist_ok=True)
    # Only a surgery run owns the event table. An interp run into the same
    # folder must leave an earlier surgery run's table alone.
    if args.crash_mode == "surgery":
        reset_events(output_path)

    print(f"crash handling: {args.crash_mode}")
    if args.max_seconds is not None:
        print(f"trimming every recording to its first {args.max_seconds:g} s")
    if args.max_samples is not None:
        print(f"truncating every recording to its first {args.max_samples} samples")
    work = partial(process_one, output_path=output_path,
                   detrend_vectors=args.detrend_vectors,
                   zscale_vectors=args.zscale_vectors,
                   max_seconds=args.max_seconds, max_samples=args.max_samples,
                   crash_mode=args.crash_mode)
    jobs = min(args.jobs, len(csv_files))
    print(f"{len(csv_files)} file(s), {jobs} worker(s)")
    counts = {"written": 0, "skipped": 0, "error": 0}

    def drain(results):
        for res in results:
            record_result(res, output_path)
            counts[res["status"]] += 1

    if jobs == 1:
        drain(map(work, csv_files))
    else:
        # spawn on every platform: fork after numpy/matplotlib have started
        # threads can deadlock, and macOS already defaults to spawn. imap keeps
        # input order, so the shared tables come out in sorted-file order
        # exactly as a serial run writes them.
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            drain(pool.imap(work, csv_files))
    print(f"done: {counts['written']} written, {counts['skipped']} skipped, "
          f"{counts['error']} error(s)")

if __name__ == "__main__":
    main()
