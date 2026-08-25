import pandas as pd
import numpy as np
import traceback
from glob import glob
from scipy.signal import detrend
import argparse
from pathlib import Path
from CrashRepair import CrashRepair
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
    parser.add_argument("--max_seconds", type=float, default=None, required=False,
                        help="Keep only the first N seconds of each recording, "
                             "measured from its own first sample. Use to compare "
                             "the full-length task against a LITE session on equal "
                             "footing (CPT runs ~596 s, CPTLITE ~296 s).")
    return parser.parse_args()

# Only these are detrended / z-scored. The loop used to run over every column
# except flip_time, which now that the crash annotations survive would detrend
# crash_count and coerce did_crash.
SIGNAL_COLS = ("user_pos", "stim_pos", "tracking", "covary",
               "abs_tracking", "abs_covary",
               "user_pos_vel", "stim_pos_vel", "tracking_vel")

REQUIRED_COLS = {"flip_time", "stim_pos", "user_pos", "crash_count"}

# Minimum usable samples for a recording to be worth processing, at 30 Hz.
# Aborted sessions do occur: X10961871's entire MOBI1A session is two such
# files, of 0.000 s and 0.034 s. The corpus is cleanly bimodal -- the next
# shortest recording is 88.4 s -- so any threshold in that gap is unambiguous.
# One second is far below anything real and far above anything degenerate.
MIN_SAMPLES = 30


def compute_velocity(df, target_col):
    """First derivative, in position units per second.

    This was `diff(pos) * diff(t) * 1000` -- a multiplication where a division
    belongs. With dt close to 1/30 the two land within 11% of each other
    (dt*1000 = 33.3 against the correct 1/dt = 30), which is why it survived;
    across a crash gap of dt = 2.58 s it is wrong by a factor of ~6600.

    NOTE: the corrected column is on a different scale to every *_vel column
    written by earlier runs. Old and new outputs are not comparable.
    """
    dt = df["flip_time"].diff()
    vel = df[target_col].diff() / dt.where(dt > 0)
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


def process_file(file_path, output_path, detrend_vectors, zscale_vectors,
                 max_seconds=None):
    try:
        df = pd.read_csv(file_path)
        if not REQUIRED_COLS.issubset(df.columns):
            # LSL marker files live in the same folder and have a different schema
            print(f"skip (not an events file): {file_path}")
            return
        # Every file in this set ends with a duplicated flip_time. A zero dt
        # makes velocity undefined and the resulting NaN poisons detrend.
        keep = np.r_[True, np.diff(df.flip_time.values.astype(float)) > 0]
        df = df.loc[keep].reset_index(drop=True)
        if max_seconds is not None and len(df) > 1:
            df = trim_to(df, max_seconds)
        if len(df) < MIN_SAMPLES:
            span = (df.flip_time.iloc[-1] - df.flip_time.iloc[0]) if len(df) > 1 else 0.0
            print(f"skip (aborted recording: {len(df)} usable sample(s), "
                  f"{span:.3f} s): {file_path}")
            return
        ursi = get_ursi(str(file_path))
        crash_count = df.crash_count.max()
        with open("crash_count.csv",'a') as f:
            f.write(f"{ursi},{crash_count}\n")

        df.user_pos = df.user_pos * -1
        cr = CrashRepair(df)
        cr.set_target_max_position() # Set the reset value for crash repair based 
                                    # on the user's data distribution.
        repaired_df = cr.repair_tracking()
        if df.crash_count.max() > 0:
            fig = cr.plot_repair(repaired_df, segment_index=0)
            if fig is not None:
                fig.savefig(output_path / file_path.name.replace(".csv", "_repaired.png"))
                plt.close()
            else:
                print("No crash report generated")
        df = repaired_df
        df["tracking"] = df.user_pos - df.stim_pos
        df["covary"] = np.abs(df.user_pos) - np.abs(df.stim_pos)
        df["abs_tracking"] = np.abs(df.tracking)
        df["abs_covary"] = np.abs(df.covary)

        for col in ["user_pos", "stim_pos", "tracking"]:
            compute_velocity(df, col)

        signal_cols = [c for c in SIGNAL_COLS if c in df.columns]
        if detrend_vectors:
            for col in signal_cols:
                df[col] = detrend(df[col])

        if zscale_vectors:
            for col in signal_cols:
                df[col] = zscale(df[col])

        # df = resample_data(df)
        
        # Annotate filename with tags
        filename = file_path.name.replace(".csv", "")
        if detrend_vectors:
            filename += "_detrend"
        if zscale_vectors:
            filename += "_zscale"
        if max_seconds is not None:
            filename += f"_trim{max_seconds:g}s"
        filename += ".csv"
        
        df.user_pos = df.user_pos * -1
        df.to_csv(output_path / filename, index=False)
    except Exception:
        # was a bare `except:`, which also swallowed KeyboardInterrupt and
        # logged a filename with no indication of what went wrong
        traceback.print_exc()
        print(f"err:{file_path}")
        with open("errs.log", 'a') as f:
            f.write(f"{file_path}\n{traceback.format_exc()}\n")

def main():
    args = parse_arguments()
    base_path = Path(args.base_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.max_seconds is not None:
        print(f"trimming every recording to its first {args.max_seconds:g} s")
    for file_path in base_path.glob("*.csv"):
        print(file_path)
        process_file(file_path, output_path, args.detrend_vectors,
                     args.zscale_vectors, args.max_seconds)

if __name__ == "__main__":
    main()
