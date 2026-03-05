import pandas as pd
import numpy as np
from scipy.signal import detrend
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from CrashRepair import CrashRepair
import matplotlib.pyplot as plt

def zscale(series):
    std = series.std()
    if std == 0:
        return series - series.mean()
    return (series - series.mean()) / std

def get_ursi(file_path: Path):
    return file_path.stem.split('_')[0].split('-')[-1]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--zscale_vectors", action="store_true", required=False)
    parser.add_argument("--detrend_vectors", action="store_true", required=False)
    return parser.parse_args()

def compute_velocity(df, target_col):
    dt = df['flip_time'].diff()
    vel = (df[target_col].diff() / dt).fillna(0)
    vel.replace([np.inf, -np.inf], 0, inplace=True)
    df[f"{target_col}_vel"] = vel

def process_file(file_path, output_path, detrend_vectors, zscale_vectors):
    try:
        df = pd.read_csv(file_path)
        ursi = get_ursi(file_path)
        crash_count = df.crash_count.max()

        df.user_pos = df.user_pos * -1
        cr = CrashRepair(df)
        cr.set_target_max_position()
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

        for col in df.columns:
            if col != "flip_time":
                if detrend_vectors:
                    df[col] = detrend(df[col])
                if zscale_vectors:
                    df[col] = zscale(df[col])

        filename = file_path.name.replace(".csv", "")
        if detrend_vectors:
            filename += "_detrend"
        if zscale_vectors:
            filename += "_zscale"
        filename += ".csv"

        df.user_pos = df.user_pos * -1
        df.to_csv(output_path / filename, index=False)
        return ursi, crash_count, None
    except Exception as e:
        traceback.print_exc()
        return None, None, str(file_path)

def main():
    args = parse_arguments()
    base_path = Path(args.base_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(base_path.glob("*.csv"))
    print(f"Processing {len(files)} files...")
    worker = partial(process_file, output_path=output_path,
                     detrend_vectors=args.detrend_vectors,
                     zscale_vectors=args.zscale_vectors)
    with Pool() as pool:
        results = pool.map(worker, files)

    with open(output_path / "crash_count.csv", 'w') as cc, \
         open(output_path / "errs.log", 'w') as err:
        for ursi, crash_count, error_path in results:
            if error_path:
                err.write(f"{error_path}\n")
            else:
                cc.write(f"{ursi},{crash_count}\n")

if __name__ == "__main__":
    main()
