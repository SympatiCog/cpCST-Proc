"""Synthetic cpCST recordings built from the plant identity.

    d(stim)/dt = GAIN * LAMBDA * e,   e = stim + user_raw

A crash is ~1.9 s of error sharing the stimulus's sign (runaway to BOUNDARY),
then RESET_GAP_S with no rows logged, then a reset to stim = 0 and 0.5 s of
one-signed error (settle). That is the corpus's crash shape.
"""
import numpy as np
import pandas as pd

FS = 30.0
DT = 1.0 / FS
LAMBDA = 0.15
GAIN = 3.0
RESET_GAP_S = 2.583
BOUNDARY = 0.5


def make_recording(duration_s=60.0, crash_times=(), seed=0, t0=3.5):
    """Raw-file frame: user_pos in the raw (unflipped) sign, ending with the
    duplicated last flip_time every corpus file carries."""
    rng = np.random.default_rng(seed)
    n_target = int(round(duration_s * FS))
    pending = sorted(crash_times)
    rec = []
    state = {"t": t0, "stim": 0.0, "cc": 0, "did_crash": False}

    def push(e):
        rec.append((state["t"], state["stim"], e - state["stim"],
                    state["cc"], state["did_crash"]))
        state["did_crash"] = False
        state["stim"] += DT * GAIN * LAMBDA * e
        state["t"] += DT

    settle_left = 0
    while len(rec) < n_target:
        if pending and state["t"] - t0 >= pending[0]:
            pending.pop(0)
            sgn = 1.0 if state["stim"] >= 0 else -1.0
            while abs(state["stim"]) < BOUNDARY:
                push(sgn * 0.6)
            state["t"] += RESET_GAP_S - DT
            state["cc"] += 1
            state["stim"] = 0.0
            state["did_crash"] = True
            settle_left = 15
            continue
        if settle_left:
            settle_left -= 1
            push(0.05)
        else:
            push(rng.normal(0.0, 0.01) - 0.3 * state["stim"])

    df = pd.DataFrame(rec, columns=["flip_time", "stim_pos", "user_pos",
                                    "crash_count", "did_crash"])
    df["expected_time"] = df["flip_time"] - 0.015
    df["lambda_val"] = LAMBDA
    return pd.concat([df, df.iloc[[-1]]], ignore_index=True)


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    make_recording(60.0, crash_times=(20.0, 40.0)).to_csv(
        out / "synth_crash.csv", index=False)
