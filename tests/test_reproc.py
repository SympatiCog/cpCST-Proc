from pathlib import Path

import numpy as np
import pandas as pd

import reproc_cpCST as r
from synth import make_recording, FS

DATA = Path(__file__).parent / "data"
ANNOT = ["crash_phase", "is_valid", "epoch", "time_since_crash", "time_to_crash"]


def run(tmp_path, mode, **kw):
    out = tmp_path / mode
    out.mkdir()
    r.process_file(DATA / "synth_crash.csv", out, kw.pop("detrend", False),
                   kw.pop("zscale", False), crash_mode=mode, **kw)
    return out


def test_interp_matches_golden(tmp_path):
    out = run(tmp_path, "interp")
    got = pd.read_csv(out / "synth_crash_interp.csv")
    want = pd.read_csv(DATA / "golden_interp.csv")
    pd.testing.assert_frame_equal(got, want)
    assert (out / "synth_crash_interp_repaired.png").exists()


def test_surgery_keeps_rows_and_clock(tmp_path):
    out = run(tmp_path, "surgery")
    got = pd.read_csv(out / "synth_crash.csv")
    raw = pd.read_csv(DATA / "synth_crash.csv").iloc[:-1]     # dup dropped
    assert len(got) == len(raw)
    assert np.array_equal(got["flip_time"].values, raw["flip_time"].values)
    assert np.array_equal(got["user_pos"].values, raw["user_pos"].values)
    for c in ANNOT:
        assert c in got.columns
    assert "was_repaired" not in got.columns
    assert got["is_valid"].dtype == bool
    assert (~got["is_valid"]).sum() > 0


def test_surgery_velocity_nan_across_reset_gap(tmp_path):
    out = run(tmp_path, "surgery")
    got = pd.read_csv(out / "synth_crash.csv")
    gap = np.r_[False, np.diff(got["flip_time"].values) > 2.0 / FS]
    assert gap.sum() == 2
    for c in ("user_pos_vel", "stim_pos_vel", "tracking_vel"):
        assert got[c][gap].isna().all()
        assert got[c][~gap].notna().all()


def test_surgery_zscore_fits_on_valid_only(tmp_path):
    out = run(tmp_path, "surgery", zscale=True)
    got = pd.read_csv(out / "synth_crash_zscale.csv")
    v = got["is_valid"].values
    z = got["tracking"].values
    assert abs(z[v].mean()) < 1e-9
    assert abs(z[v].std(ddof=1) - 1.0) < 1e-9
    assert np.abs(z[~v]).max() > 3            # the excursion is off-scale by design


def test_surgery_detrend_fits_on_valid_only(tmp_path):
    out = run(tmp_path, "surgery", detrend=True)
    got = pd.read_csv(out / "synth_crash_detrend.csv")
    v = got["is_valid"].values
    y = got["stim_pos"].values
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x[v], y[v], 1)[0]
    assert abs(slope) < 1e-9


def test_surgery_writes_events_and_plot(tmp_path):
    out = run(tmp_path, "surgery")
    ev = pd.read_csv(out / r.EVENTS_FILE)
    assert list(ev.columns) == list(r.EVENT_COLS)
    assert len(ev) == 2 and list(ev["k"]) == [1, 2]
    assert (ev["file"] == "synth_crash.csv").all()
    assert (out / "synth_crash_excised.png").exists()


def test_events_file_is_fresh_per_run(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / r.EVENTS_FILE).write_text("stale\n")
    r.reset_events(out)
    r.process_file(DATA / "synth_crash.csv", out, False, False)
    ev = pd.read_csv(out / r.EVENTS_FILE)
    assert len(ev) == 2


def test_surgery_max_samples_counts_invalid_rows(tmp_path):
    out = run(tmp_path, "surgery", max_samples=1000)
    got = pd.read_csv(out / "synth_crash_trim1000samp.csv")
    assert len(got) == 1000
    assert "is_valid" in got.columns


def test_crash_free_surgery_has_all_valid(tmp_path):
    src = tmp_path / "clean.csv"
    make_recording(40.0).to_csv(src, index=False)
    out = tmp_path / "o"
    out.mkdir()
    r.process_file(src, out, False, False)
    got = pd.read_csv(out / "clean.csv")
    assert got["is_valid"].all()
    assert not (out / "clean_excised.png").exists()
    assert not (out / r.EVENTS_FILE).exists() or len(pd.read_csv(out / r.EVENTS_FILE)) == 0
