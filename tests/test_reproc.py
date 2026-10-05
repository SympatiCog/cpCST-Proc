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


def test_surgery_skip_on_max_samples_leaves_no_trace(tmp_path):
    out = run(tmp_path, "surgery", max_samples=100000)
    assert list(out.iterdir()) == []


def test_interp_run_keeps_surgery_event_table(tmp_path, monkeypatch):
    src = tmp_path / "raw"
    src.mkdir()
    (src / "synth_crash.csv").write_bytes((DATA / "synth_crash.csv").read_bytes())
    out = tmp_path / "o"
    monkeypatch.chdir(tmp_path)               # crash_count.csv lands here
    for mode in ("surgery", "interp"):
        monkeypatch.setattr("sys.argv", ["reproc_cpCST.py", "--base_path", str(src),
                                         "--output_path", str(out), "--crash_mode", mode])
        r.main()
    assert len(pd.read_csv(out / r.EVENTS_FILE)) == 2
    assert (out / "synth_crash.csv").exists() and (out / "synth_crash_interp.csv").exists()


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


def _corpus(tmp_path):
    """A small input folder covering every outcome a run can have."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "sub-A1_ses-1_task-CPT_events.csv").write_bytes(
        (DATA / "synth_crash.csv").read_bytes())
    make_recording(40.0).to_csv(src / "sub-B2_ses-1_task-CPT_events.csv", index=False)
    (src / "sub-C3_StimMarkers.csv").write_text(
        "StimMarkers_alpha,lsl_timestamp,ext_time,hh_mm_ss\nx,1.0,1.0,00:00:00\n")
    bad = pd.read_csv(DATA / "synth_crash.csv").astype({"flip_time": object})
    bad.loc[5, "flip_time"] = "garbage"
    bad.to_csv(src / "sub-D4_ses-1_task-CPT_events.csv", index=False)
    return src


def _run_main(tmp_path, monkeypatch, src, name, *extra):
    cwd = tmp_path / f"cwd_{name}"
    cwd.mkdir()
    out = tmp_path / f"out_{name}"
    monkeypatch.chdir(cwd)                    # crash_count.csv and errs.log land here
    monkeypatch.setattr("sys.argv", ["reproc_cpCST.py", "--base_path", str(src),
                                     "--output_path", str(out), *extra])
    r.main()
    return cwd, out


def test_process_one_writes_no_shared_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    res = r.process_one(DATA / "synth_crash.csv", out, False, False)
    assert res["status"] == "written"
    assert res["crash_count"] is not None
    assert len(res["events"]) == 2
    assert not (out / r.EVENTS_FILE).exists()
    assert not (tmp_path / "crash_count.csv").exists()
    assert (out / "synth_crash.csv").exists()


def test_process_one_reports_every_outcome(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = _corpus(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    status = {p.name.split("_")[0]: r.process_one(p, out, False, False)
              for p in sorted(src.glob("*.csv"))}
    assert status["sub-A1"]["status"] == "written"
    assert status["sub-B2"]["status"] == "written" and status["sub-B2"]["events"] == []
    assert status["sub-C3"]["status"] == "skipped"
    assert "not an events file" in status["sub-C3"]["message"]
    assert status["sub-D4"]["status"] == "error"
    assert "Traceback" in status["sub-D4"]["message"]
    short = r.process_one(src / "sub-A1_ses-1_task-CPT_events.csv", out, False, False,
                          max_samples=100000)
    assert short["status"] == "skipped" and short["events"] == []


def test_parallel_run_matches_serial(tmp_path, monkeypatch):
    src = _corpus(tmp_path)
    cwd1, out1 = _run_main(tmp_path, monkeypatch, src, "serial", "--jobs", "1")
    cwd4, out4 = _run_main(tmp_path, monkeypatch, src, "pool", "--jobs", "4")
    csvs = sorted(p.name for p in out1.glob("*.csv"))
    assert csvs == sorted(p.name for p in out4.glob("*.csv"))
    assert r.EVENTS_FILE in csvs and len(csvs) == 3     # two outputs + event table
    for name in csvs:
        assert (out1 / name).read_bytes() == (out4 / name).read_bytes(), name
    assert (cwd1 / "crash_count.csv").read_text() == (cwd4 / "crash_count.csv").read_text()
    assert (cwd1 / "crash_count.csv").read_text().count("\n") == 2


def test_one_bad_file_does_not_stop_the_run(tmp_path, monkeypatch):
    src = _corpus(tmp_path)
    cwd, out = _run_main(tmp_path, monkeypatch, src, "pool", "--jobs", "2")
    log = (cwd / "errs.log").read_text()
    assert "sub-D4" in log and "Traceback" in log
    assert "sub-A1" not in log
    assert (out / "sub-A1_ses-1_task-CPT_events.csv").exists()
    assert (out / "sub-B2_ses-1_task-CPT_events.csv").exists()
    ev = pd.read_csv(out / r.EVENTS_FILE)
    assert set(ev["file"]) == {"sub-A1_ses-1_task-CPT_events.csv"}


def test_jobs_must_be_positive(tmp_path, monkeypatch):
    import pytest
    src = _corpus(tmp_path)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, src, "zero", "--jobs", "0")
