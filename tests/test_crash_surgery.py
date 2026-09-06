import numpy as np
import pandas as pd
import pytest

import CrashSurgery as cs
from synth import make_recording, RESET_GAP_S


def load(crash_times=(20.0,)):
    df = make_recording(60.0, crash_times=crash_times)
    df["user_pos"] = df["user_pos"] * -1        # the pipeline's sign flip
    return df


def test_plant_fit_is_exact_on_synthetic():
    ann, _ = cs.prepare(load())
    assert cs.check_plant_fit(ann) > 0.999


def test_detects_one_event_with_sensible_geometry():
    ann, ev = cs.prepare(load())
    assert len(ev) == 1
    e = ev[0]
    t = ann["flip_time"].values
    assert e["reset"] == int(np.argmax(ann["crash_count"].values == 1))
    assert e["onset"] < e["reset"] < e["settle"]
    assert 1.5 < e["runaway_s"] < 2.5
    assert abs(e["reset_gap_s"] - RESET_GAP_S) < 1e-6
    assert 0.4 <= e["settle_s"] <= 2.0
    assert e["onset_time"] == t[e["onset"]]


def test_annotation_columns():
    ann, ev = cs.prepare(load())
    e = ev[0]
    valid = ann["is_valid"].values
    assert not valid[e["onset"]:e["settle"] + 1].any()
    assert valid[:e["onset"]].all() and valid[e["settle"] + 1:].all()
    phase = ann["crash_phase"].values
    assert (phase[e["onset"]:e["reset"]] == cs.PHASE_RUNAWAY).all()
    assert (phase[e["reset"]:e["settle"] + 1] == cs.PHASE_REACQUIRE).all()
    assert (ann["epoch"].values[:e["reset"]] == 0).all()
    assert (ann["epoch"].values[e["reset"]:] == 1).all()
    t = ann["flip_time"].values
    assert np.isnan(ann["time_since_crash"].values[:e["reset"]]).all()
    assert ann["time_since_crash"].values[e["reset"]] == 0.0
    assert ann["time_to_crash"].values[e["reset"]] == 0.0
    assert np.isnan(ann["time_to_crash"].values[e["reset"] + 1:]).all()
    assert np.isclose(ann["time_to_crash"].values[0], t[e["reset"]] - t[0])


def test_flip_time_is_untouched():
    df = load()
    ann, _ = cs.prepare(df)
    raw = df["flip_time"].values[:-1]           # prepare drops the duplicate
    assert len(ann) == len(raw)
    assert np.array_equal(ann["flip_time"].values, raw)


def test_crash_free_file_is_all_valid():
    ann, ev = cs.prepare(load(crash_times=()))
    assert ev == []
    assert ann["is_valid"].all()
    assert ann["time_since_crash"].isna().all()


def test_two_crashes_give_three_epochs():
    ann, ev = cs.prepare(load(crash_times=(20.0, 40.0)))
    assert len(ev) == 2
    segs = list(cs.epoch_segments(ann))
    assert [k for k, _ in segs] == [0, 1, 2]


def test_align_epochs_removed():
    assert not hasattr(cs, "align_epochs")


def test_plot_excision_returns_figure(tmp_path):
    ann, ev = cs.prepare(load(crash_times=(20.0, 40.0)))
    fig = cs.plot_excision(ann, ev)
    assert len(fig.axes) == 3                    # overview + one per crash
    fig.savefig(tmp_path / "x.png")
    assert (tmp_path / "x.png").stat().st_size > 0
