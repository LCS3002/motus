"""live_passes (plans/live_controller/08 b, "Passes"): a pass saved under <run>/live/passes/<id>/,
listed newest first, read back, deleted (its own files only)."""

import json

import numpy as np
import pytest

from groundtrack import live_passes as lp


@pytest.fixture()
def run(tmp_path):
    r = tmp_path / "runs" / "sq01" / "v1"
    (r / "live").mkdir(parents=True)
    (r / "live" / "scene.json").write_text("{}", encoding="utf-8")
    return r


def manifest(n=3, ended="2026-10-28T19:00:41", files=("01.bin",)):
    return {"site": "sq", "ended": ended, "complete": True, "clock_s": 1.0,
            "cameras": [{"name": f"cam {f}", "file": f, "points": n, "run": "x"} for f in files]}


def test_save_list_read_delete(run, tmp_path):
    a = lp.save_pass(run, manifest(), {"01.bin": bytes(48)}, root=tmp_path)
    b = lp.save_pass(run, manifest(ended="2026-10-28T19:05:00", files=("01.bin", "02.bin")),
                     {"01.bin": bytes(48), "02.bin": bytes(48)}, root=tmp_path)
    assert a["id"] != b["id"] and lp.ID_RE.match(a["id"]) and lp.ID_RE.match(b["id"])   # the same second: "-2"
    assert [p["id"] for p in lp.list_passes([run, run])] == [b["id"], a["id"]]           # newest first, once
    m = lp.read_pass(run, b["id"])
    assert m["format"] == lp.FORMAT and m["cameras"][1]["file"] == "02.bin" and m["run"] == "runs/sq01/v1"
    assert lp.pass_file(run, b["id"], "02.bin").read_bytes() == bytes(48)
    with pytest.raises(FileNotFoundError):
        lp.pass_file(run, b["id"], "03.bin")
    # saved again with fewer cameras: the one no longer in it goes
    lp.save_pass(run, manifest(), {"01.bin": bytes(range(48))}, pass_id=b["id"])
    assert sorted(p.name for p in (lp.passes_dir(run) / b["id"]).iterdir()) == ["01.bin", "pass.json"]
    # deleted: only its own files; another job's file keeps the folder
    (lp.passes_dir(run) / a["id"] / "poses_01.json").write_text("{}", encoding="utf-8")
    lp.delete_pass(run, a["id"])
    assert [p.name for p in (lp.passes_dir(run) / a["id"]).iterdir()] == ["poses_01.json"]
    assert [p["id"] for p in lp.list_passes([run])] == [b["id"]]
    lp.delete_pass(run, b["id"])
    assert not (lp.passes_dir(run) / b["id"]).exists()


def test_checks(run):
    with pytest.raises(ValueError):
        lp.save_pass(run, manifest(), {"01.bin": bytes(47)})                    # 16 bytes a point
    with pytest.raises(ValueError):
        lp.save_pass(run, manifest(files=("../x.bin",)), {"../x.bin": bytes(48)})
    with pytest.raises(ValueError):
        lp.save_pass(run, {"cameras": []}, {})
    with pytest.raises(ValueError):
        lp.read_pass(run, "../../etc")
    with pytest.raises(FileNotFoundError):
        lp.save_pass(run.parent / "nope", manifest(), {"01.bin": bytes(48)})    # no package
    d = lp.passes_dir(run) / "20260101-000000"
    d.mkdir(parents=True)
    (d / "pass.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")
    assert lp.list_passes([run]) == []                                           # not a pass
    with pytest.raises(ValueError):
        lp.read_pass(run, "20260101-000000")


# --------------------------------------------------------------------------- 05: the recording
def recording(n_clock=3, rows=2, events=('{"k": 0, "t": 0, "T": 0, "type": "pass_start"}',)):
    clock = np.zeros(n_clock, lp.CLOCK_DTYPE)
    clock["t"] = clock["T"] = np.arange(n_clock) / 30
    clock["rate"], clock["flags"] = 1, lp.CLOCK_PLAYING
    cam = np.zeros(rows, lp.CAM_DTYPE)
    cam["R_wc"] = cam["S"] = np.eye(3)
    cam["C"] = [1, 2, 3]
    cam["k"] = cam["frame"] = np.arange(rows)
    cam["flags"], cam["src"], cam["player"] = lp.PERFORMED, 2, 1
    meta = {"started": "2026-10-29T10:00:00", "seconds": 0.067, "clock": {"file": "clock.rec", "rows": n_clock},
            "events": {"file": "events.jsonl", "n": len(events)},
            "cameras": [{"name": "cam 01.bin", "file": "01.rec", "rows": rows, "run": "x", "players": [1]}]}
    files = {"clock.rec": clock.tobytes(), "01.rec": cam.tobytes(), "events.jsonl": "".join(e + "\n" for e in events).encode()}
    return meta, files, clock, cam


def test_recording_saved_read_back_deleted(run, tmp_path):
    meta, files, clock, cam = recording()
    m = {**manifest(), "recording": meta}
    s = lp.save_pass(run, m, {"01.bin": bytes(48), **files}, root=tmp_path)
    assert s["recorded"] is True and s["rec_s"] == 0.067 and s["players"] == [1]
    folder = lp.passes_dir(run) / s["id"]
    assert sorted(p.name for p in folder.iterdir()) == ["01.bin", "01.rec", "clock.rec", "events.jsonl", "pass.json"]
    pj = lp.read_pass(run, s["id"])
    assert pj["version"] == 2 and pj["recording"]["format"] == lp.REC_FORMAT and pj["recording"]["cameras"][0]["run"] == "x"
    assert pj["recording"]["camera_fields"][1] == ["C", "<f8", [3], 72]                  # readable without numpy
    r = lp.read_recording(run, s["id"])
    assert r["clock"].tobytes() == clock.tobytes() and r["cameras"]["cam 01.bin"].tobytes() == cam.tobytes()
    assert r["events"] == [{"k": 0, "t": 0, "T": 0, "type": "pass_start"}]
    assert lp.pass_file(run, s["id"], "01.rec").stat().st_size == 2 * lp.CAM_DTYPE.itemsize
    assert lp.pass_file(run, s["id"], "events.jsonl").is_file()
    # saved again without one (an old view): version 1, the recording's files go
    lp.save_pass(run, manifest(), {"01.bin": bytes(48)}, pass_id=s["id"])
    assert sorted(p.name for p in folder.iterdir()) == ["01.bin", "pass.json"]
    assert lp.read_pass(run, s["id"])["version"] == 1 and lp.read_recording(run, s["id"]) is None
    # checked: sizes, the events' count and lines, the file names
    bad = [({**meta, "clock": {"file": "clock.rec", "rows": 4}}, files),
           ({**meta, "events": {"file": "events.jsonl", "n": 2}}, files),
           (meta, {**files, "events.jsonl": b"[1]\n"}),
           ({**meta, "clock": {"file": "clock.bin", "rows": 3}}, {**files, "clock.bin": files["clock.rec"]}),
           ({**meta, "cameras": [{**meta["cameras"][0], "file": "../01.rec"}]}, {**files, "../01.rec": files["01.rec"]})]
    for rec, fl in bad:
        with pytest.raises(ValueError):
            lp.save_pass(run, {**manifest(), "recording": rec}, {"01.bin": bytes(48), **fl})
    newer = folder / "pass.json"
    newer.write_text(json.dumps({**pj, "version": 3}), encoding="utf-8")
    with pytest.raises(ValueError):
        lp.read_pass(run, s["id"])                                                 # a newer format
    newer.write_text(json.dumps(pj), encoding="utf-8")
    lp.delete_pass(run, s["id"])
    assert not folder.exists()


def test_smooth_flag_and_event_backwards_compatible(run, tmp_path):
    """Smooth (29 Oct (4)): flag 64 on a camera's rows, a "smooth" event; a pass without them (older)
    reads as before, Smooth off. The format's versions stay."""
    flags = (lp.CHANGED, lp.HELD, lp.LEVELLING, lp.TILTED, lp.PERFORMED, lp.JOINED, lp.SMOOTH)
    assert lp.SMOOTH == 64 and len(set(flags)) == 7 and all(f & (f - 1) == 0 for f in flags) and "smooth" in lp.EVENTS
    assert lp.VERSION == 2 and lp.REC_VERSION == 1
    old = lp.save_pass(run, {**manifest(), "recording": recording()[0]}, {"01.bin": bytes(48), **recording()[1]}, root=tmp_path)
    assert not (lp.read_recording(run, old["id"])["cameras"]["cam 01.bin"]["flags"] & lp.SMOOTH).any()   # off
    meta, files, _, cam = recording(events=('{"k": 0, "t": 0, "T": 0, "type": "pass_start"}',
                                            '{"k": 1, "t": 0.03, "T": 0.03, "type": "smooth", "camera": "cam 01.bin", "value": true}'))
    cam["flags"][1] |= lp.SMOOTH
    s = lp.save_pass(run, {**manifest(), "recording": meta}, {"01.bin": bytes(48), **files, "01.rec": cam.tobytes()}, root=tmp_path)
    r = lp.read_recording(run, s["id"])
    assert list(r["cameras"]["cam 01.bin"]["flags"] & lp.SMOOTH) == [0, 64] and r["events"][1]["value"] is True


def test_files_request_envelope():
    """live.js postFiles: [u32 header length][JSON header][files], no base64."""
    head = json.dumps({"action": "save", "files": [{"file": "a.bin", "bytes": 3}, {"file": "b.rec", "bytes": 0}]}).encode()
    body = len(head).to_bytes(4, "little") + head + b"xyz"
    h, f = lp.parse_files(body)
    assert h == {"action": "save"} and f == {"a.bin": b"xyz", "b.rec": b""}
    for b in (body + b"!", body[:-1], b"\x00", (10 ** 6).to_bytes(4, "little") + head):
        with pytest.raises(ValueError):
            lp.parse_files(b)
