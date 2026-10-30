"""live_sheet (plans/live_controller/09): the performance sheet of a saved pass - the finished drawing
over the score of how it was played - built from a small synthetic recording: 2 players on 2
cameras, known poses, every button, a pause and a scrub."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from groundtrack import live_passes as lp
from groundtrack import live_sheet as ls
from groundtrack.live_export import TRACK_DTYPE

FPS = 10.0
N_ROWS = 120                           # 12 s of performance, a clock row every 0.1 s


# ----------------------------------------------------------------------------- the phone's pose
def head_matrix(pan, tilt, roll):
    """live/pose.js headMatrix (degrees): Rz(-pan) Rx(tilt) Ry(roll) B."""
    p, t, r = np.radians([pan, tilt, roll])
    B = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    Rz = np.array([[np.cos(p), np.sin(p), 0], [-np.sin(p), np.cos(p), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    return Rz @ Rx @ Ry @ B


def mat_quat(R):
    """live/pose.js matToQuat: [w, x, y, z]."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def wave(player, t):
    """Each player's pan, tilt, roll (degrees) at performance time t."""
    if player == 1:
        return 30 * np.sin(t / 2) + 5, 6 * np.sin(t / 3), 4 * np.sin(t / 1.7)
    return -20 * np.sin(t / 2.5) + 170, 3 * np.cos(t / 2) - 3, -5 * np.sin(t / 1.3)


# ----------------------------------------------------------------------------- the synthetic pass
def package(run, E0, frames=40):
    """Two people walking (one faster), a terrain sloping north, a camera looking north."""
    rows = []
    for f in range(frames):
        rows += [(1, f, -5 + 0.15 * f, 10.0 + 0.1 * f), (2, f, 4.0, 30.0 - 0.6 * f)]
    a = np.zeros(len(rows), TRACK_DTYPE)
    for i, (tid, f, x, y) in enumerate(rows):
        a[i] = (tid, f, f / FPS, x, y, 0.0, 1.0, 0, 0)
    live = run / "live"
    live.mkdir(parents=True)
    (live / "tracks.bin").write_bytes(a.tobytes())
    (live / "tracks_index.json").write_text(json.dumps({"1": {"group": "people"}, "2": {"group": "people"}}),
                                            encoding="utf-8")
    n = 12
    z = np.tile(np.linspace(1.5, 0.0, n)[:, None], (1, n)).astype("<f4")        # row 0 = north (high)
    z.tofile(live / "terrain.f32")
    (live / "scene.json").write_text(json.dumps({
        "version": 1, "origin": {"E": E0, "N": 2000.0, "datum_z": 5.0}, "fps": FPS, "frames": frames,
        "camera": {"C": [0.0, 0.0, 3.0], "yaw_deg": 0.0, "hfov_deg": 70.0}, "R_max": 40.0,
        "terrain": {"file": "terrain.f32", "nx": n, "ny": n, "x0": -30.0, "y1": 40.0, "cell": 5.0},
        "tracks": {"file": "tracks.bin", "n_points": len(a), "classes": {"0": "person"}}}), encoding="utf-8")
    return a


def drawing(a, guessed=()):
    """Everything drawn, at the footage's positions; (track, frame) guessed."""
    n = len(a)
    X = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float32)
    kind = np.array([1 if (t, f) in guessed else 0 for t, f in zip(a["track_id"], a["frame"])], np.uint8)
    part = np.zeros(n, np.uint16)
    return X.tobytes() + part.tobytes() + kind.tobytes() + (kind * 2).astype(np.uint8).tobytes()


EVENTS = [  # (row k, type, player, camera, value)
    (5, "play", None, None, None),
    (20, "space_press", 1, "cam A", "phone 1"), (35, "space_release", 1, "cam A", "phone 1"),
    (40, "level", 1, "cam A", None),
    (50, "pause", None, None, None), (62, "play", None, None, None),
    (70, "seek", None, None, 0.5),
    (75, "zero", 2, "cam B", "web"),
    (80, "gain", 2, "cam B", 2.0),
    (90, "space_press", 2, "cam B", "phone 2"),                       # held to the end
    (95, "gain", 1, "cam A", 0.5),
]


def clock_rows():
    """T: plays from k 5, pauses k 50-62, a scrub back to 0.5 s at k 70."""
    c = np.zeros(N_ROWS, lp.CLOCK_DTYPE)
    c["t"] = np.arange(N_ROWS) * 0.1
    T, playing = 0.0, False
    for k in range(N_ROWS):
        if k == 5 or k == 62:
            playing = True
        if k == 50:
            playing = False
        if k == 70:
            T = 0.5
        elif playing and k > 0:
            T += 0.1
        c["T"][k] = T
        c["flags"][k] = int(playing)
    c["rate"] = 1
    return c


def cam_rows(player, clock):
    """One row a clock row: the player's pose, a camera turning, the machine's counts, the frame."""
    r = np.zeros(N_ROWS, lp.CAM_DTYPE)
    gain = 1.0
    for k in range(N_ROWS):
        t = clock["t"][k]
        for kk, typ, pl, _, val in EVENTS:
            if typ == "gain" and pl == player and kk == k:
                gain = val
        pan, tilt, roll = wave(player, t)
        R = head_matrix(pan * 0.5, tilt, roll)
        r["R_wc"][k] = R
        r["C"][k] = [0, 0, 3]
        r["S"][k] = np.eye(3)
        r["k"][k] = k
        r["frame"][k] = min(int(np.rint(clock["T"][k] * FPS)), 39)
        r["q"][k] = mat_quat(head_matrix(pan, tilt, roll))
        r["p"][k] = np.nan
        r["gain"][k], r["scale"][k] = gain, 1
        r["placed"][k], r["guessed"][k], r["lost"][k], r["dropped"][k] = (k % 7) + player, k % 3, (k // 10) % 4, 0
        r["new"][k] = 1 if k % 9 == 0 else 0
        r["player"][k], r["src"][k], r["flags"][k], r["seq"][k] = player, 1, lp.PERFORMED, k + 1
    held = (r["k"] >= 20) & (r["k"] < 35) if player == 1 else r["k"] >= 90
    r["flags"][held] |= lp.HELD
    return r


@pytest.fixture()
def saved(tmp_path):
    root = tmp_path / "proj"
    (root / "sites").mkdir(parents=True)
    rA, rB = root / "runs" / "demoA" / "v1", root / "runs" / "demoB" / "v1"
    aA, aB = package(rA, 1000.0), package(rB, 1040.0)
    clock = clock_rows()
    recA, recB = cam_rows(1, clock), cam_rows(2, clock)
    ev = [{"k": k, "t": round(float(clock["t"][k]), 4), "T": float(clock["T"][k]), "type": typ, "player": pl,
           "camera": cam, "value": val} for k, typ, pl, cam, val in EVENTS]
    evtext = "".join(json.dumps(e) + "\n" for e in ev).encode()
    cams = [{"name": "cam A", "id": "A", "file": "A.bin", "points": len(aA), "run": str(rA), "tint": "#9b82d9",
             "origin": {"E": 1000.0, "N": 2000.0, "datum_z": 5.0}, "players": [1]},
            {"name": "cam B", "id": "B", "file": "B.bin", "points": len(aB), "run": str(rB), "tint": "#8b8fe6",
             "origin": {"E": 1040.0, "N": 2000.0, "datum_z": 5.0}, "players": [2]}]
    rec = {"started": "2026-10-29T10:00:00", "seconds": 11.9, "clock": {"file": "clock.rec", "rows": N_ROWS},
           "events": {"file": "events.jsonl", "n": len(ev)},
           "cameras": [{"name": "cam A", "file": "A.rec", "rows": N_ROWS, "run": str(rA), "tint": "#9b82d9"},
                       {"name": "cam B", "file": "B.rec", "rows": N_ROWS, "run": str(rB), "tint": "#8b8fe6"}]}
    s = lp.save_pass(rA, {"site": "demo", "complete": True, "started": "2026-10-29T10:00:00", "duration_s": 3.9,
                          "origin": {"E": 1000.0, "N": 2000.0, "datum_z": 5.0}, "cameras": cams, "recording": rec},
                     {"A.bin": drawing(aA, guessed={(2, f) for f in range(10, 16)}), "B.bin": drawing(aB),
                      "clock.rec": clock.tobytes(), "A.rec": recA.tobytes(), "B.rec": recB.tobytes(),
                      "events.jsonl": evtext}, root=root)
    return {"root": root, "run": rA, "id": s["id"], "clock": clock, "rows": {1: recA, 2: recB}, "events": ev}


@pytest.fixture()
def quick(monkeypatch):
    monkeypatch.setattr(ls, "DPI", 40)            # the same sheet, fewer pixels


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("pan,tilt,roll", [(0, 0, 0), (35, -8, 4), (-120, 20, -15), (179, 3, 30), (10, -60, 0)])
def test_pose_angles_read_the_head_back(pan, tilt, roll):
    hd, tl, rl = ls.pose_angles(ls.quat_matrix(mat_quat(head_matrix(pan, tilt, roll))))
    assert ((hd[0] - pan + 180) % 360 - 180) == pytest.approx(0, abs=1e-6)
    assert tl[0] == pytest.approx(tilt, abs=1e-6)
    assert rl[0] == pytest.approx(roll, abs=1e-6)


def test_score_reads_the_recording(saved):
    rec = lp.read_recording(saved["run"], saved["id"])
    sc = ls.score_data(rec)
    assert sc["t_end"] == pytest.approx(11.9)
    assert [p["label"] for p in sc["players"]] == ["P1", "P2"]
    p1, p2 = sc["players"]
    assert p1["cameras"][0][0] == "cam A" and p2["cameras"][0][0] == "cam B"
    assert (p1["tint"], p2["tint"]) == ("#9b82d9", "#8b8fe6")
    # the waves are the poses sent: turn from the first heading (across ±180), tilt and roll as held
    t_clk = saved["clock"]["t"]
    for p in (p1, p2):
        first = wave(p["player"], 0.0)[0]
        for k in (0, 33, 70, 119):
            t = t_clk[k]
            pan, tilt, roll = wave(p["player"], t)
            assert ls.value_at(p["t"], p["turn"], t) == pytest.approx(pan - first, abs=1e-4)
            assert ls.value_at(p["t"], p["tilt"], t) == pytest.approx(tilt, abs=1e-4)
            assert ls.value_at(p["t"], p["roll"], t) == pytest.approx(roll, abs=1e-4)
    # every press at its time
    assert p1["held"] == [(2.0, 3.5)] and p2["held"] == [(9.0, 11.9)]
    assert p1["level"] == [4.0] and p1["zero"] == [] and p2["zero"] == [7.5]
    assert list(zip(*p1["gain"])) == [(0.0, 1.0), (9.5, 0.5)]
    assert list(zip(*p2["gain"])) == [(0.0, 1.0), (8.0, 2.0)]
    # the shared clock: play, pause, the scrub
    c = sc["clock"]
    assert c["plays"] == [0.5, 6.2] and c["pauses"] == [5.0] and c["seeks"] == [(7.0, 0.5)]
    assert c["T"][69] == pytest.approx(5.3) and c["T"][70] == pytest.approx(0.5)
    assert c["playing"][10] and not c["playing"][55]
    # the machine: the counts as recorded, new identities where the frame moved on
    for m, pl in zip(sc["machine"], (1, 2)):
        rows = saved["rows"][pl]
        for k in (0, 13, 64, 119):
            for key in ("placed", "guessed", "lost"):
                assert ls.value_at(m["t"], m[key], t_clk[k]) == rows[key][k]
        nt, nv = m["new"]
        moved = np.r_[True, np.diff(rows["frame"]) != 0]
        want = (rows["new"] > 0) & moved
        assert np.allclose(nt, t_clk[want]) and (nv == 1).all()
        assert ((rows["new"] > 0) & ~moved).any()          # repeated rows (paused, at the clip's end): left out


def test_sheet_files_black_and_light(saved):
    out = ls.build_sheet(saved["run"], saved["id"], themes=("light",))
    folder = lp._folder(saved["run"], saved["id"])
    png, pdf = folder / "sheet_light.png", folder / "sheet_light.pdf"
    assert out["files"]["light"] == {"pdf": str(pdf), "png": str(png)}
    w, h = struct.unpack(">II", png.read_bytes()[16:24])
    assert w == pytest.approx(420 / 25.4 * 300, abs=1) and abs(w - 2 * h) <= 1        # 2:1 at 300 dpi
    body = pdf.read_bytes()
    assert body.startswith(b"%PDF") and b"/Subtype /Image" not in body                  # vector, no view yet
    assert b"SplineSansMono" in body                                                    # the labels' font
    assert not list(folder.glob("*.part"))
    info = json.loads((folder / "sheet.json").read_text(encoding="utf-8"))
    assert info["recorded"] and info["layout"] == "C" and info["players"] == ["P1", "P2"]


def test_every_press_is_drawn_at_its_time(saved, quick):
    out = ls.build_sheet(saved["run"], saved["id"], themes=("black",))
    marks = out["info"]["marks"]
    ev = saved["events"]
    at = lambda typ, pl=None: [e["t"] for e in ev if e["type"] == typ and (pl is None or e["player"] == pl)]
    assert marks["play"] == at("play") and marks["pause"] == at("pause") and marks["seek"] == at("seek")
    for pl, lab in ((1, "P1"), (2, "P2")):
        pm = marks["players"][lab]
        assert [a for a, _ in pm["space"]] == at("space_press", pl)
        assert [b for _, b in pm["space"]] == at("space_release", pl) + ([11.9] if pl == 2 else [])
        assert pm["level"] == at("level", pl) and pm["zero"] == at("zero", pl)
        assert [g[0] for g in pm["gain"]][1:] == at("gain", pl)
    assert marks["machine"]["cam A"]["placed"] == int(saved["rows"][1]["placed"].max())
    assert marks["machine"]["cam B"]["new"] == int(np.sum((saved["rows"][2]["new"] > 0)
                                                         & np.r_[True, np.diff(saved["rows"][2]["frame"]) != 0]))


def test_the_layouts(saved, quick):
    folder = lp._folder(saved["run"], saved["id"])
    for layout in ls.LAYOUTS:
        out = ls.build_sheet(saved["run"], saved["id"], themes=("black",), layout=layout, name=f"sheet_{layout}")
        assert (folder / f"sheet_{layout}_black.pdf").is_file() and out["info"]["layout"] == layout
    with pytest.raises(ValueError):
        ls.build_sheet(saved["run"], saved["id"], layout="D")


def test_the_drawing_in_sequence(saved):
    rec = lp.read_recording(saved["run"], saved["id"])
    dr = ls.drawing_data(saved["run"], saved["id"], rec)
    assert dr["points"] == 160 and dr["guessed"] == 6 and len(dr["pieces"]) == 4
    v = {c["name"]: c["virtual"] for c in dr["cameras"]}
    assert v["cam A"]["yaw"] == pytest.approx(wave(1, 11.9)[0] * 0.5, abs=1e-6)      # its last row
    assert dr["cameras"][1]["C"][0] == pytest.approx(40.0)                           # in the pass's frame
    tp = np.concatenate([p[3] for p in dr["pieces"]])
    assert np.isfinite(tp).all() and tp.max() <= 11.9 + 1e-9
    early = ls.until(dr["pieces"], 3.0)            # frames 0-4 only: the scrub back to 0.5 s drew the rest again
    assert 0 < sum(len(p[0]) for p in early) < dr["points"]
    assert all((p[3] <= 3.0).all() for p in early)


def test_a_pass_without_recording_gets_the_drawing_only(saved, quick):
    run = saved["run"]
    m = lp.read_pass(run, saved["id"])
    folder = lp._folder(run, saved["id"])
    old = lp.save_pass(run, {"site": "demo", "cameras": m["cameras"]},
                       {c["file"]: (folder / c["file"]).read_bytes() for c in m["cameras"]}, root=saved["root"])
    out = ls.build_sheet(run, old["id"], themes=("black",), layout="B")      # B without a recording: as A
    assert out["info"]["recorded"] is False and out["info"]["marks"]["players"] == {}
    assert (lp._folder(run, old["id"]) / "sheet_black.png").is_file()


def test_the_view_comes_through_the_route(saved, quick):
    from PIL import Image

    run, pid = saved["run"], saved["id"]
    folder = lp._folder(run, pid)
    img = Image.new("RGB", (300, 100), (0, 0, 0))                      # 3:1, the frame is 2:1: cropped
    img.paste((200, 120, 255), (140, 40, 160, 60))
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, "PNG")

    def body(head, files):
        h = json.dumps({**head, "files": [{"file": k, "bytes": len(v)} for k, v in files.items()]}).encode()
        return len(h).to_bytes(4, "little") + h + b"".join(files.values())

    seen = []
    resolve = lambda r: seen.append(r) or run
    r = ls.request(resolve, body({"action": "view", "run": "x", "id": pid, "build": False}, {"view.png": buf.getvalue()}))
    assert r == {"ok": True, "view": str(folder / ls.VIEW_FILE)} and seen == ["x"]
    assert not (folder / "sheet_black.pdf").exists()
    r = ls.request(resolve, body({"action": "view", "run": "x", "id": pid}, {"view.png": buf.getvalue()}))
    assert set(r["files"]) == {"light"} and b"/Subtype /Image" in (folder / "sheet_light.pdf").read_bytes()
    r = ls.request(resolve, body({"action": "view", "run": "x", "id": pid, "themes": ["black"]}, {"view.png": buf.getvalue()}))
    assert set(r["files"]) == {"black"}
    with pytest.raises(ValueError):
        ls.request(resolve, body({"action": "view", "run": "x", "id": pid, "themes": ["sepia"]}, {"view.png": buf.getvalue()}))
    with pytest.raises(ValueError):
        ls.request(resolve, body({"action": "view", "run": "x", "id": pid}, {"view.png": b"GIF89a"}))
    ls.request(resolve, body({"action": "error", "run": "x", "id": pid, "message": "no WebGL"}, {}))
    assert ls.ERRORS.pop((str(run.resolve()), pid)) == "no WebGL"
    with pytest.raises(ValueError):
        ls.request(resolve, body({"action": "view", "run": "x", "id": "../../etc"}, {}))


def test_the_light_view():
    """The view's lines on black laid on white: ramp as drawn, violet dark, white to mid grey."""
    cmap, norm = ls._ramp()
    fast = np.array(cmap(norm(15.9))[:3])                             # orange in the ramp
    px = np.zeros((1, 6, 3), np.uint8)
    px[0, 1] = np.round(fast * 255)                                   # a track, fully covered
    px[0, 2] = np.round(fast * 0.5 * 255)                             # its antialiased edge
    px[0, 3] = 217                                                    # a guess (white at 0.85)
    px[0, 4] = 41                                                     # a contour (0.16)
    px[0, 5] = (0x9b, 0x82, 0xd9)                                     # a virtual camera (the violet)
    out = ls.light_view(px).astype(float) / 255
    assert (out[0, 0] == 1).all()                                     # black -> the sheet's white
    assert np.allclose(out[0, 1], fast, atol=0.03)                     # the ramp as drawn
    assert np.allclose(out[0, 2], 1 - 0.5 * (1 - fast), atol=0.03)    # half covered: half on white
    assert 0.45 < out[0, 3].mean() < 0.6 and np.ptp(out[0, 3]) < 0.01  # mid grey
    assert 0.85 < out[0, 4].mean() < 0.95                             # light grey
    import matplotlib.colors as mc

    assert np.allclose(out[0, 5], mc.to_rgb(ls._dark("#9b82d9", True)), atol=0.02)


def test_view_size_and_browser_lookup(monkeypatch, tmp_path):
    assert ls.view_size("A") == (2032, 1016) and ls.view_size("C") == (3048, 1016)
    exe = tmp_path / "chrome.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("MOTUS_BROWSER", str(exe))
    assert ls.find_browser() == str(exe)


def test_cli_without_the_browser(saved, quick, capsys):
    from groundtrack.cli import main

    main(["live-sheet", str(saved["run"]), saved["id"], "--no-view", "--theme", "black", "--layout", "C"])
    folder = lp._folder(saved["run"], saved["id"])
    assert (folder / "sheet_black.pdf").is_file() and not (folder / "sheet_light.pdf").exists()
    assert "sheet_black.png" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the collage (08 sandbox)
@pytest.fixture()
def collage(saved):
    """A second play (cam A again, layer replaced): the collage is cam B from the first play and
    cam A from the second."""
    run = saved["run"]
    m = lp.read_pass(run, saved["id"])
    folder = lp._folder(run, saved["id"])
    data = {f.name: f.read_bytes() for f in folder.iterdir() if f.suffix in (".bin", ".rec", ".jsonl")}
    s2 = lp.save_pass(run, {k: m[k] for k in ("site", "complete", "started", "duration_s", "origin", "cameras",
                                              "recording")}, data, root=saved["root"])
    assert s2["id"] != saved["id"]
    layers = [{"run": run, "id": saved["id"], "camera": "cam B", "key": 1, "kind": "play", "stacked": False},
              {"run": run, "id": s2["id"], "camera": "cam A", "key": 2, "kind": "play", "stacked": False}]
    return {**saved, "id2": s2["id"], "layers": layers}


def test_the_collage_sheet(collage, quick):
    out = ls.build_collage(collage["layers"], themes=("light",), site="demo")
    d = collage["run"] / "live" / "sheets"
    folder = next(d.iterdir())
    assert out["dir"] == str(folder) and folder.name.endswith("-collage")
    assert (folder / "sheet_light.pdf").is_file() and (folder / "sheet_light.png").is_file()
    info = json.loads((folder / "layers.json").read_text(encoding="utf-8"))
    assert [(ly["camera"], ly["id"]) for ly in info["layers"]] == [("cam B", collage["id"]), ("cam A", collage["id2"])]
    assert [p["id"] for p in info["plays"]] == [collage["id"], collage["id2"]]
    gap = 2.0
    assert [p["t0"] for p in info["plays"]] == [0.0, round(11.9 + gap, 3)]
    assert info["points"] == 160 and info["guessed"] == 6 and info["players"] == ["P1", "P2"]
    pm = info["marks"]["players"]
    assert pm["P2"]["space"] == [[9.0, 11.9]]                       # held to the end of its play, not beyond
    assert pm["P1"]["space"] == [[round(2.0 + 13.9, 3), round(3.5 + 13.9, 3)]]
    assert pm["P2"]["zero"] == [7.5] and pm["P1"]["level"] == [round(4.0 + 13.9, 3)]
    assert info["marks"]["seek"] == [7.0, round(7.0 + 13.9, 3)]
    assert set(info["marks"]["machine"]) == {"cam A", "cam B"}


def test_the_camera_sheet_and_the_route(collage, quick):
    from io import BytesIO

    from PIL import Image

    out = ls.build_collage(collage["layers"], camera="cam A", themes=("light",))
    info = out["info"]
    assert out["dir"].endswith("-A") and info["kind"] == "camera" and [p["id"] for p in info["plays"]] == [collage["id2"]]
    assert info["players"] == ["P1"] and info["points"] == 80
    buf = BytesIO()
    Image.new("RGB", (60, 20)).save(buf, "PNG")

    def body(head, files):
        h = json.dumps({**head, "files": [{"file": k, "bytes": len(v)} for k, v in files.items()]}).encode()
        return len(h).to_bytes(4, "little") + h + b"".join(files.values())

    lay = [{**ly, "run": "r"} for ly in collage["layers"]]
    r = ls.request(lambda r: collage["run"], body({"action": "collage", "layers": lay, "camera": "cam B"},
                                                  {"view.png": buf.getvalue()}))
    folder = Path(r["dir"])
    assert folder.parent == collage["run"] / "live" / "sheets" and set(r["files"]) == {"light"}
    assert (folder / ls.VIEW_FILE).is_file() and b"/Subtype /Image" in (folder / "sheet_light.pdf").read_bytes()
    with pytest.raises(ValueError, match="not saved"):
        ls.request(lambda r: collage["run"], body({"action": "collage", "layers": [{**lay[0], "id": None}]}, {}))
    with pytest.raises(ValueError, match="no layer"):
        ls.build_collage(collage["layers"], camera="cam C")
