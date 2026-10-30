"""live_pass_export (plans/live_controller/05, "Export"): a saved pass as a run - what the machine
drew, in BNG metres, with the files `groundtrack rhino` and `groundtrack houdini-pop` read; written
inside the pass's own folder only."""

import json

import numpy as np
import pandas as pd
import pytest

from groundtrack import live_passes as lp
from groundtrack.homography import Homography
from groundtrack.live_export import TRACK_DTYPE
from groundtrack.live_pass_export import export_dir, export_pass


def package(run, E0, frames=20):
    """Four tracks: 1 and 2 people walking, 3 a person standing, 4 a car (class 2)."""
    rows = []
    for f in range(frames):
        rows += [(1, f, 0.3 * f, 10.0, 0), (2, f, 2.0, 15.0 - 0.2 * f, 0), (3, f, -7.0, 15.0, 0),
                 (4, f, 7.0 + 1.5 * f, 25.0, 2)]
    a = np.zeros(len(rows), TRACK_DTYPE)
    for i, (tid, f, x, y, cls) in enumerate(sorted(rows, key=lambda r: (r[1], r[0]))):
        a[i] = (tid, f, f / 30.0, x, y, 0.5, 1.0, cls, 0)
    live = run / "live"
    live.mkdir(parents=True)
    (live / "tracks.bin").write_bytes(a.tobytes())
    (live / "tracks_index.json").write_text(json.dumps(
        {str(t): {"class": "car" if t == 4 else "person", "group": "vehicles" if t == 4 else "people"} for t in (1, 2, 3, 4)}),
        encoding="utf-8")
    (live / "scene.json").write_text(json.dumps({
        "version": 1, "origin": {"E": E0, "N": 2000.0, "datum_z": 5.0}, "fps": 30.0, "frames": frames,
        "tracks": {"file": "tracks.bin", "n_points": len(a), "classes": {"0": "person", "2": "car"}}}), encoding="utf-8")
    (run / "_working").mkdir()
    h = Homography(H=np.array([[0.01, 0, -5], [0, 0.01, -3], [0, 0.0005, 1]]), origin=(E0, 2000.0))
    h.camera_params = {"E": E0 + 1, "N": 1990.0, "height_m": 12.0}
    h.save(run / "_working" / "homography_used.json")
    return a


def drawing(a, upto=14, drop=(), predicted=(), part1=()):
    """What the machine drew: up to frame upto; (track, frame) dropped / predicted / a second piece."""
    n = len(a)
    X = np.full((n, 3), np.nan, np.float32)
    kind = np.full(n, 255, np.uint8)
    part = np.zeros(n, np.uint16)
    on = a["frame"] <= upto
    X[on] = np.column_stack([a["x"], a["y"], a["z"]])[on]
    kind[on] = 0
    for i, (t, f) in enumerate(zip(a["track_id"], a["frame"])):
        if (t, f) in drop:
            kind[i] = 4
            X[i] = np.nan
        elif (t, f) in predicted:
            kind[i] = 1
        if t in part1 and f >= 10:
            part[i] = 1
    cause = (kind == 1).astype(np.uint8) * 2
    return X.tobytes() + part.tobytes() + kind.tobytes() + cause.tobytes(), int((kind < 4).sum())


@pytest.fixture()
def saved(tmp_path):
    root = tmp_path / "proj"
    (root / "sites").mkdir(parents=True)
    (root / "sites" / "demo.yaml").write_text("site: demo\n", encoding="utf-8")
    r1, r2 = root / "runs" / "demo" / "v1", root / "runs" / "demo2" / "v1"
    a1, a2 = package(r1, 1000.0), package(r2, 1010.0)
    b1, n1 = drawing(a1, drop={(3, 5)}, predicted={(2, f) for f in (6, 7, 8)}, part1={4})
    b2, n2 = drawing(a2, upto=9)
    clock = np.zeros(15, lp.CLOCK_DTYPE)
    clock["t"] = clock["T"] = np.arange(15) / 30
    cam = np.zeros(3, lp.CAM_DTYPE)
    cam["R_wc"] = np.eye(3)
    cam["C"] = [[1, -10, 12], [1, -10, 12], [1, -9, 12]]
    cam["k"], cam["frame"], cam["flags"], cam["src"], cam["player"] = [0, 5, 9], [0, 5, 9], lp.PERFORMED, 2, 1
    cams = [{"name": "cam 1", "id": "1", "file": "1.bin", "points": len(a1), "run": str(r1)},
            {"name": "cam 2", "id": "2", "file": "2.bin", "points": len(a2), "run": str(r2)}]
    rec = {"started": "2026-10-29T10:00:00", "seconds": 0.47, "clock": {"file": "clock.rec", "rows": 15},
           "events": {"file": "events.jsonl", "n": 0}, "cameras": [{"name": "cam 1", "file": "1.rec", "rows": 3, "run": str(r1)}]}
    s = lp.save_pass(r1, {"site": "demo", "complete": True, "cameras": cams, "recording": rec},
                     {"1.bin": b1, "2.bin": b2, "clock.rec": clock.tobytes(), "1.rec": cam.tobytes(), "events.jsonl": b""},
                     root=root)
    return root, r1, s["id"], (a1, a2), (n1, n2)


def tree(root):
    return {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_a_pass_exported_as_a_run(saved):
    root, run, pid, _, (n1, n2) = saved
    before = tree(root)
    res = export_pass(run, pid, rhino=True, houdini=True, build=False, log=lambda *a: None)
    out = export_dir(run, pid)
    assert out == run / "live" / "passes" / pid / f"pass_{pid.replace('-', '_')}" and res["dir"] == str(out)
    after = tree(root)
    assert {k for k in after if after[k] != before.get(k)} <= {k for k in after if k.startswith(out.relative_to(root).as_posix())}
    p = pd.read_csv(out / "data" / "points.csv")
    run_cols = ["track_id", "class", "frame", "time_s", "x", "y", "vx", "vy", "speed", "heading", "predicted", "source",
                "group", "beyond_limit", "z"]
    assert list(p.columns[:15]) == run_cols and {"camera", "source_track", "part", "kind", "cause"} <= set(p.columns)
    assert len(p) == n1 + n2 == res["points"] and p["frame"].max() == 14                 # drawn only
    one = p[(p["camera"] == "cam 1") & (p["source_track"] == 1)].sort_values("frame")
    assert np.allclose(one["x"], 1000 + 0.3 * one["frame"], atol=1e-3) and np.allclose(one["y"], 2010) and np.allclose(one["z"], 5.5)
    assert np.allclose(one["vx"].iloc[4:-4], 9.0, atol=0.02) and np.allclose(one["heading"], 90)
    two = p[p["camera"] == "cam 2"]
    assert two["frame"].max() == 9 and np.allclose(two[two["source_track"] == 1]["x"], 1010 + 0.3 * two[two["source_track"] == 1]["frame"])
    assert not ((p["camera"] == "cam 1") & (p["source_track"] == 3) & (p["frame"] == 5)).any()           # dropped
    pr = p[p["predicted"]]
    assert set(pr["source"]) == {"predicted"} and set(pr["cause"]) == {"live"} and len(pr) == 3
    assert set(p.loc[p["source_track"] == 4, "class"]) == {"car"} and set(p.loc[p["source_track"] == 4, "group"]) == {"vehicles"}
    # one track per camera, footage track and piece
    pieces = p.groupby("track_id")[["camera", "source_track", "part"]].nunique()
    assert (pieces == 1).all().all() and p["track_id"].nunique() == 4 + 1 + 4
    m = pd.read_csv(out / "data" / "track_metrics.csv")
    assert len(m) == 9 == res["tracks"] and set(m.columns) >= {"track_id", "camera", "length_m", "predicted_share"}
    gj = json.loads((out / "data" / "tracks.geojson").read_text(encoding="utf-8"))
    assert gj["crs"]["properties"]["name"].endswith("27700") and len(gj["features"]) == 7                 # 3 stands still (both cameras)
    assert gj["features"][0]["geometry"]["coordinates"][0][0] > 999
    gaps = json.loads((out / "data" / "predicted_gaps.geojson").read_text(encoding="utf-8"))
    assert len(gaps["features"]) == 1 and gaps["features"][0]["properties"]["kind"] == "predicted"
    cp = pd.read_csv(out / "data" / "camera_path.csv")
    assert len(cp) == 3 and list(cp["E"]) == [1001, 1001, 1001] and list(cp["N"]) == [1990, 1990, 1991]
    assert list(cp["Z"]) == [17, 17, 17] and list(cp["source"]) == ["mouse"] * 3 and cp["performed"].all()
    # Houdini: curves + field (the folder serves as --composite and as a clip), the POP kit
    hou = out / "houdini"
    for f in ("points.csv", "houdini_import.py", "houdini_field.py", "vector_field.csv", "vector_field_master.csv",
              "vector_field_people.csv", "vector_field_vehicles.csv"):
        assert (hou / f).is_file(), f
    assert (hou / "points.csv").as_posix() in (hou / "houdini_import.py").read_text(encoding="utf-8")
    kit = json.loads((out / "houdini_pop" / "modules.json").read_text(encoding="utf-8"))
    names = [mm["name"] for mm in kit["modules"]]
    assert names == [f"demo_pass_{pid.replace('-', '_')}_whole", out.name] and kit["origin"] == [1000.0, 2000.0]
    # Rhino: the model with every camera, and `groundtrack rhino` takes the folder as a run
    assert (out / "rhino" / "grasshopper_import.py").is_file()
    try:
        import rhino3dm
    except ImportError:
        return
    model = out / "rhino" / "demo.3dm"
    assert model.is_file() and res["rhino"]["3dm"] == str(model)
    layers = [la.Name for la in rhino3dm.File3dm.Read(str(model)).Layers]
    assert "field_people" in layers and "field_master" not in layers and "tracks_vehicles" in layers
    assert not any("cam " in la for la in layers)                     # the per-pass model: no layer per camera
    from groundtrack.cli import main

    t0 = model.stat().st_mtime_ns
    main(["rhino", "-c", str(root / "sites" / "demo.yaml"), "--run", str(out)])
    assert model.stat().st_mtime_ns >= t0 and model.stat().st_size > 1000
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["pass"] == pid and [c["drawn"] for c in meta["cameras"]] == [n1, n2]
    # deleting the pass keeps its export (it may be open in Rhino / Houdini)
    lp.delete_pass(run, pid)
    assert out.is_dir() and not (out.parent / "pass.json").exists()


def test_cli_export_files_only(saved, capsys):
    from groundtrack.cli import main

    _, run, pid, _, (n1, n2) = saved
    main(["live-pass-export", str(run), pid])
    out = export_dir(run, pid)
    assert len(pd.read_csv(out / "data" / "points.csv")) == n1 + n2
    assert not list((out / "rhino").glob("*.3dm")) and not (out / "houdini_pop").exists()
    assert f"pass export -> {out}" in capsys.readouterr().out


def test_collage_layers_exported_as_a_run(saved, capsys):
    """05, 29 Oct (3): the collage (08 sandbox: each camera a layer, built from one or more plays) and
    one camera's layer, as the view shows them (a layer can be part of a pass), into
    <run of the first layer>/live/exports/<stamp>-<collage | camera id>/ with layers.json."""
    from groundtrack.cli import main
    from groundtrack.live_pass_export import export_layers, prepare_layers

    root, r1, pid, (a1, a2), _ = saved
    r2 = root / "runs" / "demo2" / "v1"
    b1, m1 = drawing(a1, upto=6)                              # part of the pass's cam 1
    b2, m2 = drawing(a1, upto=14, part1={1})                  # a later play stacked on it, not saved yet
    b3, m3 = drawing(a2, upto=9)
    layers = [{"camera": "cam 1", "camera_id": "1", "package": r1, "run": r1, "id": pid, "key": 1, "kind": "play",
               "stacked": False, "up_to": 6, "file": "a.bin"},
              {"camera": "cam 1", "camera_id": "1", "package": r1, "run": r1, "id": None, "key": 3, "kind": "play",
               "stacked": True, "up_to": 14, "file": "b.bin"},
              {"camera": "cam 2", "camera_id": "2", "package": r2, "run": r1, "id": pid, "key": 1, "kind": "play",
               "stacked": False, "up_to": 9, "file": "c.bin"}]
    files = {"a.bin": b1, "b.bin": b2, "c.bin": b3}
    before = tree(root)
    out = prepare_layers(layers, files, site="demo", root=root)
    assert out.parent == r1 / "live" / "exports" and out.name.endswith("-collage")
    res = export_layers(out, rhino=True, houdini=True, build=False, log=lambda *a: None)
    after = tree(root)
    assert {k for k in after if after[k] != before.get(k)} <= {k for k in after if k.startswith(out.relative_to(root).as_posix())}
    meta = json.loads((out / "layers.json").read_text(encoding="utf-8"))
    assert [(ly["layer"], ly["camera"], ly["id"], ly["package"], ly["drawn"]) for ly in meta["layers"]] == [
        (1, "cam 1", pid, "runs/demo/v1", m1), (2, "cam 1", None, "runs/demo/v1", m2), (3, "cam 2", pid, "runs/demo2/v1", m3)]
    assert meta["kind"] == "collage" and meta["points"] == m1 + m2 + m3 == res["points"]
    assert (out / meta["layers"][0]["file"]).read_bytes() == b1
    p = pd.read_csv(out / "data" / "points.csv", keep_default_na=False)
    assert list(p.columns[-2:]) == ["layer", "pass"] and len(p) == m1 + m2 + m3
    assert p[p["layer"] == 1]["frame"].max() == 6 and p[p["layer"] == 2]["frame"].max() == 14           # as the view showed it
    assert set(p[p["layer"] == 2]["pass"]) == {""} and set(p[p["layer"] != 2]["pass"]) == {pid}
    pieces = p.groupby("track_id")[["layer", "source_track", "part"]].nunique()
    assert (pieces == 1).all().all() and p["track_id"].nunique() == 4 + 5 + 4                        # per layer, track, piece
    one = p[(p["layer"] == 3) & (p["source_track"] == 1)]
    assert np.allclose(one["x"], 1010 + 0.3 * one["frame"], atol=1e-3)                                 # BNG, its own package
    cp = pd.read_csv(out / "data" / "camera_path.csv")
    assert list(cp["layer"]) == [1, 1, 1] and set(cp["camera"]) == {"cam 1"}       # cam 2 was not recorded; layer 2 not saved
    gj = json.loads((out / "data" / "tracks.geojson").read_text(encoding="utf-8"))
    assert gj["crs"]["properties"]["name"].endswith("27700") and gj["features"]
    assert (out / "houdini" / "vector_field_master.csv").is_file() and (out / "houdini_pop" / "modules.json").is_file()
    assert (out / "rhino" / "grasshopper_import.py").is_file() and meta["files"]["houdini_kit"]
    rhino3dm = pytest.importorskip("rhino3dm")
    model = rhino3dm.File3dm.Read(str(out / "rhino" / "demo.3dm"))
    lay = {la.Id: la for la in model.Layers}
    path = lambda la: (path(lay[la.ParentLayerId]) + " > " if la.ParentLayerId in lay else "") + la.Name
    names = {path(la) for la in model.Layers}
    assert {"groundtrack > 1 · cam 1", "groundtrack > 2 · cam 2"} <= names                      # a layer per camera
    assert {"groundtrack > 1 · cam 1 > people > tracks_people", "groundtrack > 2 · cam 2 > vehicles > tracks_vehicles",
            "groundtrack > field_people", "groundtrack > cameras"} <= names and "groundtrack > people" not in names
    on = lambda cam: {o.Attributes.GetUserString("track_id") for o in model.Objects                   
                      if path(model.Layers[o.Attributes.LayerIndex]).startswith(f"groundtrack > {cam}")} - {None, ""}
    assert sum(model.Layers[o.Attributes.LayerIndex].Name == "cameras" for o in model.Objects) == 2   # once each
    assert {int(t) for t in on("2 · cam 2")} == set(p.loc[p["camera"] == "cam 2", "track_id"])
    # one camera's layer: the camera filter, the folder named after the camera's id; the CLI
    out2 = prepare_layers(layers, files, camera="cam 2", root=root)
    assert out2.name.endswith("-2") and out2 != out
    main(["live-layers-export", str(out2)])
    assert set(pd.read_csv(out2 / "data" / "points.csv")["camera"]) == {"cam 2"}
    assert not list((out2 / "rhino").glob("*.3dm")) and f"layers export -> {out2}" in capsys.readouterr().out
    assert json.loads((out2 / "layers.json").read_text(encoding="utf-8"))["kind"] == "camera"
    with pytest.raises(ValueError, match="no layer"):
        prepare_layers(layers, files, camera="cam 9", root=root)
    with pytest.raises(ValueError, match="package"):
        prepare_layers(layers, {**files, "c.bin": b3[:-16]}, root=root)
    with pytest.raises(ValueError, match="not in pass"):
        prepare_layers([{**layers[0], "camera": "cam 9"}], files, root=root)
    lp.delete_pass(r1, pid)                                    # its passes deleted: the folder exports again
    export_layers(out2, log=lambda *a: None)
    assert len(pd.read_csv(out2 / "data" / "points.csv")) == m3 and pd.read_csv(out2 / "data" / "camera_path.csv").empty
