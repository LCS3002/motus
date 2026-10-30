"""live_site (plans/live_controller/08 b, changed 28 Oct): the cameras a Live view can show, the
site's scene packages, with readable names and, past 6 cameras, areas by place."""

import json
import os
from pathlib import Path

import pytest

from groundtrack import live_site as ls


# --------------------------------------------------------------------------- places (pure)
def test_compass_and_the_centre():
    assert ls.compass(0, 10)[0] == "North" and ls.compass(10, 0)[0] == "East"
    assert ls.compass(0, -10)[0] == "South" and ls.compass(-10, 0)[0] == "West"
    assert ls.compass(10, 10)[0] == "North-east" and ls.compass(-10, -10)[0] == "South-west"
    assert ls.compass(-10, 10) == ("North-west", pytest.approx(315))
    assert ls.compass(4, 10)[0] == "North" and ls.compass(5, 10)[0] == "North-east"   # 22.5 deg
    assert ls.site_centre([(0, 0), (10, 4), (4, 20)]) == (5, 10)                      # the box's middle
    names = ls.place_names([(0, 50), (1, 2), (50, 0)], (0, 0))
    assert [n for n, _ in names] == ["North", "Centre", "East"]                       # near the middle
    assert names[1][1] == -1                                                          # Centre first


def test_cluster_near_full_and_too_many():
    # two tight pairs 100 m apart and a loner: the pairs join, the loner stays
    pts = [(0, 0), (5, 0), (100, 0), (104, 3), (300, 300)]
    assert ls.cluster(pts, near=20) == [[0, 1], [2, 3], [4]]
    # complete linkage: a chain 15 m a step is not one area at near 20 (ends 30 m apart)
    assert ls.cluster([(0, 0), (15, 0), (30, 0)], near=20) == [[0, 1], [2]]
    # at most max_group a group
    eight = [(i * 0.5, 0) for i in range(8)]
    g = ls.cluster(eight, near=20, max_group=6)
    assert len(g) == 2 and all(len(x) <= 6 for x in g) and sorted(sum(g, [])) == list(range(8))
    # more than max_groups areas: the nearest keep merging even when far
    spread = [(i * 100.0, 0) for i in range(8)]
    g = ls.cluster(spread, near=20, max_groups=6)
    assert len(g) == 6 and g == ls.cluster(spread, near=20, max_groups=6)            # deterministic
    assert len(ls.cluster(spread, near=20, max_groups=6, reserved=2)) == 4             # fixed areas count
    # near scales with the site, within its bounds
    assert ls.near_m([(0, 0), (100, 0)]) == ls.NEAR_MIN_M
    assert ls.near_m([(0, 0), (300, 0)]) == pytest.approx(45)
    assert ls.near_m([(0, 0), (3000, 0)]) == ls.NEAR_MAX_M


def cams(points):
    return [{"id": str(6300 + k), "clip": f"sq{6300 + k}", "E": e, "N": n} for k, (e, n) in enumerate(points)]


def test_up_to_six_a_plain_list_named_by_direction():
    out, groups, notes = ls.arrange(cams([(0, 100), (100, 0), (0, -100), (-100, 0), (2, 101)]))
    assert groups == [] and notes == []
    assert [c["name"] for c in out] == ["North · 6300", "North · 6304", "East · 6301", "South · 6302",
                                        "West · 6303"]                               # clockwise, then number
    assert all(c["group"] is None for c in out)
    one, g1, _ = ls.arrange(cams([(5, 5)]))
    assert [c["name"] for c in one] == ["6300"] and g1 == []                         # alone: its number


def test_more_than_six_areas_by_place():
    pts = [(0, 200), (10, 205), (200, 0), (205, 8), (198, -6), (0, -200), (-200, 0), (-205, 10)]
    out, groups, _ = ls.arrange(cams(pts))
    assert [(g["name"], [s.split(" · ")[1] for s in g["spots"]]) for g in groups] == [
        ("North", ["6300", "6301"]), ("East", ["6302", "6303", "6304"]), ("South", ["6305"]),
        ("West", ["6306", "6307"])]
    assert all(c["name"] == f"{c['group']} · {c['id']}" for c in out)
    assert [c["id"] for c in out] == ["6300", "6301", "6302", "6303", "6304", "6305", "6306", "6307"]
    # the same direction twice: the nearer keeps the name
    pts2 = [(0, 100), (3, 104), (0, 300), (4, 296), (300, 0), (0, -300), (-300, 0)]
    _, g2, _ = ls.arrange(cams(pts2))
    assert [g["name"] for g in g2] == ["North", "North 2", "East", "South", "West"]
    assert g2[0]["spots"] == ["North · 6300", "North · 6301"] and g2[1]["spots"] == ["North 2 · 6302", "North 2 · 6303"]


def test_the_site_config_fixes_and_renames_areas():
    pts = [(0, 200), (10, 205), (200, 0), (205, 8), (198, -6), (0, -200), (-200, 0), (-205, 10)]
    live = {"groups": {"Gallery": "North", "Steps": ["6302", "sq6305"], "Odd": 5, "Ghost": ["9999"]}}
    out, groups, notes = ls.arrange(cams(pts), live)
    by = {g["name"]: [s.split(" · ")[1] for s in g["spots"]] for g in groups}
    assert by["Steps"] == ["6302", "6305"] and by["Gallery"] == ["6300", "6301"]
    assert "North" not in by and sorted(sum(by.values(), [])) == sorted(c["id"] for c in out)
    assert any("Odd" in n for n in notes) and any("9999" in n for n in notes)
    # up to six: no areas, but the fixed names name the cameras
    out6, g6, _ = ls.arrange(cams([(0, 100), (100, 0), (0, -100)]), {"groups": {"Fountain": [6301]}})
    assert g6 == [] and "Fountain · 6301" in [c["name"] for c in out6]
    _, _, bad = ls.arrange(cams([(0, 1), (1, 0)]), {"groups": ["not", "a", "mapping"]})
    assert bad == ["live.groups is not a mapping: ignored"]


# --------------------------------------------------------------------------- the site's packages
def package(run: Path, clip: str, E: float, N: float, C=(0.3, -0.2, 1.6)):
    (run / "live").mkdir(parents=True)
    meta = {"version": 1, "clip": clip, "run": run.name, "origin": {"E": E, "N": N, "datum_z": 10.0},
            "camera": {"C": list(C)}}
    (run / "live" / "scene.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def site(tmp_path):
    """A site "sq" (base sq.yaml) with clip configs sq01..sq04 (sq03 has no package, sq04 two runs)
    and another site "other" with one clip."""
    s = tmp_path / "sites"
    s.mkdir()
    (s / "sq.yaml").write_text("site: sq\noutput_dir: ../runs\nlive:\n  groups:\n    Terrace: North\n",
                               encoding="utf-8")
    for k in (1, 2, 3, 4):
        (s / f"sq0{k}.yaml").write_text(f"extends: sq.yaml\nsite: sq0{k}\n", encoding="utf-8")
    (s / "other.yaml").write_text("site: other\noutput_dir: ../runs\n", encoding="utf-8")
    runs = tmp_path / "runs"
    package(runs / "sq01" / "v1", "sq01", 1050.0, 2100.0)
    package(runs / "sq02" / "v1", "sq02", 1100.0, 2000.0)
    package(runs / "sq04" / "v2", "sq04", 900.0, 1900.0)
    package(runs / "sq04" / "v1", "sq04", 1050.0, 1900.0)
    old = (runs / "sq04" / "v2" / "live" / "scene.json").stat().st_mtime - 60
    os.utime(runs / "sq04" / "v2" / "live" / "scene.json", (old, old))   # v1 the newest (not by the clock's tick)
    package(runs / "other" / "v1", "other", 0.0, 0.0)
    (runs / "sq03" / "v1").mkdir(parents=True)                 # a run, no package
    return tmp_path, sorted(s.glob("*.yaml"))


def test_site_cameras_from_the_sites_packages(site):
    root, configs = site
    r = ls.site_cameras(root / "runs" / "sq02" / "v1", configs)
    assert r["site"] == "sq" and r["groups"] == [] and r["notes"] == []
    by = {c["id"]: c for c in r["cameras"]}
    assert sorted(by) == ["01", "02", "04"]                         # sq03: no package; other: another site
    assert by["04"]["run"] == str((root / "runs" / "sq04" / "v1").resolve())   # the start's run name
    assert by["01"]["E"] == pytest.approx(1050.3) and by["01"]["N"] == pytest.approx(2099.8)
    assert by["01"]["clip"] == "sq01" and by["01"]["run_name"] == "v1"
    assert by["01"]["name"] == "Terrace · 01"                      # North, renamed by the site config
    assert r["start_spot"] == by["02"]["name"] == "East · 02"
    assert by["04"]["name"] == "South · 04"
    # a start run without the others' run name: the newest package of each other clip
    r1 = ls.site_cameras(root / "runs" / "sq04" / "v2", configs)
    assert r1["start_spot"].endswith("04") and len(r1["cameras"]) == 3
    assert {c["id"]: c["run"] for c in r1["cameras"]}["01"].endswith("v1")
    # a run outside the site configs: itself alone
    lone = root / "elsewhere" / "x" / "v9"
    package(lone, "x", 5.0, 5.0)
    r2 = ls.site_cameras(lone, configs)
    assert [c["name"] for c in r2["cameras"]] == ["x"] and r2["start_spot"] == "x"
    with pytest.raises(FileNotFoundError):
        ls.site_cameras(root / "runs" / "sq03" / "v1", configs)


def test_the_site_config_leaves_cameras_out(site):
    root, configs = site
    base = root / "sites" / "sq.yaml"
    base.write_text("site: sq\noutput_dir: ../runs\nlive:\n  exclude: [sq04, 9999]\n", encoding="utf-8")
    r = ls.site_cameras(root / "runs" / "sq02" / "v1", configs)
    assert sorted(c["id"] for c in r["cameras"]) == ["01", "02"]            # by clip name (or number)
    assert r["notes"] == ["live.exclude: no camera 9999 here"]
    # the start camera is never left out
    r4 = ls.site_cameras(root / "runs" / "sq04" / "v1", configs)
    assert "04" in [c["id"] for c in r4["cameras"]] and any("start camera" in n for n in r4["notes"])
    base.write_text("site: sq\noutput_dir: ../runs\nlive:\n  exclude: 4\n", encoding="utf-8")
    assert ls.site_cameras(root / "runs" / "sq02" / "v1", configs)["notes"] == ["live.exclude is not a list: ignored"]


def test_site_list_the_sites_and_their_best_calibrated_start(site):
    root, configs = site
    runs = root / "runs"
    for run, rmse in (("sq01/v1", 0.4), ("sq02/v1", 0.2), ("sq04/v2", 0.1)):
        (runs / run / "_working").mkdir(parents=True)
        (runs / run / "_working" / "homography_used.json").write_text(json.dumps({"rmse_m": rmse}), encoding="utf-8")
    lst = ls.site_list(configs)
    assert [s["site"] for s in lst] == ["other", "sq"]
    sq = lst[1]
    assert sq["title"] == "Sq" and sq["cameras"] == 3
    # the best calibrated of the newest packages (sq04's newest, v1, has no RMSE: last)
    assert sq["start_run"] == str((runs / "sq02" / "v1").resolve()) and sq["rmse"] == 0.2 and sq["start_id"] == "02"
    assert lst[0]["cameras"] == 1 and lst[0]["rmse"] is None
    cams = {c["id"]: c for c in ls.site_cameras(runs / "sq04" / "v2", configs)["cameras"]}
    assert cams["01"]["rmse"] == 0.4 and cams["04"]["rmse"] == 0.1
    (root / "sites" / "sq.yaml").write_text('site: sq\noutput_dir: ../runs\nlive:\n  exclude: ["02"]\n', encoding="utf-8")
    sq = ls.site_list(configs)[1]
    assert sq["cameras"] == 2 and sq["start_id"] == "01"                          # left out: not the start
