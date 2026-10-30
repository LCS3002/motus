"""Live view: the cameras of a run's site (plans/live_controller/08 b, changed 28 Oct).

The Live view opens on one run (the start camera). The other cameras a phone can pick are the
Live packages of the same site: every site config that extends the same base config (one small
file per clip, `extends: trafalgar.yaml`) and has a run with a scene package
(`runs/<site><clip>/<run>/live/scene.json`; the run with the start camera's run name, else the
newest). A run outside the project's site configs is a site of one camera.

Names are made to be read on a phone: "<place> · <clip number>", e.g. "North · 6354". The place
is a compass direction from the site's centre (the middle of the cameras' bounding box), or
"Centre" near it.
- Up to 6 cameras: a plain list; each camera's place is its own direction.
- More: the cameras are grouped by place: nearby cameras form an area (complete linkage: every
  two cameras of an area within "near", NEAR_FRAC of the cameras' extent (the bounding box's
  diagonal), NEAR_MIN_M to NEAR_MAX_M; at most MAX_GROUP cameras an area; areas keep merging,
  nearest first, while there are more than MAX_GROUPS). An area is named by its direction; the
  same direction twice: "North", "North 2" (nearer the centre first).
- The site config may leave cameras out and fix or rename areas (read only, never written):

      live:
        exclude: [6368]               # left out of the list (clip number or name)
        groups:
          Fountain: [6363, 6358]      # fixed: these cameras (clip number or name) form "Fountain"
          Gallery steps: North-east   # renamed: the automatic area "North-east"

  The start camera is never left out (the view opened on it).

Deterministic: the same packages give the same names and areas. `site_cameras(run, configs)`
gives the view's /api/live/site; `site_list(configs)` the sites with Live packages and each one's
default start camera (the lowest calibration RMSE) for the view's site picker, /api/live/sites.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

MAX_LIST = 6            # up to this many cameras: a plain list, no areas
MAX_GROUP = 6           # an automatic area holds at most this many cameras
MAX_GROUPS = 6          # areas keep merging (nearest first) while there are more
NEAR_FRAC = 0.15        # cameras this close (every two of an area; of the site's extent) form an area
NEAR_MIN_M, NEAR_MAX_M = 20.0, 60.0
CENTRE_FRAC = 0.2       # a place this near the centre (of the furthest place's distance): "Centre"
COMPASS = ("North", "North-east", "East", "South-east", "South", "South-west", "West", "North-west")


# --------------------------------------------------------------------------- places (pure)
def compass(dE: float, dN: float) -> tuple[str, float]:
    """The 8-point compass direction of (dE, dN) and its bearing (degrees from north,
    clockwise)."""
    b = math.degrees(math.atan2(dE, dN)) % 360.0
    return COMPASS[int(((b + 22.5) % 360.0) // 45.0)], b


def site_centre(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The middle of the points' bounding box."""
    es, ns = [p[0] for p in points], [p[1] for p in points]
    return (min(es) + max(es)) / 2.0, (min(ns) + max(ns)) / 2.0


def place_names(points: list[tuple[float, float]], centre: tuple[float, float]) -> list[tuple[str, float]]:
    """Each point's place: its compass direction from the centre, "Centre" when it is within
    CENTRE_FRAC of the furthest point's distance; and its bearing (Centre: -1, so it comes
    first)."""
    d = [math.hypot(p[0] - centre[0], p[1] - centre[1]) for p in points]
    far = max(d, default=0.0)
    out = []
    for p, di in zip(points, d):
        if far <= 0 or di <= CENTRE_FRAC * far:
            out.append(("Centre", -1.0))
        else:
            out.append(compass(p[0] - centre[0], p[1] - centre[1]))
    return out


def near_m(points: list[tuple[float, float]]) -> float:
    """How close cameras must be to form an area: NEAR_FRAC of the bounding box's diagonal,
    NEAR_MIN_M to NEAR_MAX_M."""
    es, ns = [p[0] for p in points], [p[1] for p in points]
    diag = math.hypot(max(es) - min(es), max(ns) - min(ns)) if points else 0.0
    return min(NEAR_MAX_M, max(NEAR_MIN_M, NEAR_FRAC * diag))


def cluster(points: list[tuple[float, float]], *, max_group: int = MAX_GROUP, max_groups: int = MAX_GROUPS,
            near: float = NEAR_MIN_M, reserved: int = 0) -> list[list[int]]:
    """Nearby points into areas (indices, each area sorted, areas by their first index).
    Complete linkage: the distance of two areas is that of their two furthest points. The
    nearest two areas merge while they fit (max_group) and are within near_m, or while there are
    more than max_groups areas (reserved: areas fixed elsewhere count too). Ties: the lower
    indices first."""
    groups = [[i] for i in range(len(points))]

    def dist(a, b):
        return max(math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) for i in a for j in b)

    while len(groups) > 1:
        best = None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                if len(groups[x]) + len(groups[y]) > max_group:
                    continue
                d = dist(groups[x], groups[y])
                if best is None or d < best[0] - 1e-9:
                    best = (d, x, y)
        if best is None:
            break
        d, x, y = best
        if d > near and len(groups) + reserved <= max_groups:
            break
        groups[x] = sorted(groups[x] + groups[y])
        del groups[y]
    return sorted(groups, key=lambda g: g[0])


def _numbered(names: list[str], dists: list[float]) -> list[str]:
    """The same name twice: the nearer one keeps it, the others get " 2", " 3" (by distance)."""
    out = list(names)
    for n in set(names):
        idx = sorted((i for i, m in enumerate(names) if m == n), key=lambda i: (dists[i], i))
        for k, i in enumerate(idx[1:], start=2):
            out[i] = f"{n} {k}"
    return out


def arrange(cams: list[dict], live_cfg: dict | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """Names, areas and order for cameras [{"id", "E", "N", ...}] (ids unique). Returns
    (cameras with "name", "place", "group"; groups [{"name", "spots"}], empty up to MAX_LIST
    cameras; notes about the site config's live.groups)."""
    notes: list[str] = []
    fixed: dict[str, list[int]] = {}
    renames: dict[str, str] = {}
    by_key = {}
    for i, c in enumerate(cams):
        by_key[str(c["id"])] = i
        by_key[str(c.get("clip", c["id"]))] = i
    raw = (live_cfg or {}).get("groups") if isinstance(live_cfg, dict) else None
    if raw is not None and not isinstance(raw, dict):
        notes.append("live.groups is not a mapping: ignored")
        raw = None
    taken: set[int] = set()
    for gname, v in (raw or {}).items():
        gname = str(gname).strip()
        if not gname:
            continue
        if isinstance(v, str):
            renames[v.strip()] = gname
        elif isinstance(v, list):
            members = []
            for k in v:
                i = by_key.get(str(k).strip())
                if i is None:
                    notes.append(f"live.groups {gname}: no camera {k} here")
                elif i in taken:
                    notes.append(f"live.groups {gname}: camera {k} is in another group")
                else:
                    members.append(i)
                    taken.add(i)
            if members:
                fixed[gname] = sorted(members)
        else:
            notes.append(f"live.groups {gname}: a list of cameras or the name of an area")

    pts = [(float(c["E"]), float(c["N"])) for c in cams]
    out = [dict(c) for c in cams]
    if not cams:
        return out, [], notes
    centre = site_centre(pts)

    def dist(p):
        return math.hypot(p[0] - centre[0], p[1] - centre[1])

    def mid(g):
        return (sum(pts[i][0] for i in g) / len(g), sum(pts[i][1] for i in g) / len(g))

    grouped = len(cams) > MAX_LIST
    rest = [i for i in range(len(cams)) if i not in taken]
    if grouped:                                           # nearby cameras: one area
        parts = [[rest[k] for k in g] for g in cluster([pts[i] for i in rest], near=near_m(pts), reserved=len(fixed))]
    else:                                                 # each camera its own place
        parts = [[i] for i in rest]
    mids = [mid(g) for g in parts]
    if len(cams) == 1:
        places = [("", 0.0)] * len(parts)                 # one camera: its number only
    else:
        places = place_names(mids, centre)
    names = [renames.get(n, n) for n, _ in places]
    if grouped:                                           # areas: one name each
        names = _numbered(names, [dist(m) for m in mids])
    def order(members, b):                                # Centre, then clockwise by sector, near first
        sector = -1 if b < 0 else int(((b + 22.5) % 360.0) // 45.0)
        return (sector, round(dist(mid(members)), 3), b)

    areas: list[tuple[str, list[int], tuple]] = []        # (name, members, order)
    for gname, members in fixed.items():
        m = mid(members)
        b = compass(m[0] - centre[0], m[1] - centre[1])[1] if dist(m) > 0 else -1.0
        areas.append((gname, members, order(members, b)))
    used = set(fixed)
    for g, n, (_, b) in zip(parts, names, places):
        k, base = 2, n
        while grouped and n in used:                      # a renamed area meets a fixed one
            n, k = f"{base} {k}", k + 1
        used.add(n)
        areas.append((n, g, order(g, b)))
    areas.sort(key=lambda a: (a[2], a[0]))
    rank = {}
    for k, (gname, members, _) in enumerate(areas):
        for i in members:
            rank[i] = k
            out[i]["place"] = gname
            out[i]["group"] = gname if grouped else None
            out[i]["name"] = f"{gname} · {cams[i]['id']}" if gname else str(cams[i]["id"])
    out = [out[i] for i in sorted(range(len(cams)), key=lambda i: (rank[i], str(cams[i]["id"])))]
    groups = [{"name": a[0], "spots": [out_c["name"] for out_c in out if out_c["place"] == a[0]]}
              for a in areas] if grouped else []
    return out, groups, notes


# --------------------------------------------------------------------------- the site's packages
_RAW: dict = {}                                          # (config, mtime) -> its YAML


def _raw(path: Path) -> dict:
    """A config's own YAML (no `extends` resolved), kept while the file is unchanged."""
    try:
        key = (str(path), path.stat().st_mtime_ns)
        if key not in _RAW:
            _RAW[key] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _RAW[key]
    except (OSError, yaml.YAMLError):
        return {}


def _base(path: Path, raw: dict) -> Path:
    ext = raw.get("extends")
    return (path.parent / ext).resolve() if ext else path.resolve()


_ROOTS: dict = {}                                        # (config, its files' mtimes) -> run root


def _run_root(path: Path) -> Path | None:
    """The config's run folder (<output_dir>/<site>), as the app finds it (kept while the config
    and the configs it extends are unchanged)."""
    from .config import load_config

    try:
        chain, f = [], path
        while f is not None and len(chain) < 8:
            chain.append(f.stat().st_mtime)
            ext = _raw(f).get("extends")
            f = (f.parent / ext).resolve() if ext else None
    except OSError:
        return None
    key = (str(path), tuple(chain))
    if key not in _ROOTS:
        try:
            cfg = load_config(path)
            _ROOTS[key] = ((cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site).resolve()
        except Exception:  # noqa: BLE001 (a broken config has no runs here)
            _ROOTS[key] = None
    return _ROOTS[key]


def _rmse(run: Path) -> float | None:
    """The run's calibration error (m, <run>/_working/homography_used.json rmse_m), if known."""
    try:
        v = json.loads((run / "_working" / "homography_used.json").read_text(encoding="utf-8")).get("rmse_m")
        return round(float(v), 3) if v is not None and math.isfinite(float(v)) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _camera(run: Path) -> dict | None:
    """A run's scene package as a camera: clip, run name, E / N (the package origin + C), the
    calibration's RMSE (m, or None)."""
    f = run / "live" / "scene.json"
    try:
        m = json.loads(f.read_text(encoding="utf-8"))
        o, C = m.get("origin") or {"E": 0.0, "N": 0.0}, m["camera"]["C"]   # no origin: the site's
        return {"clip": str(m.get("clip") or run.parent.name), "run_name": str(m.get("run") or run.name),
                "run": str(run), "E": round(float(o["E"]) + float(C[0]), 2),
                "N": round(float(o["N"]) + float(C[1]), 2), "rmse": _rmse(run)}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def _realtime(run: Path) -> bool:
    """A realtime recording's package (scene.json "realtime": true; plans/live_controller/10)."""
    try:
        return bool(json.loads((run / "live" / "scene.json").read_text(encoding="utf-8")).get("realtime"))
    except (OSError, ValueError, AttributeError):
        return False


def _pick(root: Path, run_name: str) -> Path | None:
    """The clip's run to use: the one called run_name, else the newest package."""
    if (root / run_name / "live" / "scene.json").is_file():
        return root / run_name
    pk = [p for p in root.glob("*/live/scene.json")] if root.is_dir() else []
    if not pk:
        return None
    return max(pk, key=lambda p: (p.stat().st_mtime, p.parent.parent.name)).parent.parent


def _short(clip: str, site: str | None) -> str:
    """The clip's number: "trafalgar6354" in site "trafalgar" -> "6354"."""
    if site and clip.lower().startswith(site.lower()) and len(clip) > len(site):
        return clip[len(site):].lstrip("_- ") or clip
    return clip


def _sites(configs: list[Path]) -> tuple[dict, dict, dict]:
    """The configs' own YAML, run roots and sites: {base config: [member configs]} (the configs
    that extend one base; a config without `extends` is its own base)."""
    raws = {Path(f).resolve(): _raw(Path(f)) for f in configs}
    roots = {f: _run_root(f) for f in raws}
    sites: dict[Path, list[Path]] = {}
    for f in sorted(raws):
        sites.setdefault(_base(f, raws[f]), []).append(f)
    return raws, roots, sites


def _site_of(base: Path, members: list[Path], raws: dict) -> tuple[str | None, dict | None]:
    """A site's name and its live section (the base config's)."""
    braw = raws.get(base) or _raw(base)
    first = raws.get(members[0], {}) if members else {}
    site = braw.get("site") or first.get("site")
    live_cfg = braw.get("live") if isinstance(braw.get("live"), dict) else None
    return (str(site) if site else None), live_cfg


def _excluded(cams: list[dict], live_cfg: dict | None, keep: Path | None) -> tuple[list[dict], list[str]]:
    """The cameras the site config's live.exclude leaves out (clip number or name; never the
    start camera `keep`), and notes about the entries."""
    raw = (live_cfg or {}).get("exclude") if isinstance(live_cfg, dict) else None
    if raw is None:
        return cams, []
    if not isinstance(raw, list):
        return cams, ["live.exclude is not a list: ignored"]
    notes, drop = [], set()
    for k in raw:
        key = str(k).strip()
        hit = [i for i, c in enumerate(cams) if key in (str(c["id"]), str(c["clip"]))]
        if not hit:
            notes.append(f"live.exclude: no camera {k} here")
        for i in hit:
            if keep is not None and Path(cams[i]["run"]) == keep:
                notes.append(f"live.exclude: {k} is the start camera: shown")
            else:
                drop.add(i)
    return [c for i, c in enumerate(cams) if i not in drop], notes


def _cameras(members: list[Path], roots: dict, run_name: str, site: str | None,
             start: dict | None = None) -> list[dict]:
    """The packages of a site's clips (the run called run_name, else the newest; start: the
    start camera's own run for its clip), with ids."""
    cams, seen = [], set()
    root = Path(start["run"]).parent if start else None
    for f in members:
        rr = roots[f]
        if rr is None or rr in seen:
            continue
        seen.add(rr)
        r = Path(start["run"]) if start and rr == root else _pick(rr, run_name)
        c = _camera(r) if r is not None else None
        if c is not None:
            cams.append(c)
    if start is not None and not any(c["run"] == start["run"] for c in cams):
        cams.append(start)
    for c in cams:
        c["id"] = _short(c["clip"], site)
    ids = [c["id"] for c in cams]
    for c in cams:                                        # two runs of one clip number: the clip
        if ids.count(c["id"]) > 1:
            c["id"] = c["clip"]
    return cams


def site_cameras(run: str | Path, configs: list[Path]) -> dict:
    """The cameras a Live view on `run` can show (its site's packages): {"site", "start_spot",
    "centre": {"E", "N"}, "cameras": [{"name", "id", "clip", "run_name", "run", "E", "N", "rmse",
    "place", "group"}], "groups": [{"name", "spots"}] (empty up to MAX_LIST cameras), "notes"}.
    configs: the project's site configs (the app's list)."""
    run = Path(run).resolve()
    start = _camera(run)
    if start is None:
        raise FileNotFoundError(f"no live package in {run}")
    root = run.parent
    raws, roots, sites = _sites(configs)
    mine = next((f for f in raws if roots[f] == root), None)
    site, live_cfg, members = None, None, []
    if mine is not None:
        base = _base(mine, raws[mine])
        members = sites.get(base, [mine])
        site, live_cfg = _site_of(base, members, raws)
        site = site or str(raws[mine].get("site") or root.name)
    cams = _cameras(members, roots, run.name, site, start)
    cams, ex_notes = _excluded(cams, live_cfg, run)
    out, groups, notes = arrange(cams, live_cfg)
    notes = ex_notes + notes
    cE, cN = site_centre([(c["E"], c["N"]) for c in out])
    if mine is None and _realtime(run):                   # a phone recording (runs/live/): its own site
        from .live_track import recording_title

        site = "Realtime"
        for c in out:
            if Path(c["run"]) == run:
                c["name"] = recording_title(run.name)
    start_name = next(c["name"] for c in out if Path(c["run"]) == run)
    return {"site": site or root.name, "start_spot": start_name, "centre": {"E": round(cE, 2), "N": round(cN, 2)},
            "cameras": out, "groups": groups, "notes": notes}


def site_list(configs: list[Path]) -> list[dict]:
    """The sites with Live packages (the view's site picker): [{"site", "title", "cameras": n,
    "start_run", "start_id", "rmse"}], by name. The default start camera is the best calibrated
    (the lowest RMSE; unknown RMSEs last, then by clip); live.exclude leaves cameras out."""
    raws, roots, sites = _sites(configs)
    out = []
    for base, members in sites.items():
        site, live_cfg = _site_of(base, members, raws)
        cams, _ = _excluded(_cameras(members, roots, "", site), live_cfg, None)
        if not cams:
            continue
        best = min(cams, key=lambda c: (c["rmse"] is None, c["rmse"] or 0.0, c["clip"]))
        name = site or Path(best["run"]).parent.name
        out.append({"site": name, "title": name[:1].upper() + name[1:], "cameras": len(cams),
                    "start_run": best["run"], "start_id": best["id"], "rmse": best["rmse"]})
    return sorted(out, key=lambda s: s["site"].lower())
