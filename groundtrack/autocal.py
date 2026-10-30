"""Calibration without clicking.

1. Same spot (`transfer`): a clip filmed from a position that is already calibrated inherits
   that calibration. From one camera position, any two views (other direction, other zoom)
   are related by a single image homography, so this is exact geometry, not a guess: match
   the two frames, chain the homographies, refit the physical camera.
2. Roads (`fit_to_roads`): with the camera position known, vehicles must drive on roads.
   The camera (direction, tilt, roll, zoom) is solved so that the tracked vehicles land on
   the OpenStreetMap road lines. No clicks and no image matching, so old aerials, blur and
   haze don't matter.
3. People (`fit_to_people`): walking people are ~1.70 m tall and walk ~1.3 m/s, and don't
   walk through buildings or water. That fixes tilt, camera height and zoom; on an open
   plaza nothing fixes the direction, so 2 clicked points finish it.

Both return a Homography like the point picker does, and `quality_checks` tests any
calibration against what the footage itself says (people ~1.7 m, vehicles on roads,
plausible speeds) so a bad automatic result is caught instead of silently used.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .homography import Homography, fit_homography

USER_AGENT = "groundtrack (architecture research tool)"
OVERPASS_MIRRORS = ("https://overpass-api.de/api/interpreter",
                    "https://lz4.overpass-api.de/api/interpreter",
                    "https://z.overpass-api.de/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter")


def _overpass(query: str, timeout: float = 60.0, rounds: int = 2) -> dict:
    """Run an Overpass query, trying the mirrors in turn (they are often busy)."""
    import time

    last = None
    for k in range(rounds):
        for url in OVERPASS_MIRRORS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode(
                    {"data": query}).encode(), headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except OSError as e:          # HTTP 429/5xx, timeouts, DNS
                last = e
        time.sleep(5 * (k + 1))
    raise OSError(f"OpenStreetMap (Overpass) unavailable: {last}")
PERSON_M = 1.70      # typical adult height
WALK_M_S = 1.3       # typical free walking speed
OVERPASS = "https://overpass-api.de/api/interpreter"


# --------------------------------------------------------------------------- 1. same spot
def match_frames(dst_bgr: np.ndarray, src_bgr: np.ndarray, max_dim: int = 1600,
                 inliers_out: list | None = None):
    """Homography taking dst pixels to src pixels (same camera position), with inlier stats.

    Returns (H, n_inliers, inlier_ratio) or (None, 0, 0). inliers_out, if given, receives the
    inlier pixel positions in the dst frame."""
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = min(1.0, max_dim / max(g.shape))
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(2.0, (8, 8)).apply(g), s

    g1, s1 = prep(dst_bgr)
    g2, s2 = prep(src_bgr)
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None, 0, 0.0
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    good = [m for m, n in (p for p in matcher.knnMatch(d1, d2, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None, len(good), 0.0
    a = np.float32([k1[m.queryIdx].pt for m in good]) / s1
    b = np.float32([k2[m.trainIdx].pt for m in good]) / s2
    H, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 4.0, maxIters=10000, confidence=0.999)
    if H is None:
        return None, 0, 0.0
    n = int(mask.sum())
    if inliers_out is not None:
        inliers_out.append(a[mask.ravel() > 0])
    return H, n, n / len(good)


ROTATION_SPREAD_MAX = 1.08   # same-spot pairs measured 1.00-1.06, moved cameras 1.3-1.4
DEPTH_SPAN_MIN = 0.20        # matches must span this much of the frame height (p10-p90)


def rotation_spread(H: np.ndarray, size_dst, size_src) -> float:
    """How far a frame-to-frame homography is from a pure camera rotation: 1.0 = exactly
    (the same spot, any zoom). Between two spots the near ground shows parallax: one
    homography then fits the matched features (often distant buildings) but not the ground,
    and a transferred calibration would be wrong there, however many features match.

    K_src^-1 H K_dst is a scaled rotation for the right focal lengths; the smallest ratio of
    its largest to smallest singular value over a grid of focal lengths is returned."""
    fs = np.geomspace(300.0, 30000.0, 150)         # 3 % steps

    def Ks(size):
        K = np.zeros((len(fs), 3, 3))
        K[:, 0, 0] = K[:, 1, 1] = fs
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = size[0] / 2, size[1] / 2, 1.0
        return K

    Kd, Ks_inv = Ks(size_dst), np.linalg.inv(Ks(size_src))
    M = Ks_inv[None] @ H[None, None] @ Kd[:, None]          # (dst focal, src focal, 3, 3)
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.min(s[..., 0] / np.maximum(s[..., 2], 1e-12)))


def transfer(src: Homography, src_frame: np.ndarray, dst_frame: np.ndarray,
             camera_prior: dict | None = None, min_inliers: int = 40, log=print):
    """Calibration for dst_frame from a calibrated src_frame filmed from the same spot.

    Returns (Homography or None, info dict)."""
    pts: list = []
    H, n, ratio = match_frames(dst_frame, src_frame, inliers_out=pts)
    info = {"method": "same-spot transfer", "inliers": n, "inlier_ratio": round(ratio, 3)}
    if H is None or n < min_inliers or ratio < 0.2:
        info["reason"] = f"frames don't match well enough ({n} inliers, {ratio:.0%})"
        return None, info
    h_dst, w_dst = dst_frame.shape[:2]
    h_src, w_src = src_frame.shape[:2]
    # the rotation test only sees a camera that moved if the matches lie at several depths:
    # features all along one distant skyline barely shift with a move of 10 m, the ground does
    v = pts[0][:, 1] if pts else np.zeros(1)
    span = float(np.percentile(v, 90) - np.percentile(v, 10)) / h_dst
    info["depth_span"] = round(span, 3)
    if span < DEPTH_SPAN_MIN:
        info["reason"] = (f"the shared features are all far away ({span:.0%} of the frame height):"
                          " can't tell whether the camera moved; click points instead")
        return None, info
    spread = rotation_spread(H, (w_dst, h_dst), (w_src, h_src))
    info["rotation_spread"] = round(spread, 3)
    if spread > ROTATION_SPREAD_MAX:
        info["reason"] = (f"filmed from a different spot: the views differ by more than a "
                          f"camera turn (parallax; rotation spread {spread:.2f})")
        return None, info
    # synthetic correspondences: a grid over the new frame, through the old calibration
    gu, gv = np.meshgrid(np.linspace(0.03, 0.97, 32) * w_dst, np.linspace(0.03, 0.97, 18) * h_dst)
    px = np.column_stack([gu.ravel(), gv.ravel()])
    q = cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    inside = (q[:, 0] > 0) & (q[:, 0] < w_src) & (q[:, 1] > 0) & (q[:, 1] < h_src)
    world = src.to_world(q)
    ok = inside & np.isfinite(world).all(axis=1)
    cam = getattr(src, "camera_params", None) or src.camera()
    if cam and ok.any():   # drop the far fringe (near the horizon one pixel is many metres)
        d = np.hypot(world[:, 0] - cam["E"], world[:, 1] - cam["N"])
        ok &= d < np.nanpercentile(d[ok], 95) * 1.5
    info["shared_points"] = int(ok.sum())
    if ok.sum() < 12:
        info["reason"] = "the two views share too little ground"
        return None, info
    px, world = px[ok], world[ok]
    if camera_prior:
        from .posefit import camera_calibration

        prior = dict(camera_prior)
        src_cam = getattr(src, "camera_params", None)
        if src_cam:             # the old fit is a better starting point than the config
            prior.update({k: src_cam[k] for k in ("E", "N", "height_m")})
            prior["position_tol_m"] = min(float(prior.get("position_tol_m", 15)), 5.0)
            if src.terrain is not None:        # its height is above the old fit's datum
                prior["ground_z_m"] = src.ground_z_m
        h = camera_calibration(px, world, (w_dst, h_dst), prior, terrain=src.terrain)
    else:
        h = fit_homography(px, world, image_size=(w_dst, h_dst))
    h.image_size = (w_dst, h_dst)
    info["fit_rmse_m"] = round(float(h.rmse_m), 3)
    log(f"  same spot: {n} matching features ({ratio:.0%}), {int(ok.sum())} shared ground points,"
        f" refit error {h.rmse_m:.2f} m")
    return h, info


# --------------------------------------------------------------------------- 2. roads
ROAD_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|"
              "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|"
              "bus_guideway|busway")


MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link",
         "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


class Road(np.ndarray):
    """(n, 2) EPSG:27700 polyline with its OSM highway class in `.kind`."""

    def __new__(cls, xy, kind=""):
        obj = np.asarray(xy, float).view(cls)
        obj.kind = kind
        return obj

    def __array_finalize__(self, obj):
        self.kind = getattr(obj, "kind", "")


def osm_roads(E: float, N: float, radius_m: float = 1500.0, timeout: float = 60.0,
              cache: Path | None = None) -> list[np.ndarray]:
    """Road centrelines around (E, N) from OpenStreetMap, as (n, 2) arrays in EPSG:27700.

    Only the bounding box of the search area is sent to the Overpass API."""
    from rasterio.warp import transform

    if cache is not None and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data and isinstance(data[0], dict):
            return [Road(np.asarray(w["xy"], float), w["kind"]) for w in data]
    lon, lat = transform("EPSG:27700", "EPSG:4326",
                         [E - radius_m, E + radius_m], [N - radius_m, N + radius_m])
    bbox = f"{lat[0]:.6f},{lon[0]:.6f},{lat[1]:.6f},{lon[1]:.6f}"
    q = f'[out:json][timeout:50];way["highway"~"^({ROAD_TYPES})$"]({bbox});out geom;'
    data = _overpass(q, timeout)
    ways = []
    for el in data.get("elements", []):
        g = el.get("geometry") or []
        if len(g) < 2:
            continue
        xs, ys = transform("EPSG:4326", "EPSG:27700", [p["lon"] for p in g], [p["lat"] for p in g])
        ways.append(Road(np.column_stack([xs, ys]), (el.get("tags") or {}).get("highway", "")))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([{"kind": w.kind, "xy": w.round(2).tolist()} for w in ways]),
                         encoding="utf-8")
    return ways


class RoadDistance:
    """Distance (m) from any ground point to the nearest road line, from a raster."""

    def __init__(self, ways: list[np.ndarray], E: float, N: float, radius_m: float,
                 res: float = 1.0, minor_penalty_m: float = 0.0):
        from scipy.ndimage import distance_transform_edt

        self.x0, self.y1, self.res = E - radius_m, N + radius_m, res
        n = int(2 * radius_m / res) + 1
        self.n = n
        dist = []
        angle = np.zeros((n, n), np.float32)          # road direction (deg, mod 180) per pixel
        for major in (True, False):
            mask = np.ones((n, n), np.uint8)
            for w in ways:
                if minor_penalty_m and (getattr(w, "kind", "") in MAJOR) != major:
                    continue
                pts = np.round(np.column_stack([(w[:, 0] - self.x0) / res,
                                                (self.y1 - w[:, 1]) / res])).astype(np.int32)
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(mask, tuple(map(int, a)), tuple(map(int, b)), 0, 1)
                    ang = np.degrees(np.arctan2(-(b[1] - a[1]), b[0] - a[0])) % 180
                    cv2.line(angle, tuple(map(int, a)), tuple(map(int, b)), float(ang), 1)
            if not minor_penalty_m:
                d, idx = distance_transform_edt(mask, return_indices=True)
                dist = [d * res]
                self.nearest = idx
                break
            d, idx = distance_transform_edt(mask, return_indices=True)
            dist.append((d * res + (0 if major else minor_penalty_m), idx))
        if minor_penalty_m:
            (dm, im), (dn, inn) = dist
            use_major = dm <= dn
            self.d = np.where(use_major, dm, dn)
            self.nearest = np.where(use_major[None], im, inn)
        else:
            self.d = dist[0]
        self.angle = angle

    def _rc(self, xy):
        return (self.y1 - xy[:, 1]) / self.res, (xy[:, 0] - self.x0) / self.res

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        r, c = self._rc(xy)
        out = map_coordinates(self.d, [r, c], order=1, mode="nearest")
        far = (c < 0) | (r < 0) | (c > self.n - 1) | (r > self.n - 1)
        return np.where(far | ~np.isfinite(out), 200.0, out)

    def direction(self, xy: np.ndarray) -> np.ndarray:
        """Direction (deg from +E, mod 180) of the road nearest to each point."""
        r, c = self._rc(xy)
        ri = np.clip(np.round(r), 0, self.n - 1).astype(int)
        ci = np.clip(np.round(c), 0, self.n - 1).astype(int)
        nr, nc = self.nearest[0][ri, ci], self.nearest[1][ri, ci]
        return self.angle[nr, nc]


class VehicleSamples(tuple):
    """(uv, ids, times) - unpacks like a tuple; .left / .right / .cls hold the box-bottom
    corners and the class for the size cue."""

    def __new__(cls, uv, ids, ts):
        obj = super().__new__(cls, (uv, ids, ts))
        obj.left = obj.right = np.zeros((0, 2))
        obj.cls = np.zeros(0, object)
        return obj


# typical footprints (length, width) in m: a box's bottom edge on the ground spans about
# width*|cos a| + length*|sin a|, a = angle between the vehicle's heading and the view ray
VEHICLE_SIZE = {"car": (4.5, 1.8), "bus": (11.0, 2.55), "truck": (8.0, 2.5),
                "motorcycle": (2.1, 0.8)}


def vehicle_track_samples(raw: pd.DataFrame, registration: dict | None = None,
                          max_tracks: int = 300, per_track: int = 12, seed: int = 0):
    """Foot pixels of moving vehicle tracks: a few samples per track, their track ids and
    times (s)."""
    from .registration import apply_registration

    r = raw[raw["class"].isin(["car", "bus", "truck", "motorcycle"])].copy()
    r = r[~r["predicted"].astype(str).str.lower().isin(["true", "1"])]
    if r.empty:
        return VehicleSamples(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))
    r["u"] = (r["x1"] + r["x2"]) / 2
    r["v"] = r["y2"]
    r["ul"], r["ur"] = r["x1"].astype(float), r["x2"].astype(float)
    if registration:
        fr = r["frame"].to_numpy()
        uv = apply_registration(fr, r[["u", "v"]].to_numpy(float), registration)
        lft = apply_registration(fr, r[["ul", "v"]].to_numpy(float), registration)
        rgt = apply_registration(fr, r[["ur", "v"]].to_numpy(float), registration)
        r["u"], r["v"], r["ul"], r["ur"] = uv[:, 0], uv[:, 1], lft[:, 0], rgt[:, 0]
        r["vl"], r["vr"] = lft[:, 1], rgt[:, 1]
    else:
        r["vl"] = r["vr"] = r["v"]
    # moving tracks only: parked cars sit beside roads and would pull the fit off them
    span = r.groupby("track_id").agg(du=("u", lambda s: s.max() - s.min()),
                                     dv=("v", lambda s: s.max() - s.min()), n=("u", "size"))
    moving = span[(np.hypot(span["du"], span["dv"]) > 40) & (span["n"] >= 8)].index
    rng = np.random.default_rng(seed)
    if len(moving) > max_tracks:
        moving = rng.choice(moving, max_tracks, replace=False)
    pts, ids, ts, left, right, cls = [], [], [], [], [], []
    for tid in moving:
        t = r[r["track_id"] == tid].sort_values("frame")
        k = np.linspace(0, len(t) - 1, min(per_track, len(t))).round().astype(int)
        pts.append(t[["u", "v"]].to_numpy(float)[k])
        left.append(t[["ul", "vl"]].to_numpy(float)[k])
        right.append(t[["ur", "vr"]].to_numpy(float)[k])
        ids.append(np.full(len(k), tid))
        ts.append(t["time_s"].to_numpy(float)[k])
        cls.append(np.full(len(k), t["class"].mode().iloc[0]))
    if not pts:
        return VehicleSamples(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))
    S = VehicleSamples(np.vstack(pts), np.concatenate(ids), np.concatenate(ts))
    S.left, S.right, S.cls = np.vstack(left), np.vstack(right), np.concatenate(cls)
    return S


class RoadObjective:
    """How well a camera puts the tracked vehicles where vehicles can be.

    Four cues, as residuals for least squares:
      road       each sample near an OpenStreetMap road (minor roads cost a few metres more)
      direction  each step of a track runs along the nearest road, not across it
      size       each vehicle's box bottom on the ground matches its footprint at that viewing
                 angle (car 4.5 x 1.8 m, ...): this sets the zoom / scale
      speed      no impossible speeds (slow traffic is normal: queues, lights)
    Parameters p: yaw, tilt, roll (deg), log focal (px), dE, dN (m), height (m).
    With a terrain (and its datum z0, the ground at (E, N)) pixels are ray-cast onto it;
    the height is then above the ground under the camera.
    """

    def __init__(self, S: VehicleSamples, road: RoadDistance, E: float, N: float, size,
                 speed_band=(0.2, 40.0), terrain=None, z0: float | None = None):
        self.S, self.road, self.E, self.N = S, road, E, N
        self.terrain, self.z0 = terrain, z0
        self.w, self.h = size
        uv, ids, ts = S
        self.uv = uv
        same = ids[1:] == ids[:-1]
        self.i0 = np.nonzero(same)[0]
        self.i1 = self.i0 + 1
        self.dts = np.maximum(ts[self.i1] - ts[self.i0], 1e-3)
        self.tracks, self.t_pair = np.unique(ids[self.i0], return_inverse=True)
        self.all_tracks, self.t_all = np.unique(ids, return_inverse=True)
        self.v_lo, self.v_hi = speed_band
        self.L = np.array([VEHICLE_SIZE.get(c, (4.5, 1.8))[0] for c in S.cls])
        self.W = np.array([VEHICLE_SIZE.get(c, (4.5, 1.8))[1] for c in S.cls])
        n = max(len(uv), 1)
        self.w_size = 8.0 * np.sqrt(n / max(len(self.all_tracks), 1))
        self.w_speed = 3.0 * np.sqrt(n / max(len(self.tracks), 1))

    def ground(self, p, pix):
        """Pixels -> ground (E, N) under camera parameters p."""
        yaw, pitch, roll, logf, dE, dN, height = p
        f = math.exp(logf)
        ry, rp, rr = math.radians(yaw), math.radians(pitch), math.radians(roll)
        fwd = np.array([math.sin(ry) * math.cos(rp), math.cos(ry) * math.cos(rp), -math.sin(rp)])
        right = np.array([math.cos(ry), -math.sin(ry), 0.0])
        down = np.cross(fwd, right)
        right, down = (math.cos(rr) * right + math.sin(rr) * down,
                       -math.sin(rr) * right + math.cos(rr) * down)
        x = (pix[:, 0] - self.w / 2) / f
        y = (pix[:, 1] - self.h / 2) / f
        ray = fwd[None] + x[:, None] * right[None] + y[:, None] * down[None]
        if self.terrain is not None:
            from .terrain import raycast

            return raycast(np.array([dE, dN, height]), ray, self.terrain, self.z0,
                           (self.E, self.N), step_m=1.0)
        t = np.where(ray[:, 2] < -1e-6, height / -np.minimum(ray[:, 2], -1e-6), np.nan)
        return np.column_stack([self.E + dE + t * ray[:, 0], self.N + dN + t * ray[:, 1]])

    def residuals(self, p, robust: float = 4.0) -> np.ndarray:
        g = self.ground(p, self.uv)
        fin = np.isfinite(g).all(axis=1)
        d = np.where(fin, self.road(np.nan_to_num(g, nan=1e9)), 200.0)
        r_road = np.sqrt(2 * robust * (np.sqrt(1 + (d / robust) ** 2) - 1))   # soft-L1
        i0, i1 = self.i0, self.i1
        mv = g[i1] - g[i0]
        # speed: geometric mean per track, only impossible values cost
        step = np.linalg.norm(mv, axis=1) / self.dts
        step = np.where(np.isfinite(step), step, 1e4)
        k = len(self.tracks)
        v = np.exp(np.bincount(self.t_pair, np.log(np.maximum(step, 1e-3)), k)
                   / np.maximum(np.bincount(self.t_pair, minlength=k), 1))
        r_speed = self.w_speed * (np.maximum(0, np.log(v / self.v_hi))
                                  + np.maximum(0, np.log(self.v_lo / v)))
        # direction: along the nearest road
        mid = np.nan_to_num((g[i0] + g[i1]) / 2, nan=1e9)
        ang = np.degrees(np.arctan2(mv[:, 1], mv[:, 0])) % 180
        dd = np.abs((ang - self.road.direction(mid) + 90) % 180 - 90)
        r_dir = 3.0 * np.sin(np.radians(np.where(np.isfinite(dd), dd, 90)))
        # size: box bottom edge on the ground vs the footprint seen from this angle
        meas = np.linalg.norm(self.ground(p, self.S.right) - self.ground(p, self.S.left), axis=1)
        heading = np.full(len(g), np.nan)
        heading[i0] = np.arctan2(mv[:, 1], mv[:, 0])
        ray = g - np.array([self.E + p[4], self.N + p[5]])
        a = heading - np.arctan2(ray[:, 1], ray[:, 0])
        expect = self.L * np.abs(np.sin(a)) + self.W * np.abs(np.cos(a))
        ratio = np.log(np.maximum(meas, 1e-3) / np.maximum(expect, 1e-3))
        use = np.isfinite(ratio)
        n_all = len(self.all_tracks)
        tr = (np.bincount(self.t_all, np.where(use, ratio, 0.0), n_all)
              / np.maximum(np.bincount(self.t_all, use.astype(float), n_all), 1))
        r_size = self.w_size * np.maximum(np.abs(tr) - 0.15, 0.0)
        return np.concatenate([r_road, r_speed, r_dir, r_size])

    def parts(self, p) -> dict:
        """Cost per cue (sum of squares), to see which cue disagrees."""
        r = self.residuals(p)
        cut = np.cumsum([len(self.uv), len(self.tracks), len(self.i0)])
        return {nm: round(float(np.sum(seg ** 2)), 1)
                for nm, seg in zip(("road", "speed", "direction", "size"), np.split(r, cut))}


def fit_to_roads(raw: pd.DataFrame, size, prior: dict, ways: list[np.ndarray],
                 registration: dict | None = None, log=print, radius_m: float = 1500.0,
                 speed_band=(0.2, 40.0), minor_penalty_m: float = 5.0,
                 extra_starts: list[dict] | None = None, clicks=None, click_px: float = 4.0,
                 search: bool = True, terrain=None):
    """Camera from the known position + vehicles-on-roads. Returns (Homography|None, info).

    clicks: optional (pixels, world) point pairs; each counts like `click_px` pixels of
    uncertainty, so a few clicks anchor the position and the vehicles add hundreds of
    constraints on direction, tilt and zoom. search=False only refines extra_starts.

    Search: every direction, tilt and zoom on a small sample of tracks (steps scaled to the
    zoom, so long lenses are not skipped), then the most promising distinct candidates are
    refined on all tracks, with the camera position free within its tolerance.

    terrain: the search runs on the ground level under the camera; the best two solutions
    are then refined with every pixel ray-cast onto the terrain (sloping streets, levels)."""
    from scipy.optimize import least_squares

    from .posefit import pose_homography

    S = vehicle_track_samples(raw, registration)
    n_tracks = int(len(set(S[1])))
    info = {"method": "vehicles on OpenStreetMap roads", "vehicle_tracks": n_tracks}
    if n_tracks < 15:
        info["reason"] = f"only {n_tracks} moving vehicle tracks (need 15+)"
        return None, info
    E, N, Hc = float(prior["E"]), float(prior["N"]), float(prior["height_m"])
    pos_tol = float(prior.get("position_tol_m", 15.0))
    h_tol = float(prior.get("height_tol_m", max(1.0, 0.06 * Hc)))
    z0 = None
    if terrain is not None:
        z0 = float(terrain.height(np.array([[E, N]]))[0])
        if not np.isfinite(z0):
            log("  (the camera is outside the terrain model: flat ground)")
            terrain = None
    road = RoadDistance(ways, E, N, radius_m, minor_penalty_m=minor_penalty_m)
    w, hh = size
    full = RoadObjective(S, road, E, N, size, speed_band)
    small = RoadObjective(vehicle_track_samples(raw, registration, max_tracks=80, per_track=6,
                                                seed=1), road, E, N, size, speed_band)

    # 1. coarse search on the small sample
    fovs = list(np.geomspace(5, 80, 21)) if search else []
    if prior.get("hfov_deg"):
        fovs.append(float(prior["hfov_deg"]))
    cands = []
    for fov in fovs:
        logf = math.log(w / 2 / math.tan(math.radians(fov / 2)))
        vfov = fov * hh / w
        for pitch in np.arange(1.0, 65.0, float(np.clip(vfov / 3, 0.75, 4.0))):
            for yaw in np.arange(0.0, 360.0, float(np.clip(fov / 3, 1.0, 6.0))):
                p = (float(yaw), float(pitch), 0.0, logf, 0.0, 0.0, Hc)
                cands.append((float(np.mean(small.residuals(p) ** 2)), p, fov))
    cands.sort(key=lambda c: c[0])

    # 2. refine distinct candidates on every track
    lo = [-1e9, 0.3, -12, math.log(150), -pos_tol, -pos_tol, max(0.5, Hc - h_tol)]
    hi = [1e9, 75, 12, math.log(60000), pos_tol, pos_tol, Hc + h_tol]
    sig_pos, sig_h = max(pos_tol / 2, 0.1), max(h_tol / 2, 0.1)

    if clicks is not None:
        from .posefit import _project

        c_px, c_world = (np.asarray(a, float).reshape(-1, 2) for a in clicks)
        w_click = math.sqrt(len(full.uv) / max(len(c_px), 1)) / click_px
        c_z = None
        if terrain is not None:          # clicked points at their height on the terrain
            c_z = np.nan_to_num(terrain.height(c_world) - z0)

    def refine(x0, obj=full, nfev=300):
        def r(p):
            prior_r = 3.0 * np.array([p[4] / sig_pos, p[5] / sig_pos, (p[6] - Hc) / sig_h])
            out = [obj.residuals(p), prior_r]
            if clicks is not None:
                z = c_z if obj.terrain is not None else None
                out.append(w_click * (_project(p, c_world, size, np.array([E, N]), z)
                                      - c_px).ravel())
            return np.concatenate(out)
        return least_squares(r, np.clip(np.asarray(x0, float), lo, hi), bounds=(lo, hi),
                             max_nfev=nfev, diff_step=1e-3)

    def distinct(x, others, tol=1.0):
        return not any(abs((x[0] - q[0] + 180) % 360 - 180) < 3 * tol and abs(x[1] - q[1]) < 3 * tol
                       and abs(x[3] - q[3]) < 0.1 * tol for q in others)

    sols = []
    for cam in extra_starts or []:      # e.g. a calibration to test, or a hint
        x0 = [cam["yaw_deg"], cam["tilt_deg"], cam.get("roll_deg", 0.0),
              math.log(cam["focal_px"]), cam["E"] - E, cam["N"] - N, cam["height_m"]]
        info.setdefault("extra_starts", []).append(full.parts(np.clip(x0, lo, hi)))
        sols.append(refine(x0))
    # 2a. quick refinement of many distinct candidates on the small sample
    picked, quick = [], []
    for c, p, fov in (cands if search else []):
        if not distinct(p, [q for q, _ in picked], tol=max(fov / 6, 1.0)):
            continue
        picked.append((p, fov))
        quick.append(refine(p, small, nfev=60))
        if len(picked) >= 40:
            break
    quick.sort(key=lambda s_: s_.cost)
    # 2b. the best few, refined on every track
    done = []
    for q in quick:
        if distinct(q.x, done):
            done.append(q.x)
            sols.append(refine(q.x))
        if len(done) >= 8:
            break
    sols.sort(key=lambda s_: s_.cost)
    if terrain is not None:
        # the best two distinct answers, refined on the terrain itself
        full = RoadObjective(S, road, E, N, size, speed_band, terrain=terrain, z0=z0)
        top = []
        for s_ in sols:
            if distinct(s_.x, [q.x for q in top]):
                top.append(s_)
            if len(top) >= 2:
                break
        sols = sorted((refine(s_.x, full, nfev=80) for s_ in top), key=lambda s_: s_.cost)
        info["terrain"] = True
    p = sols[0].x
    g = full.ground(p, full.uv)
    d = road(np.nan_to_num(g, nan=1e9))
    on_road = float(np.mean(d < 6.0))
    v_tr = np.linalg.norm(g[full.i1] - g[full.i0], axis=1) / full.dts
    # how distinct is the answer? a different solution almost as good = ambiguous
    alt = [s_ for s_ in sols[1:] if abs((s_.x[0] - p[0] + 180) % 360 - 180) > 5
           or abs(s_.x[1] - p[1]) > 3 or abs(s_.x[3] - p[3]) > 0.15]
    margin = (alt[0].cost / max(sols[0].cost, 1e-9)) if alt else float("inf")
    params = {"yaw_deg": float(p[0] % 360), "tilt_deg": float(p[1]), "roll_deg": float(p[2]),
              "focal_px": float(math.exp(p[3])), "E": E + float(p[4]), "N": N + float(p[5]),
              "height_m": float(p[6]),
              "hfov_deg": float(math.degrees(2 * math.atan(w / 2 / math.exp(p[3]))))}
    h = pose_homography(params["E"], params["N"], params["height_m"], params["yaw_deg"],
                        params["tilt_deg"], params["focal_px"], size, roll=params["roll_deg"],
                        origin=np.round([E, N]))
    h.camera_params = params
    if terrain is not None:
        h.set_terrain(terrain, z0)
    h.points = []
    h.rmse_m = float(np.median(d))
    h.max_error_m = float(np.percentile(d, 90))
    h.spread = 1.0
    info.update(on_road_share=round(on_road, 3),
                median_road_distance_m=round(float(np.median(d)), 2),
                median_vehicle_speed_m_s=round(float(np.nanmedian(v_tr)), 2),
                ambiguity_ratio=round(float(margin), 2) if np.isfinite(margin) else None,
                cost_parts=full.parts(p), camera={k: round(v, 3) for k, v in params.items()})
    log(f"  roads: {n_tracks} vehicle tracks, {on_road:.0%} of samples within 6 m of a road, "
        f"looking {params['yaw_deg']:.0f}°, tilt {params['tilt_deg']:.1f}°, field of view "
        f"{params['hfov_deg']:.1f}°")
    return h, info


# --------------------------------------------------------------------------- 3. people
def osm_obstacles(E: float, N: float, radius_m: float = 400.0, timeout: float = 60.0,
                  cache: Path | None = None) -> list[np.ndarray]:
    """Where nobody walks: building footprints and water (docks, rivers) from OpenStreetMap,
    as closed (n, 2) EPSG:27700 rings. Only the search box is sent."""
    from rasterio.warp import transform

    if cache is not None and cache.exists():
        return [np.asarray(r, float) for r in json.loads(cache.read_text(encoding="utf-8"))]
    lon, lat = transform("EPSG:27700", "EPSG:4326",
                         [E - radius_m, E + radius_m], [N - radius_m, N + radius_m])
    bbox = f"{lat[0]:.6f},{lon[0]:.6f},{lat[1]:.6f},{lon[1]:.6f}"
    q = (f'[out:json][timeout:50];(way["building"]({bbox});relation["building"]({bbox});'
         f'way["natural"="water"]({bbox});relation["natural"="water"]({bbox});'
         f'way["waterway"~"^(dock|riverbank)$"]({bbox}););out geom;')
    data = _overpass(q, timeout)
    rings = []
    for el in data.get("elements", []):
        geoms = [el.get("geometry")] if el.get("type") == "way" else \
            [m.get("geometry") for m in el.get("members", []) if m.get("role") == "outer"]
        for g in geoms:
            if not g or len(g) < 4:
                continue
            xs, ys = transform("EPSG:4326", "EPSG:27700", [p["lon"] for p in g],
                               [p["lat"] for p in g])
            rings.append(np.column_stack([xs, ys]))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([r.round(2).tolist() for r in rings]), encoding="utf-8")
    return rings


class ObstacleDepth:
    """How far (m) a ground point lies inside a building or water (0 = walkable)."""

    def __init__(self, rings: list[np.ndarray], E: float, N: float, radius_m: float,
                 res: float = 0.5):
        from scipy.ndimage import distance_transform_edt

        self.x0, self.y1, self.res = E - radius_m, N + radius_m, res
        n = int(2 * radius_m / res) + 1
        self.n = n
        inside = np.zeros((n, n), np.uint8)
        for r in rings:
            pts = np.round(np.column_stack([(r[:, 0] - self.x0) / res,
                                            (self.y1 - r[:, 1]) / res])).astype(np.int32)
            cv2.fillPoly(inside, [pts.reshape(-1, 1, 2)], 1)
        self.d = distance_transform_edt(inside) * res

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        c = (xy[:, 0] - self.x0) / self.res
        r = (self.y1 - xy[:, 1]) / self.res
        out = map_coordinates(self.d, [r, c], order=1, mode="nearest")
        far = (c < 0) | (r < 0) | (c > self.n - 1) | (r > self.n - 1)
        return np.where(far | ~np.isfinite(out), 0.0, out)


def people_samples(raw: pd.DataFrame, registration: dict | None = None, min_box_px: float = 40,
                   max_tracks: int = 300, per_track: int = 10, seed: int = 0):
    """Feet, head tops, track ids and times of clearly visible walking people."""
    from .registration import apply_registration
    from .trajectories import recover_feet

    r = raw.copy()
    r["predicted"] = r["predicted"].astype(str).str.lower().isin(["true", "1"])
    feet, squished = recover_feet(r, 60)
    keep = ((r["class"] == "person") & ~r["predicted"] & ~squished
            & ((r["y2"] - r["y1"]) >= min_box_px) & (r["y1"] > 2)).to_numpy()
    r = r[keep].copy()
    foot = feet[keep]
    top = np.column_stack([(r["x1"] + r["x2"]) / 2, r["y1"]]).astype(float)
    if registration:
        fr = r["frame"].to_numpy()
        foot = apply_registration(fr, foot, registration)
        top = apply_registration(fr, top, registration)
    r["fu"], r["fv"], r["tu"], r["tv"] = foot[:, 0], foot[:, 1], top[:, 0], top[:, 1]
    span = r.groupby("track_id").agg(du=("fu", lambda s: s.max() - s.min()),
                                     dv=("fv", lambda s: s.max() - s.min()), n=("fu", "size"))
    moving = span[(np.hypot(span["du"], span["dv"]) > 30) & (span["n"] >= 8)].index.to_numpy()
    rng = np.random.default_rng(seed)
    if len(moving) > max_tracks:
        moving = rng.choice(moving, max_tracks, replace=False)
    F, T, IDS, TS = [], [], [], []
    for tid in moving:
        t = r[r["track_id"] == tid].sort_values("frame")
        k = np.linspace(0, len(t) - 1, min(per_track, len(t))).round().astype(int)
        F.append(t[["fu", "fv"]].to_numpy(float)[k])
        T.append(t[["tu", "tv"]].to_numpy(float)[k])
        IDS.append(np.full(len(k), tid))
        TS.append(t["time_s"].to_numpy(float)[k])
    if not F:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
    return np.vstack(F), np.vstack(T), np.concatenate(IDS), np.concatenate(TS)


class PeopleObjective:
    """How well a camera explains the walking people.

      obstacles  nobody stands inside a building or in the dock
      height     people come out ~1.70 m tall (per track, robust)
      speed      people walk at ~1.3 m/s (median over tracks; 1.2-1.4 is typical)
    The height fixes tilt and camera height, the speed the zoom/scale, the obstacles the
    direction. Parameters as RoadObjective.
    """

    def __init__(self, samples, obstacles: ObstacleDepth, E, N, size, person_m=PERSON_M,
                 walk_m_s=WALK_M_S):
        self.foot, self.top, ids, ts = samples
        self.obst, self.E, self.N = obstacles, E, N
        self.w, self.h = size
        same = ids[1:] == ids[:-1]
        self.i0 = np.nonzero(same)[0]
        self.i1 = self.i0 + 1
        self.dts = np.maximum(ts[self.i1] - ts[self.i0], 1e-3)
        self.tracks, self.t_pair = np.unique(ids[self.i0], return_inverse=True)
        self.all_tracks, self.t_all = np.unique(ids, return_inverse=True)
        self.person_m, self.walk = person_m, walk_m_s
        n = max(len(self.foot), 1)
        # population constraints (robust to children, lingering, occlusions): the median
        # person is ~1.70 m (+-3 %), the median walker ~1.3 m/s (+-12 %, the weaker cue)
        self.w_height = math.sqrt(n) / 0.03
        self.w_track = 0.5 * math.sqrt(n / max(len(self.all_tracks), 1))
        self.w_speed = math.sqrt(n) / 0.12

    def camera(self, p):
        from .demo import SyntheticCamera

        yaw, pitch, roll, logf, dE, dN, height = p
        return SyntheticCamera(width=self.w, height=self.h, f=math.exp(logf), height_m=height,
                               pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll, cam_local=(dE, dN),
                               origin=np.array([self.E, self.N]))

    def residuals(self, p, robust: float = 1.0) -> np.ndarray:
        cam = self.camera(p)
        P = cam.K @ np.column_stack([cam.R, -cam.R @ cam.C])
        G = P[:, [0, 1, 3]]
        fh = np.column_stack([self.foot, np.ones(len(self.foot))]) @ np.linalg.inv(G).T
        with np.errstate(divide="ignore", invalid="ignore"):
            g = fh[:, :2] / fh[:, 2:3]
        # a ground hit only counts in front of the camera
        fwd = cam.R[2]
        ahead = (g - cam.C[:2]) @ fwd[:2] > 0
        ok = np.isfinite(g).all(axis=1) & ahead
        g = np.where(ok[:, None], g, np.nan)
        # obstacles
        dep = np.where(ok, self.obst(np.nan_to_num(g, nan=1e9)), 5.0)
        r_obst = 2.0 * np.sqrt(2 * robust * (np.sqrt(1 + (dep / robust) ** 2) - 1))
        # height (z such that the point z m above the foot projects onto the head row)
        X = np.column_stack([np.nan_to_num(g, nan=0.0), np.zeros(len(g)), np.ones(len(g))])
        a, c = X @ P[1], X @ P[2]
        b, d = P[1, 2], P[2, 2]
        v = self.top[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (a - v * c) / (v * d - b)
        lz = np.log(np.clip(np.where(ok & (z > 0), z, np.nan), 0.3, 6.0) / self.person_m)
        n_all = len(self.all_tracks)
        use = np.isfinite(lz)
        cnt = np.bincount(self.t_all, use.astype(float), n_all)
        tr = np.bincount(self.t_all, np.where(use, lz, 0.0), n_all) / np.maximum(cnt, 1)
        tr = np.where(cnt > 0, tr, 0.5)
        med_h = float(np.median(tr[cnt > 0])) if (cnt > 0).any() else 0.5
        r_height = np.concatenate([[self.w_height * med_h],          # the median person
                                   self.w_track * np.sqrt(2 * 0.1 * (np.sqrt(1 + (tr / 0.1) ** 2)
                                                                    - 1))])
        # speed: the median walking speed of the tracks
        step = np.linalg.norm(g[self.i1] - g[self.i0], axis=1) / self.dts
        k = len(self.tracks)
        good = np.isfinite(step)
        vs = np.exp(np.bincount(self.t_pair, np.where(good, np.log(np.maximum(step, 1e-3)), 0), k)
                    / np.maximum(np.bincount(self.t_pair, good.astype(float), k), 1))
        med = float(np.median(vs)) if len(vs) else self.walk
        r_speed = np.array([self.w_speed * math.log(max(med, 1e-3) / self.walk)])
        return np.concatenate([r_obst, r_height, r_speed])

    def parts(self, p) -> dict:
        r = self.residuals(p)
        cut = np.cumsum([len(self.foot), len(self.all_tracks) + 1])
        return {nm: round(float(np.sum(seg ** 2)), 1)
                for nm, seg in zip(("obstacles", "height", "speed"), np.split(r, cut))}


def fit_to_people(raw: pd.DataFrame, size, prior: dict, obstacles: list[np.ndarray],
                  registration: dict | None = None, log=print, radius_m: float = 400.0,
                  clicks=None, click_px: float = 4.0, extra_starts: list[dict] | None = None):
    """Camera from the known position + walking people. Returns (Homography|None, info)."""
    from scipy.optimize import least_squares

    from .posefit import _project, pose_homography

    S = people_samples(raw, registration)
    n_tracks = int(len(set(S[2])))
    info = {"method": "people on walkable ground", "people_tracks": n_tracks}
    if n_tracks < 12:
        info["reason"] = f"only {n_tracks} clearly visible walking people (need 12+)"
        return None, info
    E, N, Hc = float(prior["E"]), float(prior["N"]), float(prior["height_m"])
    pos_tol = float(prior.get("position_tol_m", 10.0))
    h_tol = float(prior.get("height_tol_m", max(0.5, 0.2 * Hc)))
    obst = ObstacleDepth(obstacles, E, N, radius_m)
    w, hh = size
    full = PeopleObjective(S, obst, E, N, size)
    small = PeopleObjective(people_samples(raw, registration, max_tracks=60, per_track=5, seed=1),
                            obst, E, N, size)
    lo = [-1e9, -15.0, -12, math.log(150), -pos_tol, -pos_tol, max(0.3, Hc - h_tol)]
    hi = [1e9, 60.0, 12, math.log(20000), pos_tol, pos_tol, Hc + h_tol]
    sig_pos, sig_h = max(pos_tol / 2, 0.1), max(h_tol / 2, 0.1)
    if clicks is not None:
        c_px, c_world = (np.asarray(a, float).reshape(-1, 2) for a in clicks)
        w_click = math.sqrt(len(full.foot) / max(len(c_px), 1)) / click_px

    f_prior = (math.log(w / 2 / math.tan(math.radians(float(prior["hfov_deg"]) / 2)))
               if prior.get("hfov_deg") else None)
    w_f = math.sqrt(len(full.foot)) / 0.05            # lens known to ~5 %

    def refine(x0, obj=full, nfev=300):
        def r(p):
            out = [obj.residuals(p),
                   3.0 * np.array([p[4] / sig_pos, p[5] / sig_pos, (p[6] - Hc) / sig_h])]
            if f_prior is not None:
                out.append(np.array([w_f * (p[3] - f_prior)]))
            if clicks is not None:
                out.append(w_click * (_project(p, c_world, size, np.array([E, N])) - c_px).ravel())
            return np.concatenate(out)
        return least_squares(r, np.clip(np.asarray(x0, float), lo, hi), bounds=(lo, hi),
                             max_nfev=nfev, diff_step=1e-3)

    # coarse search: every direction, tilt and zoom (phone lenses: ~15-110 deg)
    fovs = list(np.geomspace(15, 110, 14))
    if prior.get("hfov_deg"):
        fovs.append(float(prior["hfov_deg"]))
    cands = []
    for fov in fovs:
        logf = math.log(w / 2 / math.tan(math.radians(fov / 2)))
        vfov = 2 * math.degrees(math.atan(hh / 2 / math.exp(logf)))
        for pitch in np.arange(-5.0, 35.0, float(np.clip(vfov / 4, 1.0, 4.0))):
            for yaw in np.arange(0.0, 360.0, float(np.clip(fov / 4, 2.0, 8.0))):
                p = (float(yaw), float(pitch), 0.0, logf, 0.0, 0.0, Hc)
                cands.append((float(np.mean(small.residuals(p) ** 2)), p, fov))
    cands.sort(key=lambda c: c[0])

    def distinct(x, others, tol=1.0):
        return not any(abs((x[0] - q[0] + 180) % 360 - 180) < 4 * tol and abs(x[1] - q[1]) < 3 * tol
                       and abs(x[3] - q[3]) < 0.1 * tol for q in others)

    sols = []
    for cam in extra_starts or []:
        x0 = [cam["yaw_deg"], cam["tilt_deg"], cam.get("roll_deg", 0.0),
              math.log(cam["focal_px"]), cam["E"] - E, cam["N"] - N, cam["height_m"]]
        info.setdefault("extra_starts", []).append(full.parts(np.clip(x0, lo, hi)))
        sols.append(refine(x0))
    picked, quick = [], []
    for c, p, fov in cands:
        if not distinct(p, [q for q, _ in picked], tol=max(fov / 8, 1.0)):
            continue
        picked.append((p, fov))
        quick.append(refine(p, small, nfev=60))
        if len(picked) >= 40:
            break
    quick.sort(key=lambda s_: s_.cost)
    done = []
    for q in quick:
        if distinct(q.x, done):
            done.append(q.x)
            sols.append(refine(q.x))
        if len(done) >= 8:
            break
    sols.sort(key=lambda s_: s_.cost)
    p = sols[0].x
    params = {"yaw_deg": float(p[0] % 360), "tilt_deg": float(p[1]), "roll_deg": float(p[2]),
              "focal_px": float(math.exp(p[3])), "E": E + float(p[4]), "N": N + float(p[5]),
              "height_m": float(p[6]),
              "hfov_deg": float(math.degrees(2 * math.atan(w / 2 / math.exp(p[3]))))}
    h = pose_homography(params["E"], params["N"], params["height_m"], params["yaw_deg"],
                        params["tilt_deg"], params["focal_px"], size, roll=params["roll_deg"],
                        origin=np.round([E, N]))
    h.camera_params = params
    h.points = []
    h.spread = 1.0
    parts = full.parts(p)
    g = h.to_world(S[0])
    inside = float(np.mean(obst(np.nan_to_num(g, nan=1e9)) > 0.5))
    h.rmse_m = h.max_error_m = float("nan")
    info.update(cost_parts=parts, inside_obstacles_share=round(inside, 3),
                camera={k: round(v, 3) for k, v in params.items()})
    log(f"  people: {n_tracks} walking people, {inside:.0%} of foot points inside buildings or "
        f"water, looking {params['yaw_deg']:.0f}°, tilt {params['tilt_deg']:.1f}°, field of view "
        f"{params['hfov_deg']:.1f}°, {params['height_m']:.1f} m up")
    return h, info


# --------------------------------------------------------------------------- 4. checks
def quality_checks(h: Homography, raw: pd.DataFrame, cfg, registration: dict | None = None,
                   ways: list[np.ndarray] | None = None, people: bool = True) -> dict:
    """Does this calibration agree with the footage? Returns {checks: [...], verdict}."""
    from .selfcal import people_check
    from .trajectories import max_range_for

    checks = []
    if people and "person" in cfg.classes and (raw["class"] == "person").sum() > 30:
        chk = people_check(raw, h, registration, max_range_for(cfg, h, "people"))
        if chk:
            err = chk["scale_error_pct"]
            checks.append({"name": "people height", "value": f"{chk['implied_person_height_m']} m",
                           "ok": abs(err) <= 12, "note": f"{err:+.0f}% vs 1.70 m"})
    veh = raw["class"].isin(["car", "bus", "truck", "motorcycle"])
    if veh.sum() > 30 and ways:
        uv = vehicle_track_samples(raw, registration)[0]
        cam = h.camera_params or h.camera() or {}
        if len(uv) and cam:
            r = 1500.0
            road = RoadDistance(ways, cam["E"], cam["N"], r)
            g = h.to_world(uv)
            d = road(np.nan_to_num(g, nan=1e9))
            share = float(np.mean(d < 6.0))
            checks.append({"name": "vehicles on roads", "value": f"{share:.0%}",
                           "ok": share >= 0.7, "note": "share of vehicle positions within 6 m "
                                                        "of an OpenStreetMap road"})
    verdict = "ok" if checks and all(c["ok"] for c in checks) else (
        "check" if checks else "no checks possible")
    return {"checks": checks, "verdict": verdict}


# --------------------------------------------------------------------------- 5. lanes
class LaneField:
    """Direction of the painted lines / kerbs in the aerial (structure tensor), per ground
    point: (direction in degrees from +E, 0-180; coherence 0-1, high on clear lines)."""

    def __init__(self, raster, ext, res: float = 0.25, line_sigma_m: float = 0.3,
                 window_m: float = 3.0):
        from scipy.ndimage import map_coordinates  # noqa: F401  (used in sample)

        x0, x1, y0, y1 = ext
        self.x0, self.y1, self.res = x0, y1, res
        w, h = max(int((x1 - x0) / res), 2), max(int((y1 - y0) / res), 2)
        c0, r0 = raster.world_to_pixel([[x0, y1]])[0]
        c1, r1 = raster.world_to_pixel([[x1, y0]])[0]
        M = np.array([[(c1 - c0) / w, 0, c0], [0, (r1 - r0) / h, r0]], np.float32)
        gray = raster.image if raster.image.ndim == 2 else \
            cv2.cvtColor(raster.image, cv2.COLOR_RGB2GRAY)
        g = cv2.warpAffine(gray, M, (w, h),
                           flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP).astype(np.float32)
        # Gaussian derivatives: direction-unbiased (Sobel is off by up to ~0.8 deg between
        # the 45 deg multiples, as large as the effects this measures)
        from scipy.ndimage import gaussian_filter

        sg = max(line_sigma_m / res, 1.0)
        gx = gaussian_filter(g, sg, order=(0, 1))
        gy = gaussian_filter(g, sg, order=(1, 0))
        s = window_m / res
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), s)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), s)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), s)
        coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
        line = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2       # image coords, y down
        angle = np.degrees(np.arctan2(-np.sin(line), np.cos(line))) % 180
        t2 = np.radians(2 * angle)                    # doubled angle: interpolates cleanly
        self.c2 = (np.cos(t2) * coherence).astype(np.float32)
        self.s2 = (np.sin(t2) * coherence).astype(np.float32)
        self.shape = g.shape

    def sample(self, xy):
        from scipy.ndimage import map_coordinates

        c = (xy[:, 0] - self.x0) / self.res
        r = (self.y1 - xy[:, 1]) / self.res
        ok = (c >= 0) & (r >= 0) & (c < self.shape[1] - 1) & (r < self.shape[0] - 1)
        rr, cc = np.where(ok, r, 0), np.where(ok, c, 0)
        c2 = map_coordinates(self.c2, [rr, cc], order=1)
        s2 = map_coordinates(self.s2, [rr, cc], order=1)
        return (np.where(ok, np.degrees(0.5 * np.arctan2(s2, c2)) % 180, np.nan),
                np.where(ok, np.hypot(c2, s2), 0.0))


def lane_check(points: pd.DataFrame, raster, min_len_m: float = 25.0) -> dict | None:
    """Real-footage check against the aerial: do straight vehicle tracks run parallel to the
    painted lanes (a calibration rotation shows as one sign), and how much do they scatter
    sideways around a straight line (an upper bound on position noise)?"""
    p = points[(points["group"] == "vehicles") & ~points["predicted"].astype(bool)]
    if p.empty or raster is None:
        return None
    rows = []
    for _, t in p.groupby("track_id"):
        xy = t.sort_values("frame")[["x", "y"]].to_numpy(float)
        if len(xy) < 10:
            continue
        d = xy[-1] - xy[0]
        L = float(np.hypot(*d))
        path = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
        if L < min_len_m or L / max(path, 1e-6) < 0.97:
            continue
        c = xy.mean(axis=0)
        vt = np.linalg.svd(xy - c, full_matrices=False)[2]
        rows.append((xy, d, float(np.sqrt(np.mean(((xy - c) @ vt[1]) ** 2)))))
    if len(rows) < 5:
        return None
    allxy = np.vstack([r[0] for r in rows])
    ext = (allxy[:, 0].min() - 20, allxy[:, 0].max() + 20,
           allxy[:, 1].min() - 20, allxy[:, 1].max() + 20)
    field = LaneField(raster, ext)
    ang = []
    for xy, d, _ in rows:
        a, coh = field.sample(xy)
        ok = coh > 0.5
        if ok.sum() < 3:
            continue
        t2 = np.radians(2 * a[ok])
        lane = math.degrees(0.5 * math.atan2(np.sin(t2).mean(), np.cos(t2).mean())) % 180
        ang.append((math.degrees(math.atan2(d[1], d[0])) % 180 - lane + 90) % 180 - 90)
    if len(ang) < 5:
        return None
    ang = np.array(ang)
    scatter = np.array([r[2] for r in rows])
    return {"straight_vehicle_tracks": int(len(ang)),
            "rotation_vs_painted_lanes_deg": round(float(np.median(ang)), 2),
            "abs_angle_vs_lanes_deg": round(float(np.median(np.abs(ang))), 2),
            "sideways_scatter_cm": round(float(100 * np.median(scatter)), 1),
            "note": "tracks vs the lane markings in the aerial; a calibration rotation shows as "
                    "a consistent sign. Sideways scatter of straight tracks is an upper bound "
                    "on position noise."}


# --------------------------------------------------------------------------- orchestration
def _latest_run(cfg) -> Path | None:
    from .layout import Run

    root = (cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site
    runs = sorted((p for p in root.glob("*") if Run(p).is_run()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _rail_cache(out: Path, prior: dict) -> Path:
    return out.parent / (f"osm_rails_visible_{round(prior['E'], -2):.0f}_"
                         f"{round(prior['N'], -2):.0f}.json")


def osm_rails(E: float, N: float, radius_m: float = 1500.0, timeout: float = 60.0,
              cache: Path | None = None) -> list[np.ndarray]:
    """Rail lines a camera can see (rail, light rail, metro, tram; not in tunnels or below
    ground) around (E, N), from OpenStreetMap."""
    from rasterio.warp import transform

    if cache is not None and cache.exists():
        return [np.asarray(w, float) for w in json.loads(cache.read_text(encoding="utf-8"))]
    lon, lat = transform("EPSG:27700", "EPSG:4326",
                         [E - radius_m, E + radius_m], [N - radius_m, N + radius_m])
    bbox = f"{lat[0]:.6f},{lon[0]:.6f},{lat[1]:.6f},{lon[1]:.6f}"
    q = (f'[out:json][timeout:50];way["railway"~"^(rail|light_rail|subway|tram|narrow_gauge)$"]'
         f'["tunnel"!~"."]({bbox});out geom tags;')
    data = _overpass(q, timeout)
    ways = []
    for el in data.get("elements", []):
        g = el.get("geometry") or []
        try:
            below = float(str((el.get("tags") or {}).get("layer", "0")).split(";")[0]) < 0
        except ValueError:
            below = False
        if len(g) >= 2 and not below:
            xs, ys = transform("EPSG:4326", "EPSG:27700", [p["lon"] for p in g],
                               [p["lat"] for p in g])
            ways.append(np.column_stack([xs, ys]))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([w.round(2).tolist() for w in ways]), encoding="utf-8")
    return ways


def _obstacle_cache(out: Path, prior: dict) -> Path:
    return out.parent / (f"osm_obstacles_{round(prior['E'], -2):.0f}_"
                         f"{round(prior['N'], -2):.0f}.json")


def _attach_terrain(h: Homography, terrain, log=print) -> None:
    """A camera fitted on one flat ground (roads, people, a flat calibration): project onto
    the terrain from now on. Its plane is taken as the ground it saw (the median terrain
    height under the lower half of the frame)."""
    w, hh = h.image_size
    uu, vv = np.meshgrid(np.linspace(0.05, 0.95, 20) * w, np.linspace(0.55, 0.95, 10) * hh)
    g = h.plane_to_world(np.column_stack([uu.ravel(), vv.ravel()]))
    z = terrain.height(g[np.isfinite(g).all(axis=1)])
    if np.isfinite(z).any():
        h.set_terrain(terrain, float(np.nanmedian(z)))
        log(f"  ground: terrain model, this fit's flat ground taken at {h.ground_z_m:.1f} m")


def _road_cache(out: Path, prior: dict) -> Path:
    """One road download per camera spot (100 m grid), shared by every clip filmed there."""
    return out.parent / f"osm_roads_{round(prior['E'], -2):.0f}_{round(prior['N'], -2):.0f}.json"


def best_by_cross_validation(raw, size, prior: dict, clicks, reg, out: Path, log=print,
                             pooled: dict | None = None, terrain=None):
    """With 4+ clicked points: fit every method that applies, keep the one that predicts the
    held-out clicks best (leave-one-out). Returns (homography, {method: median error m}).

    pooled: {site: (px, world)} clicks of other clips filmed from the same spot, already
    mapped into this frame; they are only ever used for fitting, never as held-out points."""
    from .posefit import camera_calibration

    px, world = (np.asarray(a, float).reshape(-1, 2) for a in clicks)
    quiet = lambda *_: None  # noqa: E731

    def clicked(p, w):
        return camera_calibration(p, w, size, prior, terrain=terrain)

    methods = {"clicked points": clicked}
    if pooled:
        pp = np.vstack([v[0] for v in pooled.values()])
        pw = np.vstack([v[1] for v in pooled.values()])
        names = ", ".join(pooled)
        methods[f"clicked points + {len(pp)} clicks of {names} (same spot)"] = \
            lambda p, w: clicked(np.vstack([p, pp]), np.vstack([w, pw]))
    if raw["class"].isin(["car", "bus", "truck"]).sum() >= 100:
        ways = osm_roads(prior["E"], prior["N"], cache=_road_cache(out, prior))
        methods["clicked points + vehicles on roads"] = lambda p, w: fit_to_roads(
            raw, size, prior, ways, reg, log=quiet, clicks=(p, w), search=False,
            extra_starts=[clicked(p, w).camera_params], terrain=terrain)[0]
    if terrain is not None:
        log("  (terrain model: the walking-people fit assumes one flat ground, not compared)")
    if terrain is None and (raw["class"] == "person").sum() >= 300:
        rings = osm_obstacles(prior["E"], prior["N"], cache=_obstacle_cache(out, prior))
        methods["clicked points + walking people"] = lambda p, w: fit_to_people(
            raw, size, prior, rings, reg, log=quiet, clicks=(p, w),
            extra_starts=[clicked(p, w).camera_params])[0]
    scores = {}
    for name, fit in methods.items():
        errs = []
        for k in range(len(px)):
            keep = np.arange(len(px)) != k
            try:
                hk = fit(px[keep], world[keep])
                e = float(np.linalg.norm(hk.to_world(px[k:k + 1])[0] - world[k]))
            except Exception:      # a method that cannot fit counts as a miss
                e = float("nan")
            errs.append(e if np.isfinite(e) else 50.0)          # beyond the horizon = miss
        scores[name] = round(float(np.median(errs)), 3)
        log(f"  {name}: a hidden clicked point lands {scores[name]:.2f} m off (median)")
    best = min(scores, key=scores.get)
    h = methods[best](px, world)
    h.method = f"{best} (best of {len(scores)} by leave-one-out)"
    h.method_info = {"leave_one_out_median_m": scores}
    return h, scores


def run_autocalibration(cfg, log=print, run_dir: Path | None = None, device: str = "auto",
                        replace: bool = False, track_frames: int = 900) -> dict:
    """Calibrate a site without clicking, as far as the footage allows. Returns the report.

    1. same spot: another site config in the same folder, already calibrated and filmed
       from the same position -> exact transfer.
    2. vehicles on roads (needs camera_position and moving vehicles): fit the camera to
       OpenStreetMap roads; clicked points, if there are any, are used as well.
    The result goes to the configured homography file if there is none yet (or `replace`),
    otherwise next to it as <name>_auto.json, so a hand calibration is never overwritten.
    """
    from .config import load_config
    from .layout import open_run
    from .pipeline import new_run_dir, stage_track
    from .registration import load_registration
    from .video import read_frame

    out = cfg.path("homography")
    if out is None:
        raise ValueError("Set `homography:` (output path) in the site config")
    prior = cfg.get("camera_position")
    report: dict = {"site": cfg.site, "tried": []}
    from .terrain import resolve_terrain

    terrain = resolve_terrain(cfg, log)
    video = cfg.path("video")
    frame_idx = Homography.load(out).reference_frame if out.exists() else 0
    frame = read_frame(video, frame_idx)
    size = (frame.shape[1], frame.shape[0])
    h, raw, reg, ways = None, None, None, None

    # 1. same spot ---------------------------------------------------------------------
    best = None
    same_spot_clicks: dict = {}
    for other in sorted(cfg.base_dir.glob("*.yaml")):
        try:
            oc = load_config(other)
        except Exception:
            continue
        op, ov = oc.path("homography"), oc.path("video")
        if oc.site == cfg.site or op is None or not op.exists() or op == out \
                or ov is None or not ov.exists():
            continue
        ocam = oc.get("camera_position")
        if prior and ocam and (math.hypot(ocam["E"] - prior["E"], ocam["N"] - prior["N"]) > 30
                               or abs(ocam["height_m"] - prior["height_m"]) > 10):
            continue                                  # a different camera position
        hs = Homography.load(op)
        src_frame = read_frame(ov, hs.reference_frame)
        hh, info = transfer(hs, src_frame, frame, prior, log=lambda *_: None)
        info["from"] = oc.site
        report["tried"].append(info)
        if hh is not None and (best is None or info["inliers"] > best[1]["inliers"]):
            best = (hh, info)
        # its clicked points, seen in this frame (for pooling when this clip has clicks too)
        o_csv = op.with_name(op.stem + "_points.csv")
        if hh is not None and info["inliers"] >= 100 and o_csv.exists():
            from .calibrate import load_points_csv

            Hm, _, _ = match_frames(frame, src_frame)          # this frame -> other frame
            if Hm is not None:
                opx, ow = load_points_csv(o_csv)
                q = cv2.perspectiveTransform(opx.reshape(-1, 1, 2).astype(np.float64),
                                             np.linalg.inv(Hm)).reshape(-1, 2)
                inside = (q[:, 0] >= 0) & (q[:, 0] < size[0]) & (q[:, 1] >= 0) & (q[:, 1] < size[1])
                if inside.sum() >= 2:
                    same_spot_clicks[oc.site] = (q[inside], ow[inside])
    own_csv = out.with_name(out.stem + "_points.csv")
    own_clicks = 0
    if own_csv.exists():
        from .calibrate import load_points_csv

        own_clicks = len(load_points_csv(own_csv)[0])
    if own_clicks >= 4:
        best = None            # its own clicks are evidence: pool, don't replace (below)
    if best and best[1]["inliers"] >= 100 and best[1]["inlier_ratio"] >= 0.3:
        h, chosen = best
        h.method = f"same spot (from {chosen['from']})"
        h.method_info = chosen
        log(f"same spot as {chosen['from']}: {chosen['inliers']} matching features -> "
            "calibration transferred (exact geometry)")
    elif best:
        log(f"same spot as {best[1]['from']}? only {best[1]['inliers']} matches: not trusted")

    # tracks: needed for the road fit and for the checks
    rd = run_dir or _latest_run(cfg)
    if rd is None and h is None:
        log(f"no tracked run yet: tracking the first {track_frames} frames ...")
        from .device import resolve_device

        rd = new_run_dir(cfg, "autocal")
        stage_track(cfg, rd, resolve_device(device), log=log, max_frames=track_frames)
    if rd is not None:
        run = open_run(rd)
        raw = pd.read_csv(run.raw_tracks)
        if run.registration.exists():
            reg, ref, _ = load_registration(run.registration)
            if h is None:
                frame_idx = ref                      # the tracks are aligned to that frame
    n_veh = 0 if raw is None else int(raw["class"].isin(["car", "bus", "truck"]).sum())
    n_ppl = 0 if raw is None else int((raw["class"] == "person").sum())
    pts_csv = out.with_name(out.stem + "_points.csv")

    # with 4+ clicked points: measure which method predicts unseen clicks best
    from .calibrate import load_points_csv

    clicks_all = load_points_csv(pts_csv) if pts_csv.exists() else None
    if h is None and prior and raw is not None and clicks_all is not None \
            and len(clicks_all[0]) >= 4:
        log(f"{len(clicks_all[0])} clicked points: comparing methods by leaving each point out")
        h, scores = best_by_cross_validation(raw, size, prior, clicks_all, reg, out, log=log,
                                             pooled=same_spot_clicks or None, terrain=terrain)
        report["cross_validation"] = scores
        if h.method.startswith("clicked points (") and out.exists():
            log("your clicked calibration is already the best of these: nothing to change")
            report["result"] = str(out)
            report["best"] = "clicked points"
            return report

    # 2. vehicles on roads -------------------------------------------------------------
    if h is None:
        if not prior:
            report["reason"] = ('needs the camera position first: motus locate '
                                '"<building>" --floor N -c <site>.yaml --write')
        elif terrain is not None and n_veh < 100:
            # the walking-people fit puts everyone on one flat ground
            report["reason"] = ("with a terrain model the walking-people fit (one flat ground) "
                                "is not used, and there is too little traffic for the road "
                                "fit: pick 4+ points (motus calibrate), on any level")
        elif n_veh < 100 and n_ppl >= 300 and float(prior["height_m"]) < 2.5:
            # tested: from eye height everyone's head sits on the horizon, which says almost
            # nothing about the tilt; the people cues then made the fit worse on one of two clips
            report["reason"] = ("camera at eye height: the walking people can't fix the tilt "
                                "from down there. Pick 5+ well-spread points (motus "
                                "calibrate), or film from a raised spot (steps, a wall, a "
                                "first-floor window) next time")
        elif n_veh < 100 and n_ppl >= 300:
            # 3. walking people: height and speed fix tilt, camera height and zoom; on an open
            # plaza only buildings / water could fix the direction, so 2 clicks do that
            from .calibrate import load_points_csv

            clicks = load_points_csv(pts_csv) if pts_csv.exists() else None
            rings = osm_obstacles(prior["E"], prior["N"], cache=_obstacle_cache(out, prior))
            starts = []
            if out.exists() and getattr(Homography.load(out), "camera_params", None):
                starts.append(Homography.load(out).camera_params)
            hp, info = fit_to_people(raw, size, prior, rings, reg, log=log, clicks=clicks,
                                     extra_starts=starts)
            report["tried"].append(info)
            if hp is not None and clicks is not None and len(clicks[0]) >= 2:
                h = hp
                h.method = f"walking people + {len(clicks[0])} clicked points"
                h.method_info = {k: v for k, v in info.items() if k != "extra_starts"}
            elif hp is not None:
                c = info["camera"]
                report["reason"] = (
                    f"the walking people give tilt {c['tilt_deg']:.1f}°, field of view "
                    f"{c['hfov_deg']:.0f}° and camera height {c['height_m']:.1f} m, but on an "
                    "open plaza nothing in the footage fixes the direction: click 2 points "
                    "(motus calibrate, then Enter twice) and run autocalibrate again")
            else:
                report["reason"] = info.get("reason", "not enough people")
        elif n_veh < 100:
            report["reason"] = ("not enough moving vehicles or walking people in the tracks: "
                                "pick 3-4 points (motus calibrate)")
        else:
            from .calibrate import load_points_csv

            ways = osm_roads(prior["E"], prior["N"], cache=_road_cache(out, prior))
            clicks = load_points_csv(pts_csv) if pts_csv.exists() else None
            starts = []
            if out.exists() and getattr(Homography.load(out), "camera_params", None):
                starts.append(Homography.load(out).camera_params)
            h, info = fit_to_roads(raw, size, prior, ways, reg, log=log, clicks=clicks,
                                   extra_starts=starts, terrain=terrain)
            info["with_clicks"] = clicks is not None
            report["tried"].append(info)
            if h is not None:
                h.method = "vehicles on roads" + (" + clicked points" if clicks is not None
                                                  else "")
                h.method_info = {k: v for k, v in info.items() if k != "extra_starts"}
            else:
                report["reason"] = (info.get("reason", "the road fit failed")
                                    + ": pick 4+ points (motus calibrate)")
    if h is None:
        log("automatic calibration not possible: " + report.get("reason", "see the report"))
        report["result"] = None
        return report

    if terrain is not None and h.terrain is None:
        _attach_terrain(h, terrain, log)

    # 4. checks ------------------------------------------------------------------------
    if raw is not None:
        if ways is None and prior and n_veh > 100:
            try:
                ways = osm_roads(prior["E"], prior["N"], cache=_road_cache(out, prior))
            except OSError as e:
                log(f"  (OpenStreetMap roads unavailable: {e}; road check skipped)")
                ways = None
        report["quality"] = quality_checks(h, raw, cfg, reg, ways)
        if not report["quality"]["checks"]:
            log("  no automatic check possible for this clip: look at the check image")
        for c in report["quality"]["checks"]:
            log(f"  check {c['name']}: {c['value']} ({c['note']}) -> "
                + ("ok" if c["ok"] else "CHECK"))
    h.reference_frame = int(frame_idx)
    h.image_size = size
    target = out if (replace or not out.exists()) else out.with_name(out.stem + "_auto.json")
    if replace and out.exists():
        import shutil

        shutil.copy2(out, out.with_name(out.stem + ".before_auto.json"))
    h.save(target)
    try:
        from .geo import load_geotiff
        from .overlay_check import map_in_video

        mp, ins = map_in_video(frame, h, load_geotiff(cfg.path("geotiff")))
        blend = frame.copy()
        blend[ins] = (0.5 * frame[ins] + 0.5 * mp[ins]).astype(np.uint8)
        if raw is not None:                  # people and number plates blurred
            from .debug_video import blur_box, privacy_boxes, raw_all

            dv = cfg.get("debug_video") or {}
            hide = privacy_boxes(raw_all(rd, raw), bool(dv.get("blur_people", True)),
                                 bool(dv.get("blur_plates", True)))
            for r in hide[hide["frame"] == int(frame_idx)].itertuples():
                blur_box(blend, r.x1, r.y1, r.x2, r.y2)
        cv2.imwrite(str(target.with_name(target.stem + "_check_video.jpg")), blend)
    except Exception as e:  # the check image is a convenience
        log(f"  (check image skipped: {e})")
    report["result"] = str(target)
    report["camera"] = getattr(h, "camera_params", None)
    target.with_name(target.stem + "_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"saved {target}" + ("" if target == out else
                             f"\n  (your calibration {out.name} is kept; compare, then rerun "
                             "with --replace to use the automatic one)"))
    return report
