"""Local web interface: `groundtrack ui`.

A small server on 127.0.0.1 (not reachable from other computers) and one page: pick or
create a site, set where the camera stood (OpenStreetMap search), open the calibration
picker / ROI drawing, start runs with a live log, and browse results and packages.

Every action runs the normal `groundtrack` command in a subprocess, so the command line
and the UI always do exactly the same thing. Nothing leaves the computer except the text
typed into the place search (sent to OpenStreetMap's Nominatim).
"""

from __future__ import annotations

import io
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from .layout import Run, open_run

HTML = Path(__file__).with_name("ui.html")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
IMG_EXT = {".png", ".jpg", ".jpeg"}
VID_EXT = {".mp4"}


# --------------------------------------------------------------------------- jobs
class Job:
    """One `python -m groundtrack ...` subprocess with a captured log."""

    def __init__(self, kind: str, args: list[str], cwd: Path, label: str):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.args, self.label = kind, args, label
        self.lines: list[str] = []
        self.progress = ""
        self.started, self.ended = time.time(), None
        self.returncode: int | None = None
        self.cancelled = False
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-m", "groundtrack", *args], cwd=str(cwd),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = ""
        while True:
            chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") \
                else self.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while True:  # tqdm redraws with \r: keep only its latest state as `progress`
                m = re.search(r"[\r\n]", buf)
                if not m:
                    break
                part, sep, buf = buf[:m.start()], m.group(), buf[m.end():]
                if sep == "\r":
                    if part.strip():
                        self.progress = part.strip()
                elif part.strip() or self.lines:
                    if "Warning" not in part:
                        self.lines.append(part.rstrip())
                    self.progress = ""
            del self.lines[:-3000]
        if buf.strip():
            self.lines.append(buf.rstrip())
        self.returncode = self.proc.wait()
        self.ended = time.time()
        self.progress = ""

    @property
    def status(self) -> str:
        if self.returncode is None:
            return "running"
        if self.cancelled:
            return "stopped"
        return "done" if self.returncode == 0 else "failed"

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
                "started": self.started, "ended": self.ended, "returncode": self.returncode,
                "last": self.progress or (self.lines[-1] if self.lines else "")}

    def stop(self):
        if self.returncode is None:
            self.cancelled = True
            self.proc.terminate()


# --------------------------------------------------------------------------- state
class App:
    job_class = Job       # the desktop app runs its jobs without a console window

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.jobs: dict[str, Job] = {}
        self._thumbs: dict[tuple, bytes] = {}

    # ---- sites
    def config_files(self) -> list[Path]:
        files = sorted((self.root / "sites").glob("*.yaml"))
        if not files:  # a fresh checkout: show the examples until there are real sites
            files = sorted((self.root / "examples").glob("*.yaml"))
        bases = set()
        for f in files:
            try:
                ext = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("extends")
            except (OSError, yaml.YAMLError):
                continue
            if ext:
                bases.add((f.parent / ext).resolve())
        return [f for f in files if f.resolve() not in bases]

    def resolve_config(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p not in [f.resolve() for f in self.config_files()]:
            raise ValueError(f"unknown config {rel!r}")
        return p

    def rel(self, p: Path | str | None) -> str | None:
        if p is None:
            return None
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    def allowed_roots(self) -> list[Path]:
        roots = {self.root}
        for f in self.config_files():
            try:
                cfg = _load(f)
            except Exception:
                continue
            for key in ("output_dir", "homography"):
                q = cfg.path(key)
                if q is not None:
                    roots.add((q if key == "output_dir" else q.parent).resolve())
        return list(roots)

    def sites(self) -> list[dict]:
        out = []
        for f in self.config_files():
            item = {"config": self.rel(f), "name": f.stem}
            try:
                cfg = _load(f)
                item["name"] = cfg.site
                item["video_ok"] = bool(cfg.path("video") and cfg.path("video").exists())
                item["map_ok"] = bool(cfg.path("geotiff") and cfg.path("geotiff").exists())
                item["calibrated"] = bool(cfg.path("homography") and cfg.path("homography").exists())
                item["camera"] = bool(cfg.get("camera_position"))
                rd = _run_root(cfg)
                item["runs"] = len([p for p in rd.glob("*") if Run(p).is_run()]) \
                    if rd.exists() else 0
            except Exception as e:
                item["error"] = str(e)
            out.append(item)
        return out

    def site(self, rel: str) -> dict:
        from .homography import Homography

        f = self.resolve_config(rel)
        cfg = _load(f)
        d: dict = {"config": self.rel(f), "name": cfg.site, "config_path": str(f)}
        v = cfg.path("video")
        d["video"] = {"path": str(v) if v else None, "ok": bool(v and v.exists())}
        if v and v.exists():
            try:
                from .video import video_info

                i = video_info(v, cfg.get("fps"))
                d["video"].update(width=i.width, height=i.height, fps=round(i.fps, 3),
                                  duration_s=round(i.duration_s, 1))
            except Exception as e:
                d["video"]["error"] = str(e)
        g = cfg.path("geotiff")
        d["map"] = {"path": str(g) if g else None, "ok": bool(g and g.exists())}
        d["camera_position"] = cfg.get("camera_position")
        from .detect import site_roi

        try:
            d["roi"] = site_roi(cfg) is not None
        except Exception:
            d["roi"] = False
        hp = cfg.path("homography")
        cal = {"path": str(hp) if hp else None, "ok": bool(hp and hp.exists())}
        if cal["ok"]:
            try:
                h = Homography.load(hp)
                cal.update(rmse_m=h.rmse_m, n_points=len(h.points or []),
                           frame=h.reference_frame, camera=getattr(h, "camera_params", None),
                           method=getattr(h, "method", "points"))
                auto = hp.with_name(hp.stem + "_auto.json")
                if auto.exists():
                    cal["auto_candidate"] = str(auto)
                    try:
                        ha = Homography.load(auto)
                        cal["auto_method"] = getattr(ha, "method", "")
                        info = getattr(ha, "method_info", None) or {}
                        cal["auto_scores"] = info.get("leave_one_out_median_m")
                    except Exception:
                        pass
                    ac = auto.with_name(auto.stem + "_check_video.jpg")
                    if ac.exists():
                        cal["auto_check"] = str(ac)
            except Exception as e:
                cal["error"] = str(e)
            checks = []
            for suffix in ("_check_video.jpg", "_check.png"):
                q = hp.with_name(hp.stem + suffix)
                if q.exists():
                    checks.append(str(q))
            cal["checks"] = checks
        d["calibration"] = cal
        d["runs"] = self.runs(cfg)
        return d

    def runs(self, cfg) -> list[dict]:
        rd = _run_root(cfg)
        if not rd.exists():
            return []
        out = []
        for p in sorted(rd.glob("*"), key=lambda q: q.stat().st_mtime, reverse=True):
            if not Run(p).is_run():
                continue
            run = open_run(p)
            r = {"name": p.name, "path": str(p), "processed": run.processed(),
                 "images": run.metrics_csv.exists(), "mtime": p.stat().st_mtime}
            sj = run.stats_json
            if sj.exists():
                try:
                    s = json.loads(sj.read_text(encoding="utf-8"))
                    r["n_tracks"] = s.get("n_tracks")
                    r["check"] = s.get("calibration_check")
                except ValueError:
                    pass
            out.append(r)
        return out

    def run_detail(self, rel: str, run: str) -> dict:
        cfg = _load(self.resolve_config(rel))
        r = open_run(self._run_dir(cfg, run))
        d = {"name": r.root.name, "path": str(r.root)}

        def files(folder: Path, exts=None) -> list[str]:
            return [str(q) for q in sorted(folder.glob("*")) if q.is_file()
                    and (exts is None or q.suffix.lower() in exts)] if folder.exists() else []

        plate = r.plate(cfg.site)
        d["plate"] = str(plate) if plate.exists() else None
        imgs = files(r.images, IMG_EXT)
        d["images"] = [q for q in imgs if not q.endswith("_layer.png")]
        d["layers"] = [q for q in imgs if q.endswith("_layer.png")]
        d["process"] = files(r.images / "process", IMG_EXT)
        d["labels"] = files(r.labels, IMG_EXT)
        d["videos"] = files(r.videos, VID_EXT)
        d["extras"] = files(r.extras, IMG_EXT)
        d["data"] = files(r.data) + files(r.labels, {".txt"})
        d["houdini"] = files(r.houdini)
        d["rhino"] = files(r.rhino)
        if r.metrics_csv.exists():
            import csv

            with open(r.metrics_csv, encoding="utf-8") as fh:
                d["metrics"] = list(csv.DictReader(fh))
        if r.stats_json.exists():
            s = json.loads(r.stats_json.read_text(encoding="utf-8"))
            d["check"] = s.get("calibration_check")
        log = r.log
        if log.exists():
            d["log_tail"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        return d

    def _run_dir(self, cfg, run: str) -> Path:
        if not NAME_RE.match(run or ""):
            raise ValueError("bad run name")
        p = (_run_root(cfg) / run).resolve()
        if not p.is_dir():
            raise ValueError(f"no run {run!r}")
        return p

    # ---- combined plans (groundtrack composite)
    def composites(self) -> list[dict]:
        root = self.root / "runs" / "composites"
        return [{"name": q.name, "images": len(list((q / "images").glob("*.png")))}
                for q in sorted(root.glob("*")) if (q / "images").is_dir()] \
            if root.exists() else []

    def composite_detail(self, name: str) -> dict:
        if not NAME_RE.match(name or ""):
            raise ValueError("bad name")
        q = (self.root / "runs" / "composites" / name).resolve()
        if not q.is_dir():
            raise ValueError(f"no combined plan {name!r}")

        def files(folder: Path, exts=None) -> list[str]:
            return [str(f) for f in sorted(folder.glob("*")) if f.is_file()
                    and (exts is None or f.suffix.lower() in exts)] if folder.exists() else []

        imgs = files(q / "images", IMG_EXT)
        title = (q / "labels" / "title.txt")
        return {"name": name, "path": str(q),
                "title": title.read_text(encoding="utf-8").splitlines() if title.exists() else [],
                "images": [f for f in imgs if not f.endswith("_layer.png")],
                "layers": [f for f in imgs if f.endswith("_layer.png")],
                "labels": files(q / "labels", IMG_EXT),
                "data": files(q / "data") + files(q / "labels", {".txt"}),
                "houdini": files(q / "houdini"), "rhino": files(q / "rhino")}

    # ---- actions
    def start_job(self, body: dict) -> dict:
        kind = body.get("kind")
        if kind == "composite":
            return self._start_composite(body)
        f = self.resolve_config(body.get("config", "")) if kind != "device" else None
        rel = self.rel(f) if f else None
        cfg = _load(f) if f else None
        run = body.get("run")
        opts = body.get("options") or {}
        if kind == "run":
            args = ["run", "-c", rel]
            if opts.get("run_name"):
                if not NAME_RE.match(opts["run_name"]):
                    raise ValueError("run name: letters, numbers, - _ . only")
                args += ["--run-name", opts["run_name"]]
            if opts.get("max_frames"):
                args += ["--max-frames", str(int(opts["max_frames"]))]
            if opts.get("debug_video"):
                args.append("--debug-video")
        elif kind in ("process", "package", "debug-video", "stabilize", "rhino",
                      "calibration-picture", "process-sequence"):
            self._run_dir(cfg, run)
            args = [kind, "-c", rel, "--run", run]
            if kind == "process-sequence":
                args += ["--at", str(float(opts.get("at", 2.0)))]
            if kind == "process" and opts.get("debug_video"):
                args.append("--debug-video")
        elif kind == "calibrate":
            args = ["calibrate", "-c", rel]
            if opts.get("time") not in (None, ""):
                args += ["--time", str(float(opts["time"]))]
        elif kind == "roi":
            args = ["roi", "-c", rel]
        elif kind == "autocalibrate":
            args = ["autocalibrate", "-c", rel] + (["--replace"] if opts.get("replace") else [])
        elif kind == "device":
            args = ["device"]
        else:
            raise ValueError(f"unknown job {kind!r}")
        heavy = {"run", "process", "debug-video", "autocalibrate", "stabilize"}
        if kind in heavy and any(j.kind in heavy and j.status == "running"
                                 for j in self.jobs.values()):
            raise ValueError("another run is still going: wait for it or stop it first")
        label = f"{kind} · {cfg.site if cfg else ''}" + (f" · {run}" if run else "")
        job = self.job_class(kind, args, self.root, label)
        self.jobs[job.id] = job
        return job.summary()

    def _start_composite(self, body: dict) -> dict:
        """Several clips of one place on one plan -> runs/composites/<name>."""
        opts = body.get("options") or {}
        name = str(opts.get("name") or "").strip()
        if not NAME_RE.match(name):
            raise ValueError("name: letters, numbers, - _ . only")
        rels = [self.rel(self.resolve_config(c)) for c in body.get("configs") or []]
        if not rels:
            raise ValueError("pick at least one clip")
        args = ["composite", *rels, "--out", f"runs/composites/{name}"]
        if opts.get("title"):
            args += ["--title", str(opts["title"])[:120]]
        if opts.get("only_ok"):
            args.append("--only-ok")
        job = self.job_class("composite", args, self.root, f"combine · {name} ({len(rels)} clips)")
        self.jobs[job.id] = job
        return job.summary()

    def adopt_auto(self, body: dict) -> dict:
        """Use <name>_auto.json as the calibration; the current one is kept as a backup."""
        import shutil

        cfg = _load(self.resolve_config(body.get("config", "")))
        hp = cfg.path("homography")
        auto = hp.with_name(hp.stem + "_auto.json")
        if not auto.exists():
            raise ValueError("no automatic calibration to use")
        if hp.exists():
            shutil.copy2(hp, hp.with_name(hp.stem + ".before_auto.json"))
        for suffix in ("_check_video.jpg",):
            old, new = hp.with_name(hp.stem + suffix), auto.with_name(auto.stem + suffix)
            if old.exists():
                shutil.copy2(old, old.with_name(hp.stem + ".before_auto" + suffix))
            if new.exists():
                shutil.move(str(new), str(old))
        shutil.move(str(auto), str(hp))
        return {"ok": True, "backup": str(hp.with_name(hp.stem + ".before_auto.json"))}

    def set_camera(self, body: dict) -> dict:
        from .locate import write_camera_position

        f = self.resolve_config(body.get("config", ""))
        cam = {}
        for k in ("E", "N", "height_m", "position_tol_m", "height_tol_m", "hfov_deg"):
            if body.get(k) not in (None, ""):
                cam[k] = round(float(body[k]), 2)
        for k in ("E", "N", "height_m"):
            if k not in cam:
                raise ValueError(f"{k} is required")
        old = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("camera_position") or {}
        if "hfov_deg" not in cam and old.get("hfov_deg"):
            cam["hfov_deg"] = old["hfov_deg"]                   # keep a known zoom
        cam.setdefault("position_tol_m", 15.0)
        cam.setdefault("height_tol_m", round(max(1.0, 0.06 * cam["height_m"]), 1))
        write_camera_position(f, cam, note=str(body.get("note", ""))[:70])
        return {"ok": True, "camera_position": cam}

    def set_paths(self, body: dict) -> dict:
        f = self.resolve_config(body.get("config", ""))
        text = f.read_text(encoding="utf-8")
        for key in ("video", "geotiff"):
            if body.get(key):
                val = str(body[key]).strip().strip('"')
                line = f"{key}: {json.dumps(val.replace(chr(92), '/'))}"
                if re.search(rf"(?m)^{key}:.*$", text):
                    text = re.sub(rf"(?m)^{key}:.*$", lambda _m, ln=line: ln, text, count=1)
                else:
                    text = text.rstrip() + "\n" + line + "\n"
        f.write_text(text, encoding="utf-8")
        return {"ok": True}

    def new_site(self, body: dict) -> dict:
        from .templates import TEMPLATES

        name = str(body.get("site", "")).strip()
        if not NAME_RE.match(name):
            raise ValueError("site name: letters, numbers, - _ . only")
        tpl = body.get("template", "plaza")
        if tpl not in TEMPLATES:
            raise ValueError("template must be plaza or motorway")
        out = self.root / "sites" / f"{name}.yaml"
        if out.exists():
            raise ValueError(f"{out.name} already exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(TEMPLATES[tpl].format(site=name, fname=f"sites/{name}.yaml"),
                       encoding="utf-8")
        for d in ("footage", "maps", "calibration", "runs"):
            (self.root / d).mkdir(exist_ok=True)
        self.set_paths({"config": f"sites/{name}.yaml", "video": body.get("video"),
                        "geotiff": body.get("geotiff")})
        return {"ok": True, "config": f"sites/{name}.yaml"}

    def open_path(self, body: dict) -> dict:
        p = self.check_file(body.get("path", ""), must_exist=True)
        target = p if p.is_dir() else p.parent
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # noqa: S606 (local desktop action)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return {"ok": True}

    def check_file(self, path: str, must_exist: bool = True) -> Path:
        p = Path(path).resolve()
        if not any(p == r or r in p.parents for r in self.allowed_roots()):
            raise PermissionError("outside the project")
        if must_exist and not p.exists():
            raise FileNotFoundError(str(p))
        return p

    def thumb(self, p: Path, w: int) -> bytes:
        """A small preview, kept in memory and on disk (next to nothing of the project:
        <temp>/groundtrack_thumbs), so a big plan decodes once, not on every visit."""
        import hashlib
        import tempfile

        from PIL import Image

        st = p.stat()
        key = (str(p), st.st_mtime, w)
        if key not in self._thumbs:
            h = hashlib.sha1(f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}|{w}".encode())
            disk = Path(tempfile.gettempdir()) / "groundtrack_thumbs" / f"{h.hexdigest()}.bin"
            if disk.exists():
                data = disk.read_bytes()
            else:
                im = Image.open(p)
                im.draft("RGB", (w, w * 4))          # JPEG: decode at a reduced size
                im.thumbnail((w, w * 4), Image.Resampling.BILINEAR, reducing_gap=2.0)
                buf = io.BytesIO()
                if im.mode in ("RGBA", "LA", "P"):
                    im.save(buf, "PNG")
                else:
                    im.convert("RGB").save(buf, "JPEG", quality=85)
                data = buf.getvalue()
                try:
                    disk.parent.mkdir(parents=True, exist_ok=True)
                    disk.write_bytes(data)
                except OSError:
                    pass
            if len(self._thumbs) > 400:
                self._thumbs.clear()
            self._thumbs[key] = data
        return self._thumbs[key]


def _finite(obj):
    """obj with every Infinity / NaN as None (e.g. the RMSE of an exact 5-pair homography)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _load(path: Path):
    from .config import load_config

    return load_config(path)


def _run_root(cfg) -> Path:
    return (cfg.path("output_dir") or cfg.base_dir / "runs") / cfg.site


# --------------------------------------------------------------------------- http
def make_handler(app: App, port: int):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "groundtrack-ui"

        def log_message(self, *a):  # quiet
            pass

        # ---- helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            try:
                body = json.dumps(obj, default=str, allow_nan=False)
            except ValueError:          # Infinity / NaN are not JSON for the browser: null
                body = json.dumps(_finite(obj), default=str)
            self._send(code, body.encode("utf-8"), "application/json; charset=utf-8")

        def _guard(self) -> bool:
            # only this machine, only this page: stops other websites from driving the UI
            if self.headers.get("Host") not in hosts:
                self._send(403, b"forbidden", "text/plain")
                return False
            return True

        # ---- routes
        def do_GET(self):  # noqa: N802
            if not self._guard():
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/sites":
                    self._json({"root": str(app.root), "sites": app.sites(),
                                "jobs": [j.summary() for j in app.jobs.values()]})
                elif u.path == "/api/site":
                    self._json(app.site(q["config"]))
                elif u.path == "/api/run":
                    self._json(app.run_detail(q["config"], q["run"]))
                elif u.path == "/api/composites":
                    self._json({"composites": app.composites()})
                elif u.path == "/api/composite":
                    self._json(app.composite_detail(q["name"]))
                elif u.path == "/api/job":
                    j = app.jobs[q["id"]]
                    since = int(q.get("since", 0))
                    self._json({**j.summary(), "lines": j.lines[since:], "n": len(j.lines)})
                elif u.path == "/file":
                    self._file(app.check_file(q["path"]), int(q.get("w", 0) or 0),
                               q.get("download") == "1")
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except FileNotFoundError as e:
                self._json({"error": f"not found: {e}"}, 404)
            except (ConnectionError, BrokenPipeError):
                pass

        do_HEAD = do_GET  # noqa: N815

        def do_POST(self):  # noqa: N802
            if not self._guard():
                return
            if self.headers.get("X-Groundtrack") != "1":   # custom header: no cross-site forms
                self._send(403, b"forbidden", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                route = urlparse(self.path).path
                if route == "/api/job":
                    self._json(app.start_job(body))
                elif route == "/api/job/stop":
                    app.jobs[body["id"]].stop()
                    self._json({"ok": True})
                elif route == "/api/locate":
                    from .locate import search

                    self._json({"hits": search(str(body.get("query", ""))[:200])})
                elif route == "/api/adopt-auto":
                    self._json(app.adopt_auto(body))
                elif route == "/api/camera":
                    self._json(app.set_camera(body))
                elif route == "/api/paths":
                    self._json(app.set_paths(body))
                elif route == "/api/new-site":
                    self._json(app.new_site(body))
                elif route == "/api/open":
                    self._json(app.open_path(body))
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError, OSError) as e:
                self._json({"error": str(e)}, 400)

        def _file(self, p: Path, w: int, download: bool):
            if p.is_dir():
                raise FileNotFoundError(str(p))
            if w and p.suffix.lower() in IMG_EXT:
                data = app.thumb(p, w)
                self._send(200, data, "image/png" if data[:4] == b"\x89PNG" else "image/jpeg")
                return
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix.lower() in {".csv", ".txt", ".py", ".json", ".geojson"}:
                ctype = "text/plain; charset=utf-8"
            size = p.stat().st_size
            extra = {"Accept-Ranges": "bytes"}
            if download:
                extra["Content-Disposition"] = f'attachment; filename="{p.name}"'
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                code = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            else:
                code = 200
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            for k, v in extra.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(p, "rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


def serve(root: Path, port: int = 8765, open_browser: bool = True) -> None:
    app = App(root)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"motus UI on {url}  (project {app.root}; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in app.jobs.values():
            j.stop()
        srv.server_close()
