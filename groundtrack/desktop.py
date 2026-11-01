"""Desktop app: `groundtrack app`.

The whole workflow in one window: sites, footage and aerial, where the camera stood,
calibration and the area to track (both drawn in the app), runs with live logs, results,
combined plans, and the hand-over to Houdini (POP kits: build, open, add to the open scene)
and Rhino / Grasshopper (models, definitions, the groundtrack-grasshopper plugin).

It is the `groundtrack ui` server (same routes, same checks: this machine only, files inside
the project or a library folder you added) with more routes and a new page, shown in a native
window (pywebview: Edge WebView2 on Windows). Without pywebview it opens in the browser.
Every action still runs the normal `groundtrack` command, so the command line and the app
always do the same thing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import desktop_cal as cal
from . import desktop_links as links
from .layout import Run, open_run
from .ui import NAME_RE, App, Job, _load, _run_root, make_handler

WEB = Path(__file__).with_name("desktop_web")
CTYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".js": "application/javascript; charset=utf-8", ".woff2": "font/woff2",
          ".svg": "image/svg+xml", ".png": "image/png", ".txt": "text/plain; charset=utf-8"}
TEXT_EXT = {".py", ".txt", ".json", ".md", ".yaml", ".csv"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class NeedsConfirm(ValueError):
    """The action is possible but risky (e.g. writing a file Rhino may have open): ask first."""


def console_python() -> str:
    """python.exe next to pythonw.exe: jobs get a (hidden) console their own tools inherit,
    instead of each ffmpeg / hython popping up a window."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


class AppJob(Job):
    """A Job run without a console window."""

    def __init__(self, kind: str, args: list[str], cwd: Path, label: str):  # noqa: D107
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.args, self.label = kind, args, label
        self.lines: list[str] = []
        self.progress = ""
        self.started, self.ended = time.time(), None
        self.returncode: int | None = None
        self.cancelled = False
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([console_python(), "-m", "groundtrack", *args],
                                     cwd=str(cwd), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                     creationflags=NO_WINDOW)
        threading.Thread(target=self._read, daemon=True).start()


# --------------------------------------------------------------------------- state
class DesktopApp(App):
    job_class = AppJob

    def __init__(self, root: Path, settings: links.Settings | None = None):
        super().__init__(root)
        self.settings = settings or links.Settings()
        self.sites_cache = cal.Sites()
        self.device: dict = {"state": "checking"}
        self._roots: tuple[float, list[Path]] = (0.0, [])
        self._roots_lock = threading.Lock()
        self._scan: tuple[float, dict | None] = (0.0, None)

    # ---- folders
    def libraries(self) -> list[Path]:
        """Where kits, models and definitions are looked for: the project's runs, the
        Grasshopper plugin's default kit folder, and the folders added in the app."""
        out = [self.root / "runs"]
        gh_default = Path.home() / "Documents" / "groundtrack houdini"
        if gh_default.is_dir():
            out.append(gh_default)
        out += [Path(p) for p in self.settings.libraries]
        seen, uniq = set(), []
        for p in out:
            if str(p) not in seen:
                seen.add(str(p))
                uniq.append(p)
        return uniq

    def allowed_roots(self) -> list[Path]:
        t, roots = self._roots
        if time.time() - t <= 5:
            return roots
        # one thread recomputes; others use the previous result (t > 0), or wait when the
        # cache was just invalidated (t == 0: the old roots must not be used)
        if not self._roots_lock.acquire(blocking=(t == 0.0)):
            return roots
        try:
            t, roots = self._roots
            if time.time() - t > 5:
                roots = super().allowed_roots() + [p.resolve() for p in self.libraries()]
                self._roots = (time.time(), roots)
            return roots
        finally:
            self._roots_lock.release()

    def scan(self, refresh: bool = False) -> dict:
        t, data = self._scan
        if refresh or data is None or time.time() - t > 30:
            data = links.scan(self.libraries())
            self._scan = (time.time(), data)
        return data

    # ---- overview
    def overview(self) -> dict:
        return {"root": str(self.root), "sites": self.sites(), "device": self.device,
                "jobs": [j.summary() for j in self.jobs.values()],
                "running": links.running_apps()}

    def check_device(self) -> None:
        """`groundtrack device` once in the background (importing torch takes seconds)."""
        def work():
            try:
                r = subprocess.run([console_python(), "-m", "groundtrack", "device"],
                                   capture_output=True, text=True, timeout=180, cwd=str(self.root),
                                   creationflags=NO_WINDOW)
                d = {"state": "ok", "text": r.stdout.strip()}
                for ln in r.stdout.splitlines():
                    if ln.startswith("selected device:"):
                        d["device"] = ln.split(":", 1)[1].strip()
                    elif ln.startswith("GPU:"):
                        d["gpu"] = ln[4:].split(",")[0].strip()
                if "device" not in d:
                    d = {"state": "error", "text": (r.stdout + r.stderr).strip()[-400:]}
                self.device = d
            except (OSError, subprocess.SubprocessError) as e:
                self.device = {"state": "error", "text": str(e)}

        threading.Thread(target=work, daemon=True).start()

    def links(self, refresh: bool = False) -> dict:
        return {"houdini": links.houdini_install(), "rhino": links.rhino_install(),
                "grasshopper": links.grasshopper_plugin(), "running": links.running_apps(),
                "libraries": [{"path": str(p), "added": str(p) in self.settings.libraries}
                              for p in self.libraries()],
                **self.scan(refresh)}

    # ---- calibration / ROI
    def cal_site(self, rel: str) -> cal.Site:
        return self.sites_cache.get(self.resolve_config(rel))

    def cal_save(self, body: dict) -> dict:
        """Save the clicked pairs and fit the calibration with `groundtrack calibrate`."""
        rel = self.rel(self.resolve_config(body.get("config", "")))
        site = self.cal_site(rel)
        frame = int(body.get("frame", 0))
        csv = cal.save_pairs(site, body.get("pairs") or [])
        args = ["calibrate", "-c", rel, "--points-csv", str(csv), "--no-gui",
                "--frame", str(frame)]
        job = self.job_class("calibrate", args, self.root, f"calibrate · {site.cfg.site}")
        self.jobs[job.id] = job
        return job.summary()

    # ---- jobs
    def start_job(self, body: dict) -> dict:
        kind = body.get("kind")
        opts = body.get("options") or {}
        if kind == "rhino" and links.running_apps()["rhino"] and not opts.get("force"):
            raise NeedsConfirm("Rhino is open. If this run's model is open in it, close it "
                               "first: the model is rewritten.")
        return super().start_job(body)

    # ---- Houdini
    def clips(self) -> list[dict]:
        """Runs with a vector field (houdini/), newest first per site: kit modules."""
        out = []
        for f in self.config_files():
            try:
                cfg = _load(f)
            except Exception:  # noqa: BLE001 (a broken config is listed as such elsewhere)
                continue
            rd = _run_root(cfg)
            if not rd.exists():
                continue
            runs = sorted((p for p in rd.glob("*") if Run(p).is_run()),
                          key=lambda q: q.stat().st_mtime, reverse=True)
            for p in runs:
                r = open_run(p)
                if (r.houdini / "houdini_field.py").exists():
                    out.append({"config": self.rel(f), "site": cfg.site, "run": p.name,
                                "path": str(r.root)})
                    break
        return out

    def composites_all(self) -> list[dict]:
        comps = [{"name": c["name"], "path": str(self.root / "runs" / "composites" / c["name"])}
                 for c in self.composites()]
        have = {c["path"] for c in comps}
        for c in self.scan()["composites"]:
            if c["path"] not in have:
                p = Path(c["path"])
                comps.append({"name": f"{p.parent.name} / {p.name}", "path": c["path"]})
        return [c for c in comps if links.is_composite(Path(c["path"]))]

    def _kit_out(self, out: str) -> Path:
        p = Path(out).resolve()
        if not any(p == r or r in p.parents for r in self.allowed_roots()):
            raise PermissionError("the kit must go inside the project or a library folder")
        return p

    def _refuse_open_scene(self, kit: Path) -> None:
        has_scene = any(q.suffix.lower() in links.HIP_EXT for q in kit.glob("*")) \
            if kit.exists() else False
        if has_scene and links.running_apps()["houdini"]:
            raise ValueError("Houdini is open and this kit's scene may be open in it: close "
                             "Houdini first, or use 'Add to open scene' instead")

    def houdini_build(self, body: dict) -> dict:
        """A site's POP kit from its combined plan and clips (`groundtrack houdini-pop`)."""
        site = str(body.get("site", "")).strip()
        if not NAME_RE.match(site):
            raise ValueError("site: letters, numbers, - _ . only")
        comps = {c["path"] for c in self.composites_all()}
        comp = str(Path(body.get("composite", "")).resolve()) if body.get("composite") else ""
        if comp not in comps:
            raise ValueError("pick a combined plan")
        known = {c["path"]: c for c in self.clips()}
        clips = [str(Path(c).resolve()) for c in body.get("clips") or []]
        bad = [c for c in clips if c not in {str(Path(k).resolve()) for k in known}]
        if bad:
            raise ValueError(f"unknown clip {bad[0]}")
        out = self._kit_out(body.get("out") or str(self.root / "runs" / "houdini" / site))
        self._refuse_open_scene(out)
        args = ["houdini-pop", "--site", site, "--composite", comp, "--out", str(out)]
        if clips:
            args += ["--clips", *clips]
        if body.get("no_build"):
            args.append("--no-build")
        job = self.job_class("houdini-pop", args, self.root, f"houdini kit · {site}")
        self.jobs[job.id] = job
        return job.summary()

    def houdini_rebuild(self, body: dict) -> dict:
        """Build the kit's scene again in Houdini (hython), e.g. a kit from Grasshopper."""
        kit = self.check_file(body.get("kit", ""))
        if not links.is_kit(kit):
            raise ValueError("not a Houdini kit (modules.json + hou_pop_build.py)")
        self._refuse_open_scene(kit)
        job = self.job_class("houdini-build", ["houdini-build", str(kit)], self.root,
                             f"houdini scene · {kit.name}")
        self.jobs[job.id] = job
        return job.summary()

    # ---- files
    def open_file(self, body: dict) -> dict:
        links.open_file(self.check_file(body.get("path", "")))
        return {"ok": True}

    def text(self, path: str) -> dict:
        p = self.check_file(path)
        if p.suffix.lower() not in TEXT_EXT or p.stat().st_size > 2_000_000:
            raise ValueError("not a text file this app shows")
        return {"path": str(p), "text": p.read_text(encoding="utf-8", errors="replace")}

    def library(self, body: dict) -> dict:
        if body.get("action") == "remove":
            self.settings.remove_library(str(body.get("path", "")))
        else:
            self.settings.add_library(str(body.get("path", "")))
        self._roots = (0.0, [])
        return self.links(refresh=True)

    # ---- Live view (plans/live_controller/03): a run's scene package, the phone hub
    _hub = None                              # one LiveHub per app, made when Live mode opens
    _hub_lock = threading.Lock()
    hub_options = {"port": 8766, "view_port": 8767, "host": "0.0.0.0"}
    live_files = {"scene.json", "terrain.f32", "aerial.jpg", "tracks.bin", "tracks_index.json",
                  "video.mp4", "cloud.json", "cloud.bin"}   # what the view may fetch (no check.png, no log)

    def live_dir(self, run: str) -> Path:
        """The run's scene package (<run>/live/), through the usual checks (inside the
        project or a library folder)."""
        live = self.check_file(str(Path(run) / "live"))
        if not (live / "scene.json").is_file():
            raise FileNotFoundError(f"no live package in {live.parent}")
        return live

    def live_file(self, run: str, name: str) -> Path:
        """One file of the package, by name: only what the view needs."""
        if name not in self.live_files:
            raise FileNotFoundError(name)
        return self.check_file(str(self.live_dir(run) / name))

    def live_info(self, run: str) -> dict:
        try:
            live = self.live_dir(run)
        except (FileNotFoundError, PermissionError):
            return {"live": False}
        m = json.loads((live / "scene.json").read_text(encoding="utf-8"))
        return {"live": True, "run": str(live.parent), "clip": m.get("clip"), "name": m.get("run"),
                "frames": m.get("frames"), "fps": m.get("fps"), "duration_s": m.get("duration_s"),
                "n_tracks": m.get("tracks", {}).get("n_tracks"),
                "blurred": bool(m.get("video", {}).get("blurred"))}

    def live_rules(self, run: str) -> dict:
        """The machine's rules for Live (plans/live_controller/08 c): the run's own config
        (<run>/_working/config_used.yaml) as live_rules.rules_config gives them to rules.js; a run
        without one: groundtrack's defaults (config.default_config), "source": "defaults" (the
        view says so in its HUD)."""
        from .config import default_config
        from .live_rules import rules_config

        live = self.live_dir(run)
        cfg = self.check_file(str(live.parent / "_working" / "config_used.yaml"), must_exist=False)
        if cfg.is_file():
            return {**rules_config(cfg), "source": "run"}
        return {**rules_config(default_config()), "source": "defaults"}

    def live_clips(self) -> list[dict]:
        """Every run of the project's sites with a scene package (the clip picker)."""
        out = []
        for f in self.config_files():
            try:
                rd = _run_root(_load(f))
            except Exception:  # noqa: BLE001 (a broken config is listed as such elsewhere)
                continue
            for p in sorted(rd.glob("*/live/scene.json")) if rd.is_dir() else []:
                out.append({"name": f"{rd.name}/{p.parent.parent.name}", "run": str(p.parent.parent)})
        return out

    def live_site(self, run: str) -> dict:
        """The cameras a Live view on this run can show: its site's scene packages, with names,
        places and areas (live_site.site_cameras; plans/live_controller/08 b)."""
        from .live_site import site_cameras

        live = self.live_dir(run)
        return site_cameras(live.parent, self.config_files())

    def live_sites(self) -> dict:
        """The sites with Live packages and each one's default start camera (the site picker)."""
        from .live_site import site_list

        return {"sites": site_list(self.config_files())}

    # passes (plans/live_controller/08 b, "Passes"): saved under <run>/live/passes/ only
    def live_passes(self, run: str) -> dict:
        """The passes saved for the run's site: under every camera's run, newest first."""
        from .live_passes import list_passes
        from .live_site import site_cameras

        live = self.live_dir(run)
        runs = [live.parent] + [Path(c["run"]) for c in site_cameras(live.parent, self.config_files())["cameras"]]
        return {"passes": list_passes(runs)}

    def live_pass(self, run: str, pass_id: str) -> dict:
        from .live_passes import read_pass

        m = read_pass(self.live_dir(run).parent, pass_id)
        for c in m.get("cameras") or []:                  # each camera's run, absolute (for the view)
            if c.get("run"):
                c["path"] = str((self.root / c["run"]).resolve())
        return m

    def live_pass_file(self, run: str, pass_id: str, name: str) -> Path:
        from .live_passes import pass_file

        return self.check_file(str(pass_file(self.live_dir(run).parent, pass_id, name)))

    def live_pass_post(self, body: dict) -> dict:
        """save {run, id (none: a new pass), manifest, data: {file: base64}} / delete {run, id}.
        Only inside the project's runs/ folder."""
        from .live_passes import delete_pass, save_pass

        run = self._pass_run(body.get("run"))
        action = body.get("action")
        if action == "save":
            return save_pass(run, body.get("manifest") or {}, body.get("data") or {}, body.get("id") or None,
                             root=self.root)
        if action == "delete":
            delete_pass(run, str(body.get("id", "")))
            return {"ok": True}
        raise ValueError("action: save or delete")

    def _pass_run(self, run) -> Path:
        """The run a pass is saved under: a run with a live package, inside the project's runs/."""
        run = self.live_dir(str(run or "")).parent.resolve()
        if (self.root / "runs").resolve() not in run.parents:
            raise PermissionError("passes are saved inside the project's runs/ only")
        return run

    # recordings, renders, exports (plans/live_controller/05)
    def live_pass_save(self, body: bytes) -> dict:
        """A save sent as [u32 header length][JSON header][files] (live_passes.parse_files)."""
        from .live_passes import parse_files, save_pass

        head, files = parse_files(body)
        if head.get("action") != "save":
            raise ValueError("action: save")
        return save_pass(self._pass_run(head.get("run")), head.get("manifest") or {}, files,
                         head.get("id") or None, root=self.root)

    def renders(self):
        if self.__dict__.get("_renders") is None:
            from .live_render import Renders

            self._renders = Renders()
        return self._renders

    def live_render(self, body: dict) -> dict:
        """begin {run, id, fps, width, height, frames, view} / finish {token, keep} / cancel {token}."""
        action = body.get("action")
        if action == "begin":
            return self.renders().begin(self._pass_run(body.get("run")), str(body.get("id", "")),
                                        body.get("fps", 30), body.get("width", 1920),
                                        body.get("height", 1080), body.get("frames", 0),
                                        body.get("view") or "orbit")
        if action == "finish":
            return self.renders().finish(str(body.get("token", "")), bool(body.get("keep")))
        if action == "cancel":
            return self.renders().cancel(str(body.get("token", "")))
        raise ValueError("action: begin, finish or cancel")

    def live_export(self, body: dict) -> dict:
        """`groundtrack live-pass-export` as a job: {run, id, rhino, houdini, no_build, force, cloud (the
        cameras' point clouds: Live sends it while its Point cloud is on)}."""
        from .live_passes import read_pass

        run, pid = self._pass_run(body.get("run")), str(body.get("id", ""))
        read_pass(run, pid)
        running = links.running_apps()
        if not body.get("force"):
            if body.get("rhino") and running["rhino"]:
                raise NeedsConfirm("Rhino is open. If this pass's model is open in it, close it "
                                   "first: the model is rewritten.")
            if body.get("houdini") and not body.get("no_build") and running["houdini"]:
                raise NeedsConfirm("Houdini is open. If this pass's kit scene is open in it, close "
                                   "it first: the scene is rebuilt.")
        args = ["live-pass-export", str(run), pid]
        args += ["--rhino"] if body.get("rhino") else []
        args += ["--houdini"] if body.get("houdini") else []
        args += ["--no-build"] if body.get("no_build") else []
        args += ["--cloud"] if body.get("cloud") is True else []
        what = "+".join(k for k in ("rhino", "houdini") if body.get(k)) or "files"
        job = self.job_class("live-pass-export", args, self.root, f"pass export · {pid} · {what}")
        self.jobs[job.id] = job
        from .live_pass_export import export_dir

        return {**job.summary(), "dir": str(export_dir(run, pid))}

    def live_cloud(self, body: dict) -> dict:
        """A site camera's point cloud (plans/live_controller/11; Live asks only with its cloud on):
        `groundtrack live-cloud <run>` as a job, {run, remove}. A Realtime recording's is made while it
        records, not here."""
        live = self.live_dir(str(body.get("run", "")))
        meta = json.loads((live / "scene.json").read_text(encoding="utf-8"))
        if meta.get("realtime"):
            raise ValueError("a Realtime recording: its cloud is made while it records")
        args = ["live-cloud", str(live.parent)] + (["--remove"] if body.get("remove") else [])
        what = "remove" if body.get("remove") else "make"
        job = self.job_class("live-cloud", args, self.root, f"cloud · {meta.get('clip')} · {what}")
        self.jobs[job.id] = job
        return job.summary()

    def live_layers_export(self, body: bytes) -> dict:
        """The collage (or one camera's layer) as a run: GeoJSON and the files always, Rhino and Houdini
        unless the header says false (Live's Output: All, or one of them): [u32 header
        length][JSON header][files] (live_passes.parse_files). Header {camera (one camera's layer; none:
        the collage), site, layers: [{camera, camera_id, package, run, id, key, kind, stacked, up_to,
        file}], rhino, houdini, force, cloud}; one drawing per layer. Prepared in <run of the first
        layer>/live/exports/, then `groundtrack live-layers-export` as a job."""
        from .live_pass_export import MAX_LAYERS, prepare_layers
        from .live_passes import parse_files

        head, files = parse_files(body)
        raw = head.get("layers")
        if not isinstance(raw, list) or not raw or len(raw) > MAX_LAYERS:
            raise ValueError("layers: a list")
        layers = []
        for ly in raw:
            ok = isinstance(ly, dict) and isinstance(ly.get("camera"), str) and isinstance(ly.get("file"), str)
            if not ok:
                raise ValueError("a layer: {camera, package, run, id, file}")
            pkg = self.live_dir(str(ly.get("package") or "")).parent.resolve()
            run = self._pass_run(ly.get("run") or pkg)
            pid = ly.get("id")
            layers.append({**{k: ly.get(k) for k in ("camera", "camera_id", "key", "kind", "stacked", "up_to", "file")},
                           "package": pkg, "run": run, "id": str(pid) if pid else None})
        cam = head.get("camera") if isinstance(head.get("camera"), str) and head.get("camera") else None
        rhino, houdini = head.get("rhino", True) is not False, head.get("houdini", True) is not False
        running = links.running_apps()
        if not head.get("force"):
            if rhino and running["rhino"]:
                raise NeedsConfirm("Rhino is open. The export writes a new model in live/exports/ (nothing "
                                   "open is rewritten). Export?")
            if houdini and running["houdini"]:
                raise NeedsConfirm("Houdini is open. The export builds a new kit scene in live/exports/ with "
                                   "hython (nothing open is rebuilt). Export?")
        out = prepare_layers(layers, files, camera=cam, site=str(head.get("site") or ""), root=self.root)
        args = ["live-layers-export", str(out)] + (["--rhino"] if rhino else []) + (["--houdini"] if houdini else [])
        args += ["--cloud"] if head.get("cloud") is True else []
        what = "+".join(k for k, on in (("rhino", rhino), ("houdini", houdini)) if on) or "files"
        job = self.job_class("live-layers-export", args, self.root,
                             f"export · {cam or 'collage'} · {len(layers)} layers · {what}")
        self.jobs[job.id] = job
        return {**job.summary(), "dir": str(out)}

    def live_hub(self, body: dict) -> dict:
        """The phone hub: start (Live mode opens), info, new-code (a new pairing code), stop."""
        action = body.get("action", "info")
        with self._hub_lock:
            if action == "start":
                if self._hub is None:
                    from .live_hub import LiveHub

                    self._hub = LiveHub(**self.hub_options)
                return self._hub.start()
            if self._hub is None:
                return {"running": False}
            if action == "new-code":
                return self._hub.rotate_token()
            if action == "stop":
                self._hub.stop()
            return self._hub.info()

    # realtime (plans/live_controller/10): one phone films, the laptop tracks
    _realtime = None                         # live_track.LiveTracker while the mode is on

    def live_realtime(self, body: dict) -> dict:
        """The realtime mode: start (the hub too; the model loads in the background), stop (the
        recording saved, the model released: picking another site), end (the recording only),
        size (long_side 1920 or 1280: the next recording's frames; start takes it too), cloud (on:
        the point cloud while recording, plans/live_controller/11; off by default; start takes it
        too), status.
        Always with the recordings (runs/live/, newest first), unless "light" (the Live view's
        poll: the status only)."""
        from .live_track import LiveTracker, list_recordings

        action = body.get("action", "status")
        with self._hub_lock:
            if action == "start":
                if self._hub is None:
                    from .live_hub import LiveHub

                    self._hub = LiveHub(**self.hub_options)
                self._hub.start()
                if self._realtime is None:
                    self._realtime = LiveTracker(self.root)
                if body.get("long_side") is not None:
                    self._realtime.set_long_side(body["long_side"])
                if body.get("cloud") is not None:
                    self._realtime.set_cloud(body["cloud"] is True)
                self._realtime.load()
                self._hub.set_recorder(self._realtime)
            elif action == "stop" and self._realtime is not None:
                rt, self._realtime = self._realtime, None
                if self._hub is not None:
                    self._hub.set_recorder(None)
                rt.end()
                threading.Thread(target=rt.close, name="rt-close", daemon=True).start()
            elif action == "end" and self._realtime is not None:
                self._realtime.end()
            elif action == "size" and self._realtime is not None:
                self._realtime.set_long_side(body.get("long_side"))
            elif action == "cloud" and self._realtime is not None:
                self._realtime.set_cloud(body.get("on") is True)
            elif action not in ("start", "stop", "end", "size", "cloud", "status"):
                raise ValueError("action: start, stop, end, size, cloud or status")
            st = self._realtime.status() if self._realtime is not None else {"mode": "off"}
        if body.get("light"):
            return st
        return {**st, "recordings": list_recordings(self.root)}

    def live_realtime_tracks(self, rid: str, n0: int) -> bytes:
        """A recording's points from n0 on for the Live view (step 3): [u32 header length][the
        header JSON (live_track.LiveTracker.tracks_since)][TRACK_DTYPE records]. A finished
        recording's come from its package, the realtime mode on or not."""
        import struct

        from .live_track import package_tracks

        rt = self._realtime
        head, data = rt.tracks_since(rid, n0) if rt is not None else package_tracks(self.root, rid, n0)
        h = json.dumps(head).encode()
        return struct.pack("<I", len(h)) + h + data

    def live_realtime_cloud(self, rid: str, v0: int) -> bytes:
        """A recording's rough cloud so far while it records with the cloud on (11, step 2): [u32 header
        length][the header JSON (live_cloud.CloudBuilder.snapshot)][CLOUD_DTYPE records]."""
        import struct

        rt = self._realtime
        if rt is None:
            raise FileNotFoundError("realtime is off")
        head, data = rt.cloud_since(rid, v0)
        h = json.dumps(head).encode()
        return struct.pack("<I", len(h)) + h + data

    def live_realtime_preview(self, rid: str) -> bytes | None:
        """The recording's newest blurred picture (JPEG), while it records."""
        rt = self._realtime
        return rt.preview(rid) if rt is not None else None

    def stop_hub(self) -> None:
        if self._realtime is not None:
            rt, self._realtime = self._realtime, None
            rt.close()
        if self._hub is not None:
            self._hub.stop()


# --------------------------------------------------------------------------- http
def make_desktop_handler(app: DesktopApp, port: int):
    Base = make_handler(app, port)

    post_routes = {
        "/api/job": app.start_job,
        "/api/cal/fit": None, "/api/cal/save": app.cal_save,
        "/api/roi/save": None, "/api/roi/clear": None,
        "/api/app/library": app.library, "/api/app/open-file": app.open_file,
        "/api/houdini/build": app.houdini_build, "/api/houdini/rebuild": app.houdini_rebuild,
        "/api/live/hub": app.live_hub, "/api/live/pass": app.live_pass_post,
        "/api/live/realtime": app.live_realtime,
        "/api/live/render": app.live_render, "/api/live/export": app.live_export,
        "/api/live/cloud": app.live_cloud,
        "/api/live/pass/save": None, "/api/live/render/frame": None,       # binary bodies
        "/api/live/sheet": None,          # the performance sheet (09): the view's PNG in, the sheet out
        "/api/live/export/layers": None,  # the collage / a camera's layer: the layers' drawings in, a job
    }

    class Handler(Base):
        server_version = "groundtrack-app"

        def _static(self, name: str):
            p = (WEB / name).resolve()
            if WEB.resolve() not in p.parents or not p.is_file():
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, p.read_bytes(), CTYPES.get(p.suffix.lower(),
                                                        "application/octet-stream"))

        def do_GET(self):  # noqa: N802
            if not self._guard():
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self._static("index.html")
                elif u.path.startswith("/static/"):
                    self._static(u.path[len("/static/"):])
                elif u.path == "/api/app/overview":
                    self._json(app.overview())
                elif u.path == "/api/app/links":
                    self._json(app.links(q.get("refresh") == "1"))
                elif u.path == "/api/app/kit-sources":
                    self._json({"composites": app.composites_all(), "clips": app.clips()})
                elif u.path == "/api/app/text":
                    self._json(app.text(q["path"]))
                elif u.path == "/api/cal/state":
                    self._json(cal.state(app.cal_site(q["config"])))
                elif u.path == "/api/cal/frame":
                    data = cal.frame_jpeg(app.cal_site(q["config"]), int(q.get("frame", 0)),
                                          int(q.get("w", 0) or 0))
                    self._send(200, data, "image/jpeg")
                elif u.path == "/api/cal/map":
                    data = cal.map_jpeg(app.cal_site(q["config"]), int(q.get("w", 0) or 0))
                    self._send(200, data, "image/jpeg")
                # Live view: the page, a run's package files (video with ranges), the hub
                elif u.path == "/live":
                    self._static("live/live.html")
                elif u.path == "/live/realtime":
                    self._static("live/realtime.html")
                elif u.path == "/api/live/realtime":
                    self._json(app.live_realtime({"action": "status", "light": q.get("light") == "1"}))
                elif u.path == "/api/live/realtime/tracks":
                    self._send(200, app.live_realtime_tracks(q["id"], int(q.get("from", 0) or 0)),
                               "application/octet-stream")
                elif u.path == "/api/live/realtime/cloud":
                    self._send(200, app.live_realtime_cloud(q["id"], int(q.get("v") or -1)),
                               "application/octet-stream")
                elif u.path == "/api/live/realtime/preview":
                    data = app.live_realtime_preview(q["id"])
                    if data is None:
                        self._send(204, b"", "image/jpeg")
                    else:
                        self._send(200, data, "image/jpeg")
                elif u.path.startswith("/live/pkg/"):
                    self._file(app.live_file(q["run"], u.path[len("/live/pkg/"):]), 0, False)
                elif u.path == "/api/live/info":
                    self._json(app.live_info(q["run"]))
                elif u.path == "/api/live/clips":
                    self._json({"clips": app.live_clips()})
                elif u.path == "/api/live/site":
                    self._json(app.live_site(q["run"]))
                elif u.path == "/api/live/sites":
                    self._json(app.live_sites())
                elif u.path == "/api/live/passes":
                    self._json(app.live_passes(q["run"]))
                elif u.path == "/api/live/pass":
                    self._json(app.live_pass(q["run"], q["id"]))
                elif u.path == "/live/passdata":
                    self._file(app.live_pass_file(q["run"], q["id"], q["file"]), 0, False)
                elif u.path == "/api/live/rules":
                    self._json(app.live_rules(q["run"]))
                elif u.path == "/api/live/hub":
                    self._json(app.live_hub({"action": "info"}))
                else:
                    super().do_GET()
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except FileNotFoundError as e:
                self._json({"error": str(e)}, 404)
            except (ConnectionError, BrokenPipeError):
                pass
            except Exception as e:  # noqa: BLE001 (shown in the app, not a dropped connection)
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):  # noqa: N802
            route = urlparse(self.path).path
            if route not in post_routes:
                super().do_POST()
                return
            if not self._guard():
                return
            if self.headers.get("X-Groundtrack") != "1":   # custom header: no cross-site forms
                self._send(403, b"forbidden", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                if route == "/api/live/pass/save":
                    self._json(app.live_pass_save(self.rfile.read(n)))
                    return
                if route == "/api/live/sheet":
                    from .live_sheet import request as sheet_request

                    self._json(sheet_request(app._pass_run, self.rfile.read(n)))
                    return
                if route == "/api/live/export/layers":
                    self._json(app.live_layers_export(self.rfile.read(n)))
                    return
                if route == "/api/live/render/frame":
                    q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                    self._json(app.renders().frame(q["token"], int(q["frame"]), self.rfile.read(n)))
                    return
                body = json.loads(self.rfile.read(n) or b"{}")
                if route == "/api/cal/fit":
                    site = app.cal_site(body.get("config", ""))
                    self._json(cal.fit(site, body.get("pairs") or [], body.get("query"),
                                       body.get("camera")))
                elif route == "/api/roi/save":
                    p = cal.save_roi(app.cal_site(body.get("config", "")), body.get("points"))
                    self._json({"ok": True, "path": str(p)})
                elif route == "/api/roi/clear":
                    b = cal.clear_roi(app.cal_site(body.get("config", "")))
                    self._json({"ok": True, "backup": str(b) if b else None})
                else:
                    self._json(post_routes[route](body))
            except NeedsConfirm as e:
                self._json({"error": str(e), "confirm": True}, 409)
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except (KeyError, ValueError, OSError) as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


# --------------------------------------------------------------------------- window
class Bridge:
    """What the page may ask the window for (pywebview js_api): native file dialogs and,
    with the custom frame, the window buttons."""

    def __init__(self):
        self._window = None
        self._custom = False

    def frame(self) -> dict:
        from . import desktop_frame

        return {"custom": self._custom,
                "maximized": self._custom and desktop_frame.is_maximized(self._window)}

    def minimize(self) -> None:
        self._window.minimize()

    def toggle_maximize(self) -> None:
        from . import desktop_frame

        if desktop_frame.is_maximized(self._window):
            self._window.restore()
        else:
            self._window.maximize()

    def close(self) -> None:
        self._window.destroy()          # asks first while jobs run (the closing handler)

    def fullscreen(self) -> None:
        """Full screen on / off (the Live view's F)."""
        self._window.toggle_fullscreen()

    def resize(self, edge: str) -> None:
        from . import desktop_frame

        if self._custom:
            desktop_frame.resize_from(self._window, str(edge))

    def pick(self, kind: str = "file", what: str = "") -> str | None:
        import webview

        types = {"video": ("Video (*.mp4;*.mov;*.MOV;*.m4v;*.avi)", "All files (*.*)"),
                 "geotiff": ("GeoTIFF (*.tif;*.tiff)", "All files (*.*)")}.get(what, ())
        dlg = getattr(webview, "FileDialog", None)
        if kind == "folder":
            mode = dlg.FOLDER if dlg else webview.FOLDER_DIALOG
            res = self._window.create_file_dialog(mode)
        else:
            mode = dlg.OPEN if dlg else webview.OPEN_DIALOG
            res = self._window.create_file_dialog(mode, file_types=types)
        if not res:
            return None
        return str(res[0] if isinstance(res, (list, tuple)) else res)


# the app icon: the mark of the wordmark (desktop_web/index.html), the long ō of mōtus, a ring
# under a line that runs past it; light on a near-black tile with a hairline edge
ICON_FILE = "motus-1.ico"                 # a new name when the icon changes (it is cached)
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def _icon_mark(px: int) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """The ring (cx, cy, r), the line (x0, x1, y) and the stroke width, in pixels at this size."""
    if px == 16:                          # its own drawing: on whole pixels, a clear gap
        return (8, 10, 3.5), (2.5, 13.5, 3.5), 1.15
    unit = px / 32                        # the tile is 32 units; the mark's 32-unit box sits in
    area = 24 if px <= 32 else 20         # its middle 20 units (24 at 32 px and below)
    k = unit * area / 32
    o = (32 - area) / 2 * unit
    (cx, cy, r), (x0, x1, y) = (o + 16 * k, o + 19 * k, 8 * k), (o + 3 * k, o + 29 * k, o + 6.5 * k)
    if px < 64:                           # thin strokes on pixel centres: crisp, not grey
        cx, cy, r, y = round(cx), round(cy), round(r - 0.5) + 0.5, round(y - 0.5) + 0.5
    return (cx, cy, r), (x0, x1, y), max(1.25 * k, 1.15)   # never under 1.15 px


def _icon_image(px: int):
    """One size of the icon, drawn 16x larger and box-averaged down."""
    from PIL import Image, ImageDraw

    ss, n = 16, px * 16
    rx = 7 * px / 32                      # corner radius; the hairline is 1 px at every size
    outer = Image.new("L", (n, n), 0)
    ImageDraw.Draw(outer).rounded_rectangle((0, 0, n - 1, n - 1), radius=(rx + 0.5) * ss, fill=255)
    inner = Image.new("L", (n, n), 0)
    ImageDraw.Draw(inner).rounded_rectangle((ss, ss, n - ss - 1, n - ss - 1),
                                            radius=(rx - 0.5) * ss, fill=255)
    (cx, cy, r), (x0, x1, y), sw = _icon_mark(px)
    mark = Image.new("L", (n, n), 0)
    d, h = ImageDraw.Draw(mark), sw / 2
    d.ellipse(((cx - r - h) * ss, (cy - r - h) * ss, (cx + r + h) * ss, (cy + r + h) * ss),
              outline=255, width=round(sw * ss))
    d.rectangle((x0 * ss, (y - h) * ss, x1 * ss, (y + h) * ss), fill=255)
    for x in (x0, x1):                    # round caps
        d.ellipse(((x - h) * ss, (y - h) * ss, (x + h) * ss, (y + h) * ss), fill=255)
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    for m, rgb in ((outer, (0x3A, 0x3A, 0x40)), (inner, (0x0B, 0x0B, 0x0D)), (mark, (0xED, 0xED, 0xED))):
        layer = Image.new("RGBA", (px, px), rgb + (255,))
        layer.putalpha(m.reduce(ss))
        im = Image.alpha_composite(im, layer)
    return im


def icon_path() -> Path | None:
    """The app icon (.ico, each size drawn at its own size), made once in the settings folder."""
    p = links.settings_dir() / ICON_FILE
    if p.exists():
        return p
    try:
        ims = [_icon_image(px) for px in ICON_SIZES]
        p.parent.mkdir(parents=True, exist_ok=True)
        ims[-1].save(p, sizes=[(px, px) for px in ICON_SIZES], append_images=ims[:-1])
        return p
    except Exception:  # noqa: BLE001 (no icon is fine)
        return None


def make_shortcut(root: Path) -> list[Path]:
    """Desktop and Start-menu shortcuts that open the app on this project (Windows)."""
    if not sys.platform.startswith("win"):
        raise SystemExit("shortcuts are made on Windows only")
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe") if exe.with_name("pythonw.exe").exists() else exe
    ico = icon_path()
    targets = [Path.home() / "Desktop" / "motus.lnk",
               Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
               / "motus.lnk"]
    made = []
    for lnk in targets:
        if not lnk.parent.exists():
            continue
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GT_LNK); "
              "$s.TargetPath = $env:GT_EXE; $s.Arguments = $env:GT_ARGS; "
              "$s.WorkingDirectory = $env:GT_ROOT; $s.Description = 'motus'; "
              "if ($env:GT_ICO) { $s.IconLocation = $env:GT_ICO }; $s.Save()")
        env = dict(os.environ, GT_LNK=str(lnk), GT_EXE=str(pyw), GT_ROOT=str(root),
                   GT_ARGS=f'-m groundtrack app --root "{root}"', GT_ICO=str(ico or ""))
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, check=True,
                       creationflags=NO_WINDOW)
        made.append(lnk)
    return made


def _ask_quit(n: int) -> bool:
    """Native yes / no before closing while jobs run (they are stopped)."""
    if not sys.platform.startswith("win"):
        return True
    import ctypes

    jobs, them = ("jobs", "them") if n != 1 else ("job", "it")
    msg = (f"{n} {jobs} still running (a run, a calibration or a kit). Closing motus "
           f"stops {them}.\n\nClose anyway?")
    return ctypes.windll.user32.MessageBoxW(None, msg, "motus", 0x34) == 6  # yes / no


def _is_webview_busy(exc: BaseException) -> bool:
    """WebView2 error 0x800700AA (ERROR_BUSY): its user-data folder is in use."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = " ".join(str(a) for a in (*exc.args, exc)).upper()
        if "800700AA" in text or "-2147024726" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def serve(root: Path, port: int = 0, window: bool = True, debug: bool = False,
          custom_frame: bool = True) -> None:
    app = DesktopApp(root)
    srv = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_desktop_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    app.check_device()
    print(f"motus app on {url}  (project {app.root})")
    try:
        try:
            import webview
        except ImportError:
            webview = None
        if not window or webview is None:
            if window:
                print("pywebview is not installed (pip install pywebview): opening the browser")
            import webbrowser

            if window:
                webbrowser.open(url)
            while True:
                time.sleep(3600)
        if sys.platform.startswith("win"):          # its own taskbar button and icon
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("groundtrack.app")
        # a user-wide WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS with --disable-gpu renders in software
        # (the Live view at ~15 fps): this window drops that one flag and keeps the others
        flags = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
        if "--disable-gpu" in flags.split():
            os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = " ".join(
                a for a in flags.split() if a != "--disable-gpu")
        bridge = Bridge()
        on_windows = sys.platform.startswith("win")
        # on Windows it appears once sized to the screen and framed (no title-bar flash)
        win = webview.create_window("motus", url, js_api=bridge, width=1500, height=960,
                                    min_size=(1100, 700), background_color="#000000",
                                    text_select=True, hidden=on_windows)
        bridge._window = win
        shown = {"done": not on_windows}

        def show_once():
            if not shown["done"]:
                shown["done"] = True
                win.show()

        if on_windows:
            threading.Timer(15, show_once).start()       # whatever happens, show it

        def closing():
            n = sum(1 for j in app.jobs.values() if j.status == "running")
            return True if n == 0 else _ask_quit(n)

        win.events.closing += closing

        def push_frame():
            st = bridge.frame()
            win.evaluate_js(f"window.gtFrame && window.gtFrame({json.dumps(st)})")

        def loaded():
            from . import desktop_frame

            if not shown["done"]:
                desktop_frame.fit_to_screen(win, 1500, 960)
            if custom_frame and not bridge._custom:
                bridge._custom = desktop_frame.apply(win)
            push_frame()
            show_once()

        win.events.loaded += loaded
        win.events.maximized += push_frame
        win.events.restored += push_frame
        ico = icon_path()
        storage = links.settings_dir() / "webview"
        try:
            webview.start(gui="edgechromium" if sys.platform.startswith("win") else None,
                          debug=debug, private_mode=False, storage_path=str(storage),
                          icon=str(ico) if ico else None)
        except Exception as e:
            if not _is_webview_busy(e):
                raise
            print("The motus window could not start: another motus window is still open "
                  "or closing. Close it and try again. (WebView2 0x800700AA)\n"
                  "To use the browser instead, run `groundtrack app --no-window` and open "
                  "the URL it prints.", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        for j in app.jobs.values():
            j.stop()
        app.stop_hub()                              # the Live view's phone hub
        srv.shutdown()
        srv.server_close()
