"""Houdini, Rhino and Grasshopper for the desktop app: where they are installed, whether they
are running, the groundtrack-grasshopper plugin's version, and the Houdini kits, Rhino models
and Grasshopper definitions found in the project and in library folders.

Nothing here writes into a scene: files are only opened (as a double-click would), and the
Python Shell line only adds a subnet when the user pastes it into Houdini themselves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

KIT_FILES = ("modules.json", "hou_pop_build.py")
HIP_EXT = {".hip", ".hipnc", ".hiplc"}
RHINO_EXT = {".3dm"}
GH_EXT = {".gh", ".ghx"}
OPENABLE = HIP_EXT | RHINO_EXT | GH_EXT
# folders that only hold outputs (images, fields, frames): never worth walking into
SKIP_DIRS = {"fields", "images", "videos", "labels", "frames", "process", "extras", "dtm",
             "lexicon", "netfx", "backup", "__pycache__", ".git", ".venv", "node_modules",
             "data", "footage", "maps", "render", "References"}
PLUGIN_RELEASES = "https://github.com/LCS3002/groundtrack-grasshopper/releases"


# --------------------------------------------------------------------------- settings
def settings_dir() -> Path:
    base = os.environ.get("APPDATA") if sys.platform.startswith("win") else None
    return Path(base) / "groundtrack" if base else Path.home() / ".config" / "groundtrack"


class Settings:
    """The app's own settings (library folders), kept per user, outside the project."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else settings_dir() / "desktop.json"
        self.data: dict = {"libraries": []}
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    @property
    def libraries(self) -> list[str]:
        return [str(p) for p in self.data.get("libraries", [])]

    def add_library(self, folder: str) -> list[str]:
        p = Path(str(folder).strip().strip('"')).resolve()
        if not p.is_dir():
            raise ValueError(f"not a folder: {p}")
        libs = self.libraries
        if str(p) not in libs:
            libs.append(str(p))
        self.data["libraries"] = libs
        self.save()
        return libs

    def remove_library(self, folder: str) -> list[str]:
        self.data["libraries"] = [p for p in self.libraries if p != str(folder)]
        self.save()
        return self.libraries

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- installs
def houdini_install() -> dict | None:
    """The newest Houdini: {version, bin, houdini, hython}."""
    from .houdini_pop import find_hython

    hython = find_hython()
    if hython is None:
        return None
    bindir = hython.parent
    exe = next((bindir / n for n in ("houdini.exe", "houdinifx.exe", "houdinicore.exe",
                                     "happrentice.exe", "houdini") if (bindir / n).exists()), None)
    version = bindir.parent.name.replace("Houdini ", "")
    return {"version": version, "bin": str(bindir), "houdini": str(exe) if exe else None,
            "hython": str(hython)}


def rhino_install() -> dict | None:
    """The newest Rhino: {version, exe}."""
    if not sys.platform.startswith("win"):
        return None
    found = sorted(Path(r"C:\Program Files").glob("Rhino */System/Rhino.exe"))
    if not found:
        return None
    return {"version": found[-1].parent.parent.name.replace("Rhino ", ""), "exe": str(found[-1])}


def pe_version(path: Path) -> str | None:
    """ProductVersion (else FileVersion) from a Windows binary's version resource, e.g. a .gha."""
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    for key in ("ProductVersion", "FileVersion"):
        k = (key + "\0").encode("utf-16-le")
        i = b.find(k)
        if i < 0:
            continue
        j = i + len(k)
        j += (-j) % 4                                    # values are 32-bit aligned
        end = b.find(b"\0\0", j)
        while end != -1 and (end - j) % 2:
            end = b.find(b"\0\0", end + 1)
        if end > j:
            v = b[j:end].decode("utf-16-le", errors="replace")
            return v.split("+", 1)[0]
    return None


def grasshopper_plugin() -> dict:
    """The groundtrack-grasshopper plugin: installed copy (Libraries), Yak package, versions."""
    appdata = os.environ.get("APPDATA")
    out = {"installed": False, "releases": PLUGIN_RELEASES}
    if not appdata:
        return out
    libs = Path(appdata) / "Grasshopper" / "Libraries"
    gha = next((p for p in [libs / "groundtrack.gha", *sorted(libs.glob("groundtrack/*.gha"))]
                if p.exists()), None)
    yak_root = Path(appdata) / "McNeel" / "Rhinoceros" / "packages" / "8.0" / "groundtrack"
    yak = sorted(p.name for p in yak_root.glob("*") if p.is_dir()) if yak_root.exists() else []
    if gha:
        out.update(installed=True, path=str(gha), version=pe_version(gha),
                   libraries=str(libs))
    if yak:
        out.update(installed=True, yak=yak)
    # both at once load the plugin twice: Grasshopper refuses the second copy
    out["conflict"] = bool(gha and yak)
    return out


_RUNNING: dict = {"t": 0.0, "names": set()}


def running_apps() -> dict:
    """Is Houdini / Rhino open? (cached for 3 s)"""
    if time.time() - _RUNNING["t"] > 3:
        names: set[str] = set()
        if sys.platform.startswith("win"):
            try:
                r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                                   text=True, timeout=10,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                names = {ln.split('","', 1)[0].strip('"').lower()
                         for ln in r.stdout.splitlines() if ln.startswith('"')}
            except (OSError, subprocess.SubprocessError):
                pass
        _RUNNING.update(t=time.time(), names=names)
    n = _RUNNING["names"]
    return {"houdini": any(x.startswith(("houdini", "happrentice", "hindie")) for x in n),
            "rhino": "rhino.exe" in n}


# --------------------------------------------------------------------------- kits
def shell_line(kit: str | Path) -> str:
    """The Houdini Python Shell line that adds the kit to the open scene (same as the
    Grasshopper plugin's Houdini Export): one subnet /obj/groundtrack_<site>, nothing else."""
    kit = str(Path(kit))
    return (f'import os; os.environ["GT_KIT"] = r"{kit}"; '
            f'exec(open(r"{os.path.join(kit, "hou_pop_build.py")}").read())')


def is_kit(folder: Path) -> bool:
    return all((folder / f).exists() for f in KIT_FILES)


def kit_info(folder: Path) -> dict:
    folder = Path(folder)
    d = {"path": str(folder), "name": folder.name, "site": folder.name, "modules": [],
         "source": "Grasshopper" if any(folder.glob("*_field.geo")) else "motus",
         "mtime": (folder / "modules.json").stat().st_mtime}
    try:
        m = json.loads((folder / "modules.json").read_text(encoding="utf-8"))
        d["site"] = m.get("site") or folder.name
        d["modules"] = [x.get("name", "") for x in m.get("modules", [])]
        d["combined"] = m.get("combined") or []
        d["origin"] = m.get("origin")
    except (OSError, ValueError, AttributeError):
        d["error"] = "modules.json unreadable"
    hips = sorted((p for p in folder.glob("*") if p.suffix.lower() in HIP_EXT),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    d["scene"] = str(hips[0]) if hips else None
    d["scene_mtime"] = hips[0].stat().st_mtime if hips else None
    d["shell_line"] = shell_line(folder)
    return d


def is_composite(folder: Path) -> bool:
    """A combined plan (`groundtrack composite`): only those have the pooled master field."""
    return (folder / "houdini" / "houdini_field.py").exists() and \
        (folder / "houdini" / "vector_field_master.csv").exists()


def scan(folders: list[Path], max_depth: int = 5, budget_s: float = 4.0) -> dict:
    """Kits, composites, Houdini scenes, Rhino models and Grasshopper definitions under the
    folders (walks at most max_depth levels and budget_s seconds)."""
    out: dict = {"kits": [], "composites": [], "scenes": [], "models": [], "definitions": []}
    seen: set[str] = set()
    t0 = time.time()

    def add(key, p: Path, **extra):
        s = str(p)
        if s in seen:
            return
        seen.add(s)
        try:
            st = p.stat()
        except OSError:
            return
        out[key].append({"path": s, "name": p.name, "mtime": st.st_mtime, **extra})

    for root in folders:
        root = Path(root)
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack and time.time() - t0 < budget_s:
            d, depth = stack.pop()
            if is_kit(d):
                if str(d) not in seen:
                    seen.add(str(d))
                    out["kits"].append(kit_info(d))
                continue
            if is_composite(d):
                add("composites", d)
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if depth < max_depth and e.name not in SKIP_DIRS and not e.name.startswith("."):
                        stack.append((Path(e.path), depth + 1))
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                if ext in HIP_EXT and not e.name.endswith(".bak"):
                    add("scenes", Path(e.path))
                elif ext in RHINO_EXT:
                    add("models", Path(e.path))
                elif ext in GH_EXT:
                    add("definitions", Path(e.path))
    for v in out.values():
        v.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    # a kit's own scene is listed with the kit, not again on its own
    kit_scenes = {k["scene"] for k in out["kits"] if k.get("scene")}
    out["scenes"] = [s for s in out["scenes"] if s["path"] not in kit_scenes]
    out["truncated"] = time.time() - t0 >= budget_s
    return out


# --------------------------------------------------------------------------- open
def _exe_for(path: Path) -> str | None:
    if path.suffix.lower() in HIP_EXT:
        h = houdini_install()
        return h and h.get("houdini")
    r = rhino_install()
    return r and r.get("exe")


def open_file(path: Path) -> None:
    """Open a scene / model / definition as a double-click would (the program Windows has for
    it), else in the installed Houdini / Rhino."""
    path = Path(path)
    if path.suffix.lower() not in OPENABLE:
        raise ValueError(f"won't open {path.suffix} files")
    if sys.platform.startswith("win"):
        try:
            os.startfile(str(path))  # noqa: S606 (local desktop action)
            return
        except OSError:
            pass
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    exe = _exe_for(path)
    if not exe:
        raise ValueError(f"no program found for {path.suffix} files")
    subprocess.Popen([exe, str(path)], cwd=str(path.parent))
