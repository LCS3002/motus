r"""Builds the POP scene of a groundtrack Houdini kit. Runs INSIDE Houdini (hython, or Houdini's
Python Shell), not as part of groundtrack:

    hython hou_pop_build.py <kit folder>        (writes <site>_pop.hip / .hipnc there)
    Python Shell:  import os; os.environ["GT_KIT"] = r"<kit folder>"; exec(open(r"<kit folder>\hou_pop_build.py").read())

The kit folder holds modules.json and fields/*.csv (written by groundtrack houdini-pop). One
Geometry object per module (the whole site, each clip), each:

    field (Python SOP: the field cells + emitter seeds, group menu)  ->  place (move / rotate /
    scale the module)  ->  OUT_CELLS / OUT_FIELD (velocity volume v) / OUT_SEEDS  ->  sim (POP
    network: source at the seeds, advect by the field, optional wind, kill outside the field)
    ->  particles (coloured by speed)  ->  trails  ->  OUT

plus an object "combined" that merges the placed modules listed in its object_merge: where
they overlap their fields blend (density-weighted), and one simulation runs through all.

From hython it builds a new scene and saves it in the kit. Inside an open scene (Houdini's Python
Shell, or GT_ADD=1) it only adds: everything goes into one subnet /obj/groundtrack_<site>, nothing
else in the scene is touched (not even the frame rate), and the scene is not saved.
"""
import json
import os
import re
import sys

import hou

KIT = os.path.abspath(os.environ.get("GT_KIT") or (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else
                      os.path.dirname(globals().get("__file__", "."))))
ADD = os.environ.get("GT_ADD", "1" if hou.isUIAvailable() else "0") == "1"

FIELD_SOP = r'''
import csv
import json
import math
import os
import random

import hou

node = hou.pwd()
geo = node.geometry()
fields = json.loads(node.evalParm("fields"))
group = node.evalParm("group")
rel = fields.get(group) or next(iter(fields.values()))
# the kit folder first (the scene may be saved anywhere), then next to the scene
kit = node.evalParm("kit")
path = rel if os.path.isabs(rel) else next(
    (p for p in (os.path.join(kit, rel), os.path.join(hou.expandString("$HIP"), rel)) if os.path.exists(p)),
    os.path.join(kit, rel))
conf_min = node.evalParm("min_confidence")
with open(path, newline="") as f:
    rows = [r for r in csv.DictReader(f) if float(r["confidence"]) >= conf_min]
for name in ("speed", "density", "confidence", "smax", "pscale", "w"):
    geo.addAttrib(hou.attribType.Point, name, 0.0)
geo.addAttrib(hou.attribType.Point, "v", (0.0, 0.0, 0.0))
geo.addAttrib(hou.attribType.Point, "vel", (0.0, 0.0, 0.0))     # v * w, for the volume
for name in ("v", "vel"):        # vectors: moving / scaling / mirroring the field turns them with it
    geo.findPointAttrib(name).setOption("type", "vector")
geo.addAttrib(hou.attribType.Global, "field_file", "")
geo.setGlobalAttribValue("field_file", rel)
if rows:
    cell = float(rows[0]["cell"])
    cells = geo.createPointGroup("cells")
    pts = geo.createPoints([hou.Vector3(float(r["x"]), float(r.get("y") or 0.0), float(r["z"])) for r in rows])
    cells.add(pts)
    # the flow follows its sheet: where the field slopes (terrain heights, ramps, folds) and the
    # file has no vertical velocity, the vertical part is the slope along the flow - otherwise
    # particles slide straight out of a sloping field and stall
    xyz = [(float(r["x"]), float(r.get("y") or 0.0), float(r["z"])) for r in rows]
    vel3 = [[float(r["vx"]), float(r.get("vy") or 0.0), float(r["vz"])] for r in rows]
    grid = {}
    for i, (x, y, z) in enumerate(xyz):
        grid.setdefault((round(x / cell), round(z / cell)), []).append(i)

    def height(i, di, dk):
        """Height of the same layer's neighbouring cell (stacked layers are kept apart)."""
        x, y, z = xyz[i]
        cand = grid.get((round(x / cell) + di, round(z / cell) + dk))
        if not cand:
            return None
        j = min(cand, key=lambda j: abs(xyz[j][1] - y))
        return xyz[j][1] if abs(xyz[j][1] - y) < 3 * cell else None

    def slope(i, a, b):
        y = xyz[i][1]
        hp, hm = height(i, *a), height(i, *b)
        steps = (hp is not None) + (hm is not None)
        return 0.0 if not steps else ((hp if hp is not None else y) - (hm if hm is not None else y)) / (steps * cell)

    for i, v in enumerate(vel3):
        if abs(v[1]) < 1e-9:
            v[1] = v[0] * slope(i, (1, 0), (-1, 0)) + v[2] * slope(i, (0, 1), (0, -1))
    geo.setPointFloatAttribValues("v", [c for v in vel3 for c in v])
    for name in ("speed", "density", "confidence", "smax"):
        geo.setPointFloatAttribValues(name, [float(r[name]) for r in rows])
    geo.setPointFloatAttribValues("pscale", [0.75 * cell] * len(rows))
    # weight of each cell in the volume: its density relative to the module's median, so
    # where modules overlap the busier flow counts more
    dens = sorted(float(r["density"]) for r in rows)
    med = max(dens[len(dens) // 2], 1e-12)
    w = [min(max(float(r["density"]) / med, 0.05), 20.0) for r in rows]
    geo.setPointFloatAttribValues("w", w)
    geo.setPointFloatAttribValues("vel", [c * wi for v, wi in zip(vel3, w) for c in v])
    # emitter seeds where people / vehicles were: busier cells more (square root of the
    # density, so quiet places still emit), jittered inside the cell
    rnd = random.Random(node.evalParm("seed"))
    w = [math.sqrt(max(float(r["density"]), 0.0)) * float(r["confidence"]) for r in rows]
    if sum(w) > 0:
        pick = rnd.choices(range(len(rows)), weights=w, k=max(1, node.evalParm("seeds")))
        seeds = geo.createPointGroup("seeds")
        seeds.add(geo.createPoints([hou.Vector3(float(rows[i]["x"]) + (rnd.random() - 0.5) * cell,
                                                float(rows[i].get("y") or 0.0),
                                                float(rows[i]["z"]) + (rnd.random() - 0.5) * cell)
                                    for i in pick]))
'''

COLOUR_VEX = r'''
// speed colour on each group's own scale: the field cell's smax (input 1 = the cells)
vector ramp(float t) {
    vector c[] = array(%s);
    float x = clamp(t, 0, 1) * (len(c) - 1);
    int i = min(int(floor(x)), len(c) - 2);
    return lerp(c[i], c[i + 1], x - i);
}
int pt = nearpoint(1, @P);
float smax = pt >= 0 ? point(1, "smax", pt) : 2.5;
f@speed = length(v@v);
v@Cd = ramp(f@speed / max(smax, 1e-6));
f@pscale = chf("dot_size");
'''

KILL_VEX = """// outside the field (no cell nearby: coverage w = 0), or stalled
string fld = "op:" + opfullpath("../../OUT_FIELD");
if (volumesample(fld, "w", @P) < 1e-4 || (length(v@v) < chf("min_speed") && @age > 0.5))
    dead = 1;"""


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _parm(node, tmpl):
    g = node.parmTemplateGroup()
    g.append(tmpl)
    node.setParmTemplateGroup(g)


def field_node(geo, mod):
    py = geo.createNode("python", "field")
    py.parm("python").set(FIELD_SOP)
    groups = list(mod["fields"])
    _parm(py, hou.StringParmTemplate("fields", "Field files (JSON)", 1,
                                     default_value=(json.dumps(mod["fields"]),)))
    _parm(py, hou.StringParmTemplate("kit", "Kit folder", 1, default_value=(KIT,)))
    _parm(py, hou.StringParmTemplate("group", "Group", 1, default_value=(mod["group"],),
                                     menu_items=groups, menu_labels=groups))
    _parm(py, hou.FloatParmTemplate("min_confidence", "Min confidence", 1,
                                    default_value=(mod.get("min_confidence", 0.2),),
                                    min=0.0, max=1.0))
    _parm(py, hou.IntParmTemplate("seeds", "Emitter seeds", 1,
                                  default_value=(mod.get("seeds", 20000),), min=1, max=200000))
    _parm(py, hou.IntParmTemplate("seed", "Random seed", 1, default_value=(1,)))
    return py


def scale_of(place):
    """Expression: the uniform scale of the place node at relative path `place` (its smallest axis)."""
    return ('(abs(ch("{0}/scale")) * min(abs(ch("{0}/sx")), min(abs(ch("{0}/sy")), abs(ch("{0}/sz")))))'
            .format(place))


def chain(geo, upstream, mod, colours):
    """place -> field volume / seeds -> POP sim -> particles + trails -> OUT. Scaling the field in
    place scales everything with it: its vectors, voxels, the pull (Air Resistance: the force is
    quadratic in the speed difference, so 1 / scale), the slow-particle kill and the dots."""
    place = geo.createNode("xform", "place")
    place.setInput(0, upstream)
    place.parmTuple("p").set((mod["centre"][0], 0.0, mod["centre"][1]))   # pivot: the module
    cells = geo.createNode("blast", "cells_only")
    cells.setInput(0, place)
    cells.parm("group").set("seeds")
    cells.parm("grouptype").set("points")
    out_cells = geo.createNode("null", "OUT_CELLS")
    out_cells.setInput(0, cells)
    vol = geo.createNode("volumerasterizeattributes", "field_volume")
    vol.setInput(0, out_cells)
    vol.parm("attributes").set("vel w")      # both averaged over the cells nearby ...
    vol.parm("densityattrib").set("")
    vol.parm("normalize").set(0)
    # voxels and splats follow place's scale, so a scaled-down field keeps its detail
    vol.parm("voxelsize").setExpression("max(%g * %s, 0.0001)" % (mod["cell"] * 0.5, scale_of("../place")))
    vol.parm("particlescale").setExpression(scale_of("../place"))
    div = geo.createNode("volumewrangle", "weighted_average")
    div.setInput(0, vol)
    div.parm("snippet").set("v@vel /= max(f@w, 1e-9);   // ... so vel = sum(w v) / sum(w)")
    out_field = geo.createNode("null", "OUT_FIELD")
    out_field.setInput(0, div)
    seeds = geo.createNode("blast", "seeds_only")
    seeds.setInput(0, place)
    seeds.parm("group").set("cells")
    seeds.parm("grouptype").set("points")
    out_seeds = geo.createNode("null", "OUT_SEEDS")
    out_seeds.setInput(0, seeds)

    sim = geo.createNode("dopnet", "sim")
    out = sim.node("output") or sim.createNode("output")
    obj = sim.createNode("popobject", "particles")
    solver = sim.createNode("popsolver", "solver")
    src = sim.createNode("popsource", "source")
    src.parm("emittype").set("point")
    src.parm("usecontextgeo").set("none")
    src.parm("soppath").set("../../OUT_SEEDS")
    src.parm("impulseactiveate").set(0)
    src.parm("constantactivate").set(1)
    src.parm("constantrate").set(mod["rate"])
    src.parm("life").set(mod["life"])
    src.parm("lifevar").set(mod["life"] * 0.3)
    adv = sim.createNode("popadvectbyvolumes", "follow_field")
    adv.setInput(0, src)
    adv.parm("velsource").set("sop")
    adv.parm("soppath").set("../../OUT_FIELD")
    adv.parm("fieldname").set("vel")
    # "vel": particles take the field's velocity; "force": the field is a force field (they
    # are pushed towards its velocity, Air Resistance sets how hard)
    adv.parm("advecttype").set(mod.get("advect", "vel"))
    if mod.get("advect") == "force" and adv.parm("airresist") is not None:
        adv.parm("airresist").setExpression("%g / max(%s, 0.0001)" % (mod.get("airresist", 1.0), scale_of("../../place")))
    adv.parm("advectmethod").set("midpoint")
    wind = sim.createNode("popwind", "turbulence")
    wind.setInput(0, adv)
    wind.parm("amp").setExpression("%g * %s" % (mod["noise"], scale_of("../../place")))
    wind.bypass(True)                          # switch on for turbulence
    kill = sim.createNode("popkill", "kill_outside")
    kill.setInput(0, wind)
    kill.parm("enablerule").set(1)
    kill.parm("rulecode").set(KILL_VEX)
    _parm(kill, hou.FloatParmTemplate("min_speed", "Min speed", 1,
                                      default_value=(mod["min_speed"],)))
    kill.parm("min_speed").setExpression("%g * %s" % (mod["min_speed"], scale_of("../../place")))
    solver.setInput(0, obj)
    solver.setInput(2, kill)
    out.setInput(0, solver)
    sim.layoutChildren()
    sim.setDisplayFlag(False)

    imp = geo.createNode("dopimport", "import_particles")
    imp.parm("doppath").set("../sim")
    imp.parm("objpattern").set("*")
    col = geo.createNode("attribwrangle", "particles")
    col.setInput(0, imp)
    col.setInput(1, out_cells)
    col.parm("snippet").set(COLOUR_VEX % colours)
    _parm(col, hou.FloatParmTemplate("dot_size", "Dot size", 1,
                                     default_value=(mod["cell"] * 0.25,)))
    col.parm("dot_size").setExpression("%g * %s" % (mod["cell"] * 0.25, scale_of("../place")))
    trail = geo.createNode("trail", "trail_points")      # the last frames' positions ...
    trail.setInput(0, col)
    trail.parm("result").set("preserve")
    trail.parm("length").set(mod["trail_frames"])
    lines = geo.createNode("add", "trails")             # ... joined into one line per particle
    lines.setInput(0, trail)
    lines.parm("prims").set(1)
    lines.parm("switcher1").set(1)                     # polygons "By Group" ...
    lines.parm("add").set("attribute")                 # ... by attribute: id
    lines.parm("attrname").set("id")
    lines.parm("remove").set(1)
    merge = geo.createNode("merge", "particles_and_trails")
    merge.setInput(0, col)
    merge.setInput(1, lines)
    out = geo.createNode("null", "OUT")
    out.setInput(0, merge)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    geo.layoutChildren()
    return place


def main():
    with open(os.path.join(KIT, "modules.json"), encoding="utf-8") as f:
        kit = json.load(f)
    colours = ", ".join("{%.4f, %.4f, %.4f}" % tuple(_lin(x) for x in c)
                        for _, c in kit["ramp"])
    if ADD:     # into the open scene: one subnet, nothing else touched
        base = "groundtrack_" + (re.sub(r"[^A-Za-z0-9_]", "_", kit["site"]).strip("_") or "field")
        name, i = base, 1
        while hou.node("/obj/" + name) is not None:
            i += 1
            name = "%s%d" % (base, i)
        others = hou.node("/obj").children()
        obj = hou.node("/obj").createNode("subnet", name)
        # right of the scene's nodes (moveToGoodPosition would move them too)
        obj.setPosition(hou.Vector2(max([n.position()[0] for n in others], default=0.0) + 3.0,
                                    max([n.position()[1] for n in others], default=0.0)))
    else:       # a new scene
        hou.hipFile.clear(suppress_save_prompt=True)
        hou.setFps(kit.get("fps", 24))
        hou.playbar.setFrameRange(1, kit.get("frames", 480))
        hou.playbar.setPlaybackRange(1, kit.get("frames", 480))
        obj = hou.node("/obj")
    placed = {}
    for k, mod in enumerate(kit["modules"]):
        geo = obj.createNode("geo", mod["name"])
        for c in geo.children():
            c.destroy()
        placed[mod["name"]] = chain(geo, field_node(geo, mod), mod, colours)
        geo.setDisplayFlag(bool(mod.get("visible")))
        geo.setPosition(hou.Vector2(3.0 * (k % 6), -1.5 * (k // 6)))
    # combined: modules merged, fields blended where they overlap (after moving them)
    comb = obj.createNode("geo", "combined")
    for c in comb.children():
        c.destroy()
    om = comb.createNode("object_merge", "modules")
    names = kit.get("combined") or [m["name"] for m in kit["modules"]][:2]
    om.parm("numobj").set(len(names))
    for i, n in enumerate(names, 1):
        om.parm("objpath%d" % i).set(placed[n].path())
    first = next(m for m in kit["modules"] if m["name"] == names[0])
    chain(comb, om, dict(first, centre=first["centre"]), colours)
    comb.setDisplayFlag(bool(kit.get("show_combined")))
    comb.setPosition(hou.Vector2(0.0, -1.5 * (len(kit["modules"]) // 6 + 1)))
    note = obj.createStickyNote("about")
    note.setText(kit["note"])
    note.setSize(hou.Vector2(9.0, 4.0))
    note.setPosition(hou.Vector2(-10.0, 0.0))
    if kit.get("vdb"):
        for name in [m["name"] for m in kit["modules"]] + ["combined"]:
            fld = hou.node("%s/%s/OUT_FIELD" % (obj.path(), name))
            if fld is not None:
                path = os.path.join(KIT, "%s_%s_vel.vdb" % (kit["site"], name))
                try:
                    fld.geometry().saveToFile(path)
                    print("saved", path)
                except hou.Error as e:
                    print("could not save", path, e)
    print("force field (velocity volume, field 'vel'): %s/OUT_FIELD" % comb.path())
    if ADD:
        print("added %s - save the scene to keep it" % obj.path())
        return
    ext = ".hip" if hou.licenseCategory() == hou.licenseCategoryType.Commercial else (
        ".hiplc" if hou.licenseCategory() == hou.licenseCategoryType.Indie else ".hipnc")
    out = os.path.join(KIT, kit["site"] + "_pop" + ext)
    hou.hipFile.save(out)
    print("saved", out)


main()
