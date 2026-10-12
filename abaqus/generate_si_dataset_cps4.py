# -*- coding: utf-8 -*-
"""
SI DATASET with a STRUCTURED, SYMMETRIC CPS4 MESH - 49 plate-with-hole
geometries x 23 loads, mm / N / MPa.

    27 training geometries  +  4 VALIDATION geometries  +  13 + 5 test geometries

    cd /d <repo>\\data\\odb_si_cps4                 (an EMPTY folder - make it first)
    abaqus python ..\\..\\abaqus\\generate_si_dataset_cps4.py --mesh-only     1. build + check all 49 meshes
    abaqus python ..\\..\\abaqus\\generate_si_dataset_cps4.py                 2. build + solve everything

Same plate, material, loads, geometries and train/test split as
generate_si_dataset.py (the CPS3 version). What changes is the MESH, and
4 extra VALIDATION geometries (see below).

VALIDATION GEOMETRIES (new in this version)
    E_H1p75 (a/b = 1.75), E_V3p5 (b/a = 3.5), R_W28 (28 x 64 mm), R_H36 (64 x 36 mm)
    New shapes INSIDE each family's range, used by NEITHER training NOR the
    tests. scripts/train_b1_val.py scores them every few epochs (the
    validation curve of R2 vs epoch), so over-fitting can be seen while the
    model trains, and the test geometries stay untouched until the end.
    The polygon family has no free in-range value (n = 5..10 are all used),
    so it has no validation shape.
    If you already solved the 45 geometries in this folder, just run this
    script again: finished jobs are skipped, only the 4 x 23 new ones run.

WHY THIS SCRIPT MESHES BY ITSELF INSTEAD OF IN ABAQUS/CAE
    Abaqus' free mesher never guarantees a mesh that is symmetric about the
    hole, and a structured (mapped) quad mesh around 49 different holes would
    need a fragile set of partitions per shape. So the mesh is built here,
    in plain Python, one quarter (or one half) at a time and then MIRRORED -
    the left and right halves (and the top and bottom halves, where the hole
    allows it) are exact mirror images, node for node. The mesh is written to
    an Abaqus input file (.inp) and solved with the normal 'abaqus job=...'
    command. No CAE licence is needed for meshing.

THE MESH  (all sizes in mm, all settings in the block below)
    VERSION 2: about half the nodes of version 1, and a simpler pattern
    around rectangles and polygons.
    O-grid around the hole, out to a 250 x 250 mm box (the full plate height):
      - element size ON the hole edge: <= 3.0 mm, and smaller where the edge
        turns quickly: at most 10 degrees of turning per element, never
        below 0.25 mm. So 3 mm fillets get ~0.5 mm elements and the 1.4 mm
        tip radius of the 8:1 ellipse gets 0.25 mm. Sizes change by at
        most 20% per element along the edge.
      - 18 element layers from the hole to the box. The first layer is half
        as thick as the local edge element is long, and the layers grow
        geometrically outward.
      - ellipses: the grid lines leave the hole along its normal.
        rectangles and polygons: the line directions are averaged over
        +-20 mm of hole edge (NORMAL_SMOOTH), so the lines fan out evenly
        like around a circle instead of in parallel bundles from each
        flat side with gaps at each fillet.
    Checked before delivery: every element valid in all 49 meshes, exact
    mirror symmetry, smallest angle 36 deg for rectangles and polygons and
    19.6 deg anywhere (beside the 8:1 ellipse tips). Peak von Mises at
    10 MPa compared with the finer version-1 mesh: circle -0.6%, triangle
    -0.6%, 8:1 ellipse tip -1.6%, narrow slot R_W8 -1.8%, square fillets
    -3.0% (set TURN_PER_ELEM = 5 and H_HOLE_MIN = 0.1 to get version 1's
    accuracy back).
    Two structured blocks from the box to the plate ends (x = +-750 mm):
      - element length grows by 12% per element up to at most 25 mm
        (the paper's global seed was 22.5 mm).
    Element type CPS4: 4-node bilinear plane-stress quad, full (2 x 2) integration.
    Typical plate: 2,000 - 6,100 nodes (version 1: 3,200 - 10,500).

SYMMETRY
    every hole here is symmetric about the vertical axis -> left/right mirror
    ellipses, rectangles, polygons with an even number of sides are also
    symmetric about the horizontal axis -> top/bottom mirror as well
    (polygons with 3, 5, 7, 9 sides have a vertex at the top and a flat side
    at the bottom, so they are not, and their mesh is not forced to be)
    The script checks this for every mesh and refuses to continue if a
    mirrored node or element is missing.

CPUS
    CPUS_PER_JOB = 10 uses 10 cores for each job (cpus=10 mp_mode=threads).
    For small 2D models most of the time is Abaqus start-up, so if your
    licence has enough tokens, PARALLEL_JOBS = 5 with CPUS_PER_JOB = 2 finishes
    the 1127 jobs several times faster. PARALLEL_JOBS x CPUS_PER_JOB should not
    exceed your 10 cores.

FILES WRITTEN (in the current folder)
    plate_si_<name>_mesh.inp      nodes, elements, sets - once per geometry
    plate_si_<name>__<load>.inp   the job: *Include of the mesh + the load step
    plate_si_<name>__<load>.odb   results (what extract_odb.py reads)
    si_manifest.csv               train / val / test labels + mesh size and quality

RESUME
    Jobs whose .odb exists and whose .sta says COMPLETED are skipped, so a
    stopped run picks up where it left off - just start it again.

AFTERWARDS (from the repo root)
    abaqus python abaqus\\extract_odb.py data\\odb_si_cps4\\*.odb --out data\\raw_si_cps4_solved
    python scripts\\organize_si_cps4.py
"""

from __future__ import print_function

import math
import os
import subprocess
import sys
import time

# Inside Abaqus/CAE (File > Run Script, or abaqus cae noGUI=...) the names sum,
# abs, min, max, pow are replaced by Abaqus' own field-output functions, which
# reject ordinary Python numbers and generators ("arg1; found 'generator',
# expecting a recognized type"). Take the real Python ones back.
try:
    import builtins as _py               # Python 3 (Abaqus 2024 and later)
except ImportError:
    import __builtin__ as _py            # Python 2 (Abaqus 2023 and earlier)
sum, abs, min, max, pow, round = _py.sum, _py.abs, _py.min, _py.max, _py.pow, _py.round

# ------------------------------------------------------------------ settings

MESH_ONLY = False               # True = build + check the meshes and write the input files, solve nothing
                                # (same as --mesh-only on the command line; use this from the CAE GUI)

WIDTH, HEIGHT = 1500.0, 250.0   # mm
E, NU = 200000.0, 0.3           # MPa (= 200 GPa steel)
THICKNESS = 25.0                # mm (does not change the stress under a traction load)
TRAIN_LOADS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
               65, 70, 75, 80, 85, 90, 95, 100, 105, 110]   # MPa
TEST_LOADS = [8, 60, 130]                                    # MPa
REF_LOAD = 10.0
SOLVE = 'all'                   # 'all' = every geometry x 23 loads, 'ref' = REF_LOAD only

# mesh
H_HOLE_MAX = 3.0                # mm, largest element along the hole edge             (was 2.0)
TURN_PER_ELEM = 10.0            # degrees of edge turning per element (curvature)      (was 5)
H_HOLE_MIN = 0.25               # mm, smallest element along the hole edge            (was 0.1)
GRADING = 0.2                   # element size may change by at most 20% per element along the edge
N_RADIAL = 18                   # element layers from the hole to the box             (was 24)
FIRST_LAYER = 0.5               # first layer thickness / element length along the edge
NORMAL_SMOOTH = 20.0            # mm, rectangles + polygons only: the grid lines leave the hole in a
                                # direction averaged over +-20 mm of edge, so they fan out smoothly
                                # like around a circle instead of in bundles from each flat side
                                # (0 = off, the old behaviour)
BOX = HEIGHT / 2.0              # half-size of the O-grid box (125 mm = full plate height)
BOX_NORMAL_WEIGHT = 0.5         # where the grid lines meet the box: 1 = straight out along the hole normal, 0 = evenly spaced
LAYER_SMOOTHING = 200           # passes that line up the layers of neighbouring grid lines
H_FAR_MAX = 25.0                # mm, largest element towards the plate ends
X_GROWTH = 1.12                 # element growth per element from the box to the plate ends
X_BLEND = 375.0                 # mm, distance over which the horizontal grid lines become evenly spaced
FILLET = 3.0                    # mm, rectangle and polygon corners (as in the CPS3 dataset)

# Abaqus
ABAQUS_CMD = 'abaqus'           # the command that runs Abaqus on your machine
CPUS_PER_JOB = 10
PARALLEL_JOBS = 1               # jobs running at the same time (see CPUS above)

R0 = 32.0                       # reference circle radius (64 mm diameter)
S0 = 64.0                       # reference square side
AR = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
SIDE = [64, 56, 48, 40, 32, 24, 20, 16, 12, 8]
TEST_INTERP_AR, TEST_EXTRAP_AR = (1.5, 3, 5), (8,)
TEST_INTERP_SIDE, TEST_EXTRAP_SIDE = (48, 24, 16), (8,)
VAL_AR_H, VAL_AR_V = (1.75,), (3.5,)          # validation ellipses: a/b (horizontal), b/a (vertical)
VAL_SIDE_W, VAL_SIDE_H = (28,), (36,)         # validation rectangles: the side that shrinks, mm

ONLY = None                     # e.g. ['E_H1'] to test one geometry first


# ------------------------------------------------------------- the geometries
# identical to generate_si_dataset.py

def tag(x):
    return ('%g' % x).replace('.', 'p')


def split_of(v, interp, extrap):
    if v in extrap:
        return 'test_extrap'
    if v in interp:
        return 'test_interp'
    return 'train'


def geometries():
    g = []
    for r in AR:
        g.append(dict(name='E_H' + tag(r), family='horizontal_ellipse', kind='ellipse',
                      a=R0 * math.sqrt(r), b=R0 / math.sqrt(r), param='a/b=%g' % r,
                      split=split_of(r, TEST_INTERP_AR, TEST_EXTRAP_AR)))
    for r in AR[1:]:
        g.append(dict(name='E_V' + tag(r), family='vertical_ellipse', kind='ellipse',
                      a=R0 / math.sqrt(r), b=R0 * math.sqrt(r), param='b/a=%g' % r,
                      split=split_of(r, TEST_INTERP_AR, TEST_EXTRAP_AR)))
    for s in SIDE:
        g.append(dict(name='R_W' + tag(s), family='rect_width_shrinks', kind='rect',
                      a=float(s), b=S0, param='a=%g b=%g' % (s, S0),
                      split=split_of(s, TEST_INTERP_SIDE, TEST_EXTRAP_SIDE)))
    for s in SIDE[1:]:
        g.append(dict(name='R_H' + tag(s), family='rect_height_shrinks', kind='rect',
                      a=S0, b=float(s), param='a=%g b=%g' % (S0, s),
                      split=split_of(s, TEST_INTERP_SIDE, TEST_EXTRAP_SIDE)))
    for n in (10, 9, 8, 7, 6, 5, 3):
        g.append(dict(name='P%d' % n, family='polygon', kind='poly', n=n,
                      a=2 * R0, b=2 * R0, param='n=%d' % n,
                      split='test_extrap' if n == 3 else 'test_interp' if n == 7 else 'train'))
    # validation shapes - appended last, so the 45 above keep their order and names
    for r in VAL_AR_H:
        g.append(dict(name='E_H' + tag(r), family='horizontal_ellipse', kind='ellipse',
                      a=R0 * math.sqrt(r), b=R0 / math.sqrt(r), param='a/b=%g' % r, split='val'))
    for r in VAL_AR_V:
        g.append(dict(name='E_V' + tag(r), family='vertical_ellipse', kind='ellipse',
                      a=R0 / math.sqrt(r), b=R0 * math.sqrt(r), param='b/a=%g' % r, split='val'))
    for s in VAL_SIDE_W:
        g.append(dict(name='R_W' + tag(s), family='rect_width_shrinks', kind='rect',
                      a=float(s), b=S0, param='a=%g b=%g' % (s, S0), split='val'))
    for s in VAL_SIDE_H:
        g.append(dict(name='R_H' + tag(s), family='rect_height_shrinks', kind='rect',
                      a=S0, b=float(s), param='a=%g b=%g' % (S0, s), split='val'))
    return g


def symmetric_top_bottom(geo):
    return geo['kind'] in ('ellipse', 'rect') or geo['n'] % 2 == 0


# ------------------------------------------------------- the hole boundary
# A dense polyline of the RIGHT half of the hole edge (x >= 0), from the
# bottom crossing of the y-axis, anticlockwise round the hole, to the top
# crossing. Each point carries the edge curvature there.

def _arc(cx, cy, r, t0, t1, n):
    """points of a circular arc, angles in radians, anticlockwise, curvature 1/r"""
    return [(cx + r * math.cos(t0 + (t1 - t0) * k / float(n)),
             cy + r * math.sin(t0 + (t1 - t0) * k / float(n)), 1.0 / r) for k in range(n + 1)]


def _line(p, q, n):
    return [(p[0] + (q[0] - p[0]) * k / float(n), p[1] + (q[1] - p[1]) * k / float(n), 0.0)
            for k in range(n + 1)]


def _corner_arcs(geo):
    """(centre_x, centre_y, start_rad, end_rad) of the fillets, anticlockwise from the bottom right"""
    f = FILLET
    if geo['kind'] == 'rect':
        hw, hh = geo['a'] / 2.0 - f, geo['b'] / 2.0 - f
        return [(hw, -hh, -math.pi / 2, 0.0), (hw, hh, 0.0, math.pi / 2),
                (-hw, hh, math.pi / 2, math.pi), (-hw, -hh, math.pi, 1.5 * math.pi)]
    n = geo['n']
    a0 = -90.0 + 180.0 / n                       # flat side at the bottom
    rc = R0 - f / math.cos(math.pi / n)          # keeps circumradius = R0
    out = []
    for k in range(n):
        th = a0 + 360.0 * k / n
        out.append((rc * math.cos(math.radians(th)), rc * math.sin(math.radians(th)),
                    math.radians(th - 180.0 / n), math.radians(th + 180.0 / n)))
    return out


def hole_right_half(geo, dense=40000):
    """[(x, y, curvature)] from (0, y_bottom) to (0, y_top), x >= 0."""
    if geo['kind'] == 'ellipse':
        a, b = geo['a'], geo['b']
        pts = []
        for k in range(dense + 1):
            t = -math.pi / 2 + math.pi * k / float(dense)
            st, ct = math.sin(t), math.cos(t)
            kap = a * b / (a * a * st * st + b * b * ct * ct) ** 1.5
            pts.append((a * ct, b * st, kap))
        pts[0] = (0.0, -b, pts[0][2])
        pts[-1] = (0.0, b, pts[-1][2])
        return pts
    # rectangle / polygon: arcs + straight sides, the full loop from the bottom crossing
    f = FILLET
    arcs = _corner_arcs(geo)
    if geo['kind'] == 'poly':
        # rotate so the list starts with the first corner right of the bottom side
        arcs = sorted(arcs, key=lambda c: (math.atan2(c[1], c[0]) + 2.5 * math.pi) % (2 * math.pi))
    loop = []
    first = arcs[0]
    p_start = (first[0] + f * math.cos(first[2]), first[1] + f * math.sin(first[2]))
    y_bottom = p_start[1]
    loop += _line((0.0, y_bottom), p_start, 2000)
    for k, (cx, cy, t0, t1) in enumerate(arcs):
        loop += _arc(cx, cy, f, t0, t1, 2000)[1:]
        nxt = arcs[(k + 1) % len(arcs)]
        p0 = (cx + f * math.cos(t1), cy + f * math.sin(t1))
        p1 = (nxt[0] + f * math.cos(nxt[2]), nxt[1] + f * math.sin(nxt[2]))
        loop += _line(p0, p1, 2000)[1:]
    # keep the part up to the top crossing of x = 0
    out = [loop[0]]
    for k in range(1, len(loop)):
        x0, y0, c0 = loop[k - 1]
        x1, y1, c1 = loop[k]
        if y1 > 0 and x0 > 0 and x1 <= 0:
            t = x0 / (x0 - x1)
            out.append((0.0, y0 + t * (y1 - y0), c0))
            return out
        out.append(loop[k])
    raise RuntimeError('no top crossing found for %s' % geo['name'])


def upper_quarter(half):
    """the part of the right half with y >= 0, from (x_max, 0) to (0, y_top)"""
    for k in range(1, len(half)):
        x0, y0, c0 = half[k - 1]
        x1, y1, c1 = half[k]
        if y0 < 0 <= y1:
            t = -y0 / (y1 - y0) if y1 != y0 else 0.0
            start = (x0 + t * (x1 - x0), 0.0, c0 + t * (c1 - c0))
            rest = [p for p in half[k:] if p[1] > 1e-12]
            return [start] + rest
    raise RuntimeError('no crossing of y = 0')


# ------------------------------------------------------------ edge nodes

def distribute(curve, fixed_normal_start=None, fixed_normal_end=None):
    """Nodes along a dense curve. Returns lists: points, outward normals,
    local element sizes. Size = TURN_PER_ELEM / curvature, clipped to
    [H_HOLE_MIN, H_HOLE_MAX], then graded so it changes <= GRADING per element."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    s = [0.0]
    for k in range(1, len(curve)):
        s.append(s[-1] + math.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1]))
    turn = math.radians(TURN_PER_ELEM)
    h = []
    for p in curve:
        kap = abs(p[2])
        h.append(H_HOLE_MAX if kap < 1e-12 else min(H_HOLE_MAX, max(H_HOLE_MIN, turn / kap)))
    for k in range(1, len(h)):                       # grading, both directions
        h[k] = min(h[k], h[k - 1] + GRADING * (s[k] - s[k - 1]))
    for k in range(len(h) - 2, -1, -1):
        h[k] = min(h[k], h[k + 1] + GRADING * (s[k + 1] - s[k]))
    # element count N(s) = integral ds / h
    N = [0.0]
    for k in range(1, len(s)):
        N.append(N[-1] + (s[k] - s[k - 1]) * 0.5 * (1.0 / h[k] + 1.0 / h[k - 1]))
    n_el = max(2, int(round(N[-1])))
    pts, sizes, idx = [], [], 0
    for i in range(n_el + 1):
        target = N[-1] * i / float(n_el)
        while idx < len(N) - 2 and N[idx + 1] < target:
            idx += 1
        span = N[idx + 1] - N[idx]
        t = 0.0 if span <= 0 else min(1.0, max(0.0, (target - N[idx]) / span))
        pts.append((xs[idx] + t * (xs[idx + 1] - xs[idx]), ys[idx] + t * (ys[idx + 1] - ys[idx])))
        sizes.append(h[idx] + t * (h[idx + 1] - h[idx]))
    pts[0] = (xs[0], ys[0])
    pts[-1] = (xs[-1], ys[-1])
    # outward normal (into the material) = right of the anticlockwise tangent
    normals = []
    for i in range(len(pts)):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        L = math.hypot(tx, ty)
        normals.append((ty / L, -tx / L))
    if fixed_normal_start:
        normals[0] = fixed_normal_start
    if fixed_normal_end:
        normals[-1] = fixed_normal_end
    return pts, normals, sizes


def smooth_normals(pts, normals, length, kind):
    """Average the hole normals over +-length mm of edge (triangle weights).
    The two ends lie on symmetry lines, so the edge is mirrored there first
    and the end normals keep their exact symmetric direction."""
    if length <= 0:
        return list(normals)
    s = [0.0]
    for k in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]))
    S = s[-1]
    if kind == 'quarter':
        first = lambda n: (n[0], -n[1])        # start on the x-axis: mirror y
    else:
        first = lambda n: (-n[0], n[1])        # start on the y-axis: mirror x
    last = lambda n: (-n[0], n[1])             # end always on the y-axis
    ss, nn = list(s), list(normals)
    for k in range(1, len(pts)):
        ss.append(-s[k]); nn.append(first(normals[k]))
        ss.append(2 * S - s[-1 - k]); nn.append(last(normals[-1 - k]))
    out = []
    for i in range(len(pts)):
        if i == 0 or i == len(pts) - 1:
            out.append(normals[i])
            continue
        sx = sy = 0.0
        for sk, nk in zip(ss, nn):
            w = 1.0 - abs(sk - s[i]) / length
            if w > 0:
                sx += w * nk[0]
                sy += w * nk[1]
        L = math.hypot(sx, sy)
        out.append((sx / L, sy / L))
    return out


# ------------------------------------------------------------ the O-grid

def box_path(kind):
    """corner points of the outer edge of the O-grid region, in order"""
    B = BOX
    if kind == 'quarter':
        return [(B, 0.0), (B, B), (0.0, B)]
    return [(0.0, -B), (B, -B), (B, B), (0.0, B)]


def _path_param(path, q):
    """arc-length position of point q on the polyline path"""
    u = 0.0
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        ab = ((q[0] - a[0]) * (b[0] - a[0]) + (q[1] - a[1]) * (b[1] - a[1])) / (L * L)
        cr = abs((q[0] - a[0]) * (b[1] - a[1]) - (q[1] - a[1]) * (b[0] - a[0])) / L
        if cr < 1e-6 * BOX and -1e-9 <= ab <= 1 + 1e-9:
            return u + ab * L
        u += L
    raise RuntimeError('point not on the box path')


def _path_point(path, u):
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if u <= L + 1e-12 or k == len(path) - 2:
            t = min(1.0, max(0.0, u / L))
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        u -= L


def _ray_hit(p, n):
    """where the ray p + t n (t > 0) leaves the box"""
    B, ts = BOX, []
    if n[0] > 1e-12:
        ts.append((B - p[0]) / n[0])
    if n[1] > 1e-12:
        ts.append((B - p[1]) / n[1])
    if n[1] < -1e-12:
        ts.append((-B - p[1]) / n[1])
    t = min(ts)
    return (p[0] + t * n[0], p[1] + t * n[1])


def ogrid(hole_pts, normals, sizes, kind):
    """Structured grid between the hole nodes and the box. Returns a
    (n_t + 1) x (N_RADIAL + 1) list of points; index [i][0] is on the hole,
    [i][N_RADIAL] on the box."""
    path = box_path(kind)
    lens = [math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1]) for k in range(len(path) - 1)]
    corners_u = []
    acc = 0.0
    for L in lens[:-1]:
        acc += L
        corners_u.append(acc)
    total = sum(lens)
    n = len(hole_pts) - 1
    u_n = [_path_param(path, _ray_hit(p, nv)) for p, nv in zip(hole_pts, normals)]
    u_n[0], u_n[-1] = 0.0, total
    for k in range(1, len(u_n)):                       # monotone, just in case
        u_n[k] = max(u_n[k], u_n[k - 1])
    # each box corner gets the hole node whose normal points closest to it
    fixed = [0]
    for uc in corners_u:
        k = min(range(1, n), key=lambda i: abs(u_n[i] - uc))
        k = max(k, fixed[-1] + 1)
        fixed.append(k)
    fixed.append(n)
    targets = [0.0] + corners_u + [total]
    u = [0.0] * (n + 1)
    for seg in range(len(fixed) - 1):
        i0, i1 = fixed[seg], fixed[seg + 1]
        a0, a1 = u_n[i0], u_n[i1]
        b0, b1 = targets[seg], targets[seg + 1]
        for i in range(i0, i1 + 1):
            un = b0 + (b1 - b0) * ((u_n[i] - a0) / (a1 - a0) if a1 > a0 else (i - i0) / float(i1 - i0))
            uu = b0 + (b1 - b0) * (i - i0) / float(i1 - i0)
            u[i] = BOX_NORMAL_WEIGHT * un + (1.0 - BOX_NORMAL_WEIGHT) * uu
    box_pts = [_path_point(path, ui) for ui in u]
    for k, uc in zip(fixed[1:-1], corners_u):
        box_pts[k] = _path_point(path, uc)              # exactly on the corner
    # layer positions along every grid line: geometric, first layer = local edge size
    lengths = [math.hypot(box_pts[i][0] - hole_pts[i][0], box_pts[i][1] - hole_pts[i][1]) for i in range(n + 1)]
    frac = [[d / lengths[i] for d in radial_distances(FIRST_LAYER * sizes[i], lengths[i], N_RADIAL)] for i in range(n + 1)]
    # outer layers: line up neighbouring lines (smooth the layer fractions sideways),
    # inner layers keep their own spacing so the elements on the hole stay square
    smooth = [row[:] for row in frac]
    for _ in range(LAYER_SMOOTHING):
        new = [row[:] for row in smooth]
        for i in range(1, n):
            for j in range(1, N_RADIAL):
                new[i][j] = 0.25 * smooth[i - 1][j] + 0.5 * smooth[i][j] + 0.25 * smooth[i + 1][j]
        smooth = new
    for i in range(n + 1):
        for j in range(1, N_RADIAL):
            lam = (j / float(N_RADIAL)) ** 0.7
            frac[i][j] = (1 - lam) * frac[i][j] + lam * smooth[i][j]
        for j in range(1, N_RADIAL + 1):
            frac[i][j] = max(frac[i][j], frac[i][j - 1] + 1e-9)
    grid = []
    for i in range(n + 1):
        H, Bx, nv, L = hole_pts[i], box_pts[i], normals[i], lengths[i]
        line = []
        D = ((Bx[0] - H[0]) / L, (Bx[1] - H[1]) / L)
        for s in frac[i]:
            # leaves the hole along its normal and turns smoothly towards its box point
            vx = (1 - s) * nv[0] + s * D[0]
            vy = (1 - s) * nv[1] + s * D[1]
            line.append((H[0] + s * L * vx, H[1] + s * L * vy))
        line[0], line[-1] = H, Bx
        grid.append(line)
    return grid


def radial_distances(h0, L, n):
    """n geometric steps starting with h0 that add up to L"""
    h0 = min(h0, L / n)
    lo, hi = 1.0, 2.0
    while h0 * (hi ** n - 1) / (hi - 1) < L:
        hi *= 1.5
    for _ in range(80):
        q = 0.5 * (lo + hi)
        tot = h0 * n if abs(q - 1) < 1e-12 else h0 * (q ** n - 1) / (q - 1)
        if tot < L:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    d, step = [0.0], h0
    for _ in range(n):
        d.append(d[-1] + step)
        step *= q
    return [x * L / d[-1] for x in d]


# ------------------------------------------------------------ outer block

def x_stations(h_first):
    """x positions from BOX to WIDTH/2: growing by X_GROWTH up to H_FAR_MAX"""
    xend = WIDTH / 2.0
    sizes, x, h = [], BOX, h_first
    while x < xend - 1e-9:
        sizes.append(h)
        x += h
        h = min(h * X_GROWTH, H_FAR_MAX)
    scale = (xend - BOX) / sum(sizes)
    xs = [BOX]
    for hs in sizes:
        xs.append(xs[-1] + hs * scale)
    xs[-1] = xend
    return xs


def outer_block(y_box, h_first):
    """structured grid from x = BOX to the plate end; y_box are the node
    heights on x = BOX (bottom to top). Lines straighten towards even spacing."""
    xs = x_stations(h_first)
    y0, y1 = y_box[0], y_box[-1]
    m = len(y_box) - 1
    even = [y0 + (y1 - y0) * j / float(m) for j in range(m + 1)]
    grid = []
    for x in xs:
        t = min(1.0, (x - BOX) / (X_BLEND - BOX))
        w = t * t * (3 - 2 * t)
        grid.append([(x, (1 - w) * yb + w * ye) for yb, ye in zip(y_box, even)])
    return grid                                            # grid[i][j]: i along x, j along y


# ------------------------------------------------------------ assembly

def quads_from_grid(grid):
    out = []
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            out.append([grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]])
    return out


def mirror_x(q):
    return [(-p[0], p[1]) for p in q]


def mirror_y(q):
    return [(p[0], -p[1]) for p in q]


def signed_area(q):
    a = 0.0
    for k in range(4):
        x0, y0 = q[k]
        x1, y1 = q[(k + 1) % 4]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def build_mesh(geo):
    """returns nodes [(x, y)], elements [[n1, n2, n3, n4]] (0-based, anticlockwise)"""
    half = hole_right_half(geo)
    quads = []
    if symmetric_top_bottom(geo):
        quarter = upper_quarter(half)
        pts, nrm, h = distribute(quarter, fixed_normal_start=(1.0, 0.0), fixed_normal_end=(0.0, 1.0))
        if geo['kind'] in ('rect', 'poly'):
            nrm = smooth_normals(pts, nrm, NORMAL_SMOOTH, 'quarter')
        g = ogrid(pts, nrm, h, 'quarter')
        q_upper = quads_from_grid(g)
        # box right side, y from 0 up to BOX: the lines whose box end has x = BOX
        right = [line[-1] for line in g if abs(line[-1][0] - BOX) < 1e-9]
        right = sorted(right, key=lambda p: p[1])
        y_box = [-p[1] for p in reversed(right)][:-1] + [p[1] for p in right]
        h_first = g[0][-1][0] - g[0][-2][0]
        blk = outer_block(y_box, h_first)
        q_block = quads_from_grid(blk)
        right_half = q_upper + [mirror_y(q) for q in q_upper] + q_block
    else:
        pts, nrm, h = distribute(half, fixed_normal_start=(0.0, -1.0), fixed_normal_end=(0.0, 1.0))
        if geo['kind'] in ('rect', 'poly'):
            nrm = smooth_normals(pts, nrm, NORMAL_SMOOTH, 'half')
        g = ogrid(pts, nrm, h, 'half')
        right = sorted([line[-1] for line in g if abs(line[-1][0] - BOX) < 1e-9], key=lambda p: p[1])
        y_box = [p[1] for p in right]
        mid = min(g, key=lambda line: abs(line[-1][1]) if abs(line[-1][0] - BOX) < 1e-9 else 1e9)
        h_first = mid[-1][0] - mid[-2][0]
        right_half = quads_from_grid(g) + quads_from_grid(outer_block(y_box, h_first))
    all_q = right_half + [mirror_x(q) for q in right_half]
    # merge coincident nodes, orient every element anticlockwise
    key_of, nodes, elems = {}, [], []
    for q in all_q:
        if signed_area(q) < 0:
            q = [q[0], q[3], q[2], q[1]]
        ids = []
        for p in q:
            k = (round(p[0], 6), round(p[1], 6))
            if k not in key_of:
                key_of[k] = len(nodes)
                nodes.append((p[0], p[1]))
            ids.append(key_of[k])
        if len(set(ids)) != 4:
            raise RuntimeError('degenerate element in %s' % geo['name'])
        elems.append(ids)
    return nodes, elems


# ------------------------------------------------------------ checks

def quality(nodes, elems):
    """min corner angle (deg), max aspect ratio, min corner Jacobian (must be > 0)"""
    amin, asp, jmin = 180.0, 1.0, 1e30
    for e in elems:
        p = [nodes[i] for i in e]
        L = [math.hypot(p[(k + 1) % 4][0] - p[k][0], p[(k + 1) % 4][1] - p[k][1]) for k in range(4)]
        asp = max(asp, max(L) / min(L))
        for k in range(4):
            a, b, c = p[k - 1], p[k], p[(k + 1) % 4]
            u = (c[0] - b[0], c[1] - b[1])
            v = (a[0] - b[0], a[1] - b[1])
            cross = u[0] * v[1] - u[1] * v[0]
            jmin = min(jmin, cross)
            cosang = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
            amin = min(amin, math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return amin, asp, jmin


def check_symmetry(nodes, elems, top_bottom):
    """every node and element must have its mirror image"""
    key = dict(((round(x, 5), round(y, 5)), i) for i, (x, y) in enumerate(nodes))
    eset = set(tuple(sorted(e)) for e in elems)
    mirrors = [lambda x, y: (-x, y)] + ([lambda x, y: (x, -y)] if top_bottom else [])
    for f in mirrors:
        img = []
        for (x, y) in nodes:
            mx, my = f(x, y)
            k = (round(mx, 5) + 0.0, round(my, 5) + 0.0)
            if k not in key:
                return False
            img.append(key[k])
        for e in elems:
            if tuple(sorted(img[i] for i in e)) not in eset:
                return False
    return True


# ------------------------------------------------------------ input files

def mesh_file(geo):
    return 'plate_si_%s_mesh.inp' % geo['name']


def job_name(geo, load):
    return 'plate_si_%s__%d' % (geo['name'], int(load))


def write_mesh_inp(geo, nodes, elems):
    xl, xr = -WIDTH / 2.0, WIDTH / 2.0
    left = [i + 1 for i, (x, y) in enumerate(nodes) if abs(x - xl) < 1e-6]
    faces = {1: [], 2: [], 3: [], 4: []}              # CPS4 face k joins local nodes k and k+1
    for ei, e in enumerate(elems):
        for k in range(4):
            a, b = nodes[e[k]], nodes[e[(k + 1) % 4]]
            if abs(a[0] - xr) < 1e-6 and abs(b[0] - xr) < 1e-6:
                faces[k + 1].append(ei + 1)
    lines = ['** %s - structured symmetric CPS4 mesh, mm' % geo['name'], '*Node']
    lines += ['%d, %.9f, %.9f' % (i + 1, x, y) for i, (x, y) in enumerate(nodes)]
    lines.append('*Element, type=CPS4, elset=EALL')
    lines += ['%d, %d, %d, %d, %d' % (i + 1, e[0] + 1, e[1] + 1, e[2] + 1, e[3] + 1) for i, e in enumerate(elems)]

    def chunks(ids):
        return [', '.join(str(v) for v in ids[k:k + 16]) for k in range(0, len(ids), 16)]

    lines.append('*Nset, nset=LEFT')
    lines += chunks(left)
    surf = []
    for k in (1, 2, 3, 4):
        if faces[k]:
            lines.append('*Elset, elset=LOAD_S%d' % k)
            lines += chunks(faces[k])
            surf.append('LOAD_S%d, S%d' % (k, k))
    lines.append('*Surface, type=ELEMENT, name=LOADSURF')
    lines += surf
    lines += ['*Solid Section, elset=EALL, material=STEEL', '%g,' % THICKNESS,
              '*Material, name=STEEL', '*Elastic', '%g, %g' % (E, NU)]
    with open(mesh_file(geo), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return len(left), sum(len(v) for v in faces.values())


def write_job_inp(geo, load):
    txt = """*Heading
plate with hole %s, %d MPa, CPS4, mm-N-MPa
*Preprint, echo=NO, model=NO, history=NO, contact=NO
*Include, input=%s
*Boundary
LEFT, 1, 2
*Step, name=Step-1, nlgeom=NO
*Static
1., 1., 1e-05, 1.
** negative pressure = tension (pulls the right edge outward), MPa
*Dsload
LOADSURF, P, %g
*Output, field
*Node Output
U
*Element Output
S
*End Step
""" % (geo['name'], int(load), mesh_file(geo), -float(load))
    with open(job_name(geo, load) + '.inp', 'w') as fh:
        fh.write(txt)


# ------------------------------------------------------------ running jobs

JOB_EXTS = ('.odb', '.dat', '.msg', '.sta', '.prt', '.com', '.log', '.sim',
            '.lck', '.res', '.mdl', '.stt', '.ipm', '.abq', '.pac', '.sel', '.023')


def already_done(job):
    odb, sta = job + '.odb', job + '.sta'
    if not (os.path.exists(odb) and os.path.exists(sta)):
        return False
    try:
        return 'COMPLETED' in open(sta).read()
    except Exception:
        return False


def clean_partial(job):
    for ext in JOB_EXTS:
        if os.path.exists(job + ext):
            try:
                os.remove(job + ext)
            except Exception:
                pass


def start_job(job):
    clean_partial(job)
    cmd = '%s job=%s input=%s.inp cpus=%d mp_mode=threads interactive ask_delete=OFF' % (
        ABAQUS_CMD, job, job, CPUS_PER_JOB)
    log = open(job + '.run.txt', 'w')
    return subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT), log


def run_jobs(jobs, t0, n_total, n_before):
    """run the job names in `jobs`, PARALLEL_JOBS at a time"""
    queue, running, finished, failed = list(jobs), [], 0, []
    while queue or running:
        while queue and len(running) < PARALLEL_JOBS:
            job = queue.pop(0)
            p, log = start_job(job)
            running.append((job, p, log))
        time.sleep(1.0)
        still = []
        for job, p, log in running:
            if p.poll() is None:
                still.append((job, p, log))
                continue
            log.close()
            finished += 1
            ok = already_done(job)
            if not ok:
                failed.append(job)
            done = n_before + finished
            el = time.time() - t0
            print('    %4d/%-4d %-32s %s   elapsed %s   eta %s' % (
                done, n_total, job, 'ok' if ok else 'FAILED (see %s.run.txt / .msg)' % job,
                fmt_hms(el), fmt_hms(el / max(finished, 1) * (n_total - done))))
            sys.stdout.flush()
        running = still
    return failed


def fmt_hms(s):
    s = int(s)
    return '%dh%02dm%02ds' % (s // 3600, (s % 3600) // 60, s % 60)


def loads_to_solve():
    if SOLVE == 'ref':
        return [int(REF_LOAD)]
    return sorted(set(int(L) for L in TRAIN_LOADS + TEST_LOADS))


def write_manifest(rows, path='si_manifest.csv'):
    cols = ['geometry', 'name', 'family', 'split', 'kind', 'a_mm', 'b_mm', 'n_sides', 'param',
            'element', 'n_nodes', 'n_elements', 'min_angle_deg', 'max_aspect', 'symmetric',
            'loads_done', 'status']
    with open(path, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(str(r.get(c, '')) for c in cols) + '\n')


# ------------------------------------------------------------ main

def main(argv):
    mesh_only = MESH_ONLY or '--mesh-only' in argv
    geos = geometries()
    if ONLY:
        geos = [g for g in geos if g['name'] in ONLY]
    loads = loads_to_solve()
    total = len(geos) * len(loads)
    print('=' * 78)
    print('SI DATASET, structured symmetric CPS4 mesh: %d geometries x %d loads = %d jobs'
          % (len(geos), len(loads), total))
    print('plate %g x %g mm, E = %g MPa, nu = %g, thickness %g mm' % (WIDTH, HEIGHT, E, NU, THICKNESS))
    print('hole edge <= %.2f mm (>= %.2f mm, %g deg per element), %d layers to the %g mm box, far field <= %g mm'
          % (H_HOLE_MAX, H_HOLE_MIN, TURN_PER_ELEM, N_RADIAL, 2 * BOX, H_FAR_MAX))
    print('rectangles / polygons: grid-line directions smoothed over +-%g mm of hole edge' % NORMAL_SMOOTH)
    if not mesh_only:
        print('Abaqus: %d core(s) per job, %d job(s) at a time, command "%s"'
              % (CPUS_PER_JOB, PARALLEL_JOBS, ABAQUS_CMD))
    print('output -> %s' % os.getcwd())
    print('=' * 78)
    print('%-10s %-12s %7s %7s %10s %10s %9s' % ('geometry', 'split', 'nodes', 'elems', 'min angle', 'max aspect', 'symmetric'))
    rows, plan, bad = [], [], []
    for g in geos:
        row = dict(geometry='plate_si_' + g['name'], name=g['name'], family=g['family'],
                   split=g['split'], kind=g['kind'], a_mm='%.6f' % g['a'], b_mm='%.6f' % g['b'],
                   n_sides=g.get('n', ''), param=g['param'], element='CPS4')
        try:
            nodes, elems = build_mesh(g)
            amin, asp, jmin = quality(nodes, elems)
            sym = check_symmetry(nodes, elems, symmetric_top_bottom(g))
            n_left, n_faces = write_mesh_inp(g, nodes, elems)
        except Exception as exc:
            row.update(status='MESH FAILED: %s' % str(exc).replace(',', ';'))
            rows.append(row)
            bad.append(g['name'])
            print('%-10s MESH FAILED: %s' % (g['name'], exc))
            continue
        ok_mesh = jmin > 0 and sym and n_left > 1 and n_faces > 0
        row.update(n_nodes=len(nodes), n_elements=len(elems), min_angle_deg='%.1f' % amin,
                   max_aspect='%.1f' % asp, symmetric='both axes' if symmetric_top_bottom(g) else 'left-right')
        print('%-10s %-12s %7d %7d %9.1f%s %10.1f %9s' % (g['name'], g['split'], len(nodes), len(elems),
              amin, '' if jmin > 0 else '!', asp, 'yes' if sym else 'NO'))
        if not ok_mesh:
            row.update(status='MESH CHECK FAILED')
            bad.append(g['name'])
            rows.append(row)
            continue
        pending = [job_name(g, L) for L in loads if not already_done(job_name(g, L))]
        for L in loads:
            if job_name(g, L) in pending:
                write_job_inp(g, L)
        row['_pending'] = pending
        plan.append((g, row))
        rows.append(row)
    write_manifest(rows)
    if bad:
        print('\nMESH PROBLEMS: %s - nothing will be solved until these are fixed.' % ', '.join(bad))
        return 1
    print('\nall %d meshes passed: every element valid, mirror symmetry exact.' % len(plan))
    if mesh_only:
        print('--mesh-only: input files written, nothing solved.')
        return 0
    t0, done_before, failed = time.time(), 0, []
    n_todo = sum(len(r['_pending']) for g, r in plan)
    print('%d job(s) to run (%d already complete)\n' % (n_todo, total - n_todo))
    for gi, (g, row) in enumerate(plan):
        if row['_pending']:
            print('[%2d/%d %s] %d job(s)' % (gi + 1, len(plan), g['name'], len(row['_pending'])))
            failed += run_jobs(row['_pending'], t0, n_todo, done_before)
            done_before += len(row['_pending'])
        n_ok = len([L for L in loads if already_done(job_name(g, L))])
        row.update(loads_done=n_ok, status='completed' if n_ok == len(loads) else 'INCOMPLETE')
        write_manifest(rows)
    print('\n' + '=' * 78)
    print('finished in %s, %d failed job(s)%s' % (fmt_hms(time.time() - t0), len(failed),
                                                ': ' + ', '.join(failed[:10]) if failed else ''))
    print('manifest: %s' % os.path.abspath('si_manifest.csv'))
    print('=' * 78)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
