"""
Extract mesh + nodal von Mises from Abaqus ODB files into .npz.
Works for CPS3 (triangles) AND CPS4 (quads).

RUN THIS WITH ABAQUS' OWN PYTHON, NOT YOUR VENV:

    abaqus python extract_odb.py data/odb_si_cps4/*.odb --out data/raw_si_cps4_solved

Filename convention:  <geometry>__<load>.odb   e.g.  plate_si_E_H1__10.odb
The load value is parsed from the filename. Override with --load <value>.

Writes one file per ODB to <out>/<geometry>__<load>.npz containing:
    coords    (N, 2) float32   node order = sorted node label
    tris      (T, 3) int32     0-based. For a CPS4 mesh every quad is split
                               into 2 triangles along its shorter diagonal,
                               mirror-consistently, so the existing MGN code
                               (build_edges, node types, anchors) runs unchanged.
    quads     (Q, 4) int32     0-based, the CPS4 elements themselves (empty for CPS3)
    mises     (N,)   float32   nodal-averaged from ELEMENT_NODAL
    load      scalar float32
    geometry  str

Deliberately imports nothing from `abaqus` so it runs under `abaqus python`
(no CAE licence token needed).
"""

from __future__ import print_function

import glob
import os
import sys

import numpy as np
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL


OUT_DIR = os.path.join("data", "raw")


def parse_name(odb_path):
    """'.../plate_si_E_H1__10.odb' -> ('plate_si_E_H1', 10.0)"""
    stem = os.path.splitext(os.path.basename(odb_path))[0]
    if "__" in stem:
        geom, load = stem.rsplit("__", 1)
        try:
            return geom, float(load)
        except ValueError:
            pass
    return stem, None


def split_quad(q, xy):
    """Two triangles from one anticlockwise quad. Shorter diagonal; if both are
    equal, the choice depends on the quadrant so mirrored quads split as mirror images."""
    p = xy[q]
    d13 = np.hypot(*(p[2] - p[0]))
    d24 = np.hypot(*(p[3] - p[1]))
    if abs(d13 - d24) > 1e-9 * max(d13, d24):
        use13 = d13 < d24
    else:
        c = p.mean(axis=0)
        v = p[2] - p[0]                       # does diagonal 1-3 run like '/' or '\' ?
        slash = v[0] * v[1] > 0
        use13 = slash == (c[0] * c[1] >= 0)
    if use13:
        return [q[0], q[1], q[2]], [q[0], q[2], q[3]]
    return [q[0], q[1], q[3]], [q[1], q[2], q[3]]


def extract(odb_path, load_override=None, out_dir=OUT_DIR):
    geometry, load = parse_name(odb_path)
    if load_override is not None:
        load = load_override
    if load is None:
        raise ValueError("Could not parse load from '%s'. Rename to <geom>__<load>.odb "
                         "or pass --load <value>." % odb_path)

    odb = openOdb(path=odb_path, readOnly=True)
    try:
        inst_names = list(odb.rootAssembly.instances.keys())
        if len(inst_names) != 1:
            raise ValueError("Expected 1 instance, found %d: %s" % (len(inst_names), inst_names))
        inst = odb.rootAssembly.instances[inst_names[0]]

        # ---- nodes: sorted by label, remapped to 0-based contiguous indices
        labels = np.array([n.label for n in inst.nodes], dtype=np.int64)
        coords3 = np.array([n.coordinates for n in inst.nodes], dtype=np.float64)
        order = np.argsort(labels)
        labels = labels[order]
        xy = coords3[order][:, :2]
        coords = xy.astype(np.float32)
        lab2idx = {}
        for i in range(len(labels)):
            lab2idx[int(labels[i])] = i
        n_nodes = len(labels)

        # ---- elements: 3-node triangles or 4-node quads
        tris, quads, bad = [], [], set()
        for e in inst.elements:
            conn = [lab2idx[int(c)] for c in e.connectivity]
            if len(conn) == 3:
                tris.append(conn)
            elif len(conn) == 4:
                quads.append(conn)
            else:
                bad.add(e.type)
        if bad:
            raise ValueError("Element types %s are not supported (only 3-node triangles "
                             "and 4-node quads, e.g. CPS3 / CPS4)." % sorted(bad))
        for q in quads:
            t1, t2 = split_quad(q, xy)
            tris.append(t1)
            tris.append(t2)
        tris = np.array(tris, dtype=np.int32)
        quads = np.array(quads, dtype=np.int32).reshape(-1, 4)

        # ---- stress: ELEMENT_NODAL gives one value per element-node pair
        # (integration points extrapolated to the nodes); average per node,
        # as Abaqus does at a 100% averaging threshold.
        step_name = list(odb.steps.keys())[-1]
        frame = odb.steps[step_name].frames[-1]
        if "S" not in frame.fieldOutputs:
            raise ValueError("No 'S' field output in %s." % odb_path)
        s = frame.fieldOutputs["S"].getSubset(position=ELEMENT_NODAL)
        acc = np.zeros(n_nodes, dtype=np.float64)
        cnt = np.zeros(n_nodes, dtype=np.int64)
        for v in s.values:
            i = lab2idx[int(v.nodeLabel)]
            acc[i] += v.mises
            cnt[i] += 1
        if (cnt == 0).any():
            raise ValueError("%d orphan nodes with no stress contribution." % int((cnt == 0).sum()))
        mises = (acc / cnt).astype(np.float32)

        # ---- write
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out = os.path.join(out_dir, "%s__%d.npz" % (geometry, int(load)))
        np.savez_compressed(out, coords=coords, tris=tris, quads=quads, mises=mises,
                            load=np.float32(load), geometry=np.array(geometry))
        print("%-34s N=%5d  quads=%5d  tris=%5d  max_mises=%10.2f  -> %s"
              % (os.path.basename(odb_path), n_nodes, len(quads), len(tris), mises.max(), out))
        return out
    finally:
        odb.close()


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    load_override = None
    if "--load" in argv:
        val = argv[argv.index("--load") + 1]
        load_override = float(val)
        args = [a for a in args if a != val]
    out_dir = OUT_DIR
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
        args = [a for a in args if a != out_dir]
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])
    if not paths:
        print(__doc__)
        return 1
    for p in paths:
        extract(p, load_override, out_dir=out_dir)
    print("\nDone: %d ODB(s)." % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
