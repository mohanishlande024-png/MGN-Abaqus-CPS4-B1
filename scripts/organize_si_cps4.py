r"""
Sort the solved CPS4 SI dataset into TRAIN / VALIDATION / TEST folders.

    python scripts\organize_si_cps4.py
    python scripts\organize_si_cps4.py --val E_H3,E_V3,R_W24,R_H24      (no new Abaqus runs: borrow 4 test shapes)

UNITS  mm and MPa - exactly what abaqus\generate_si_dataset_cps4.py solved with.

INPUT
    data\raw_si_cps4_solved\plate_si_<name>__<load>.npz    every geometry at every load
                                                           (abaqus\extract_odb.py, the CPS4 version)
    data\odb_si_cps4\si_manifest.csv                       train / val / test labels (from the generator)

OUTPUT  (files are COPIED exactly as extract_odb.py wrote them, nothing is scaled)
    data\si_cps4\train\           27 training geometries   x 20 training loads    (540 files)
    data\si_cps4\val\              4 validation geometries x 20 training loads    ( 80 files)
    data\si_cps4\unseen_loads\    27 training geometries   x  3 test loads        ( 81 files)
    data\si_cps4\unseen_interp\   13 interpolation shapes  x all 23 loads         (299 files)
    data\si_cps4\unseen_extrap\    5 extrapolation shapes  x all 23 loads         (115 files)
    data\si_cps4\si_manifest.csv  the manifest, with the split actually used

WHAT EACH FOLDER IS FOR
    train          the model learns from these (compiled into one .pt file on Ubuntu)
    val            scored every few epochs DURING training -> the validation curve of
                   R2 vs epoch, and the choice of the best epoch (mgn_best_val.pt).
                   New shapes inside each family's range, at the training loads.
    unseen_*       the three TESTS, used once, at the end (scripts/eval_own_unseen.py)

--val NAMES
    By default the validation shapes are the 4 geometries the generator labels 'val'
    (E_H1p75, E_V3p5, R_W28, R_H36). If you did not solve those, name other shapes
    here and they are moved from their test set into val instead, e.g.
    --val E_H3,E_V3,R_W24,R_H24 (then the interpolation test has 9 shapes, not 13).

CHECKS, before anything is written
    1. every geometry x load in the plan has a solved file (missing ones are listed)
    2. CPS4: the files hold the quads (extract_odb.py, CPS4 version) and the quads
       were split into 2 triangles each - the graph needs the triangles
    3. QUALITY GATE: the circle E_H1 (d = 64 mm) against the Peterson
       finite-width formula - more than 5% off means the hole mesh is too coarse
    4. LINEARITY: every geometry's solves must be exact multiples of its 10 MPa solve
"""

import argparse
import csv
import glob
import os
import shutil
import sys

import numpy as np

TRAIN_LOADS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
               65, 70, 75, 80, 85, 90, 95, 100, 105, 110]   # MPa
TEST_LOADS = [8, 60, 130]                                    # MPa
ALL_LOADS = sorted(set(TRAIN_LOADS + TEST_LOADS))
REF_LOAD = 10          # MPa - the quality gate and linearity check compare against this solve
PLATE_D = 250.0        # mm - plate height = the "D" of the Kt formula
FOLDERS = ("train", "val", "unseen_loads", "unseen_interp", "unseen_extrap")


def peterson_peak(d, D=PLATE_D, gross=float(REF_LOAD)):
    """Peak stress at a centred circular hole in a finite-width plate, tension."""
    r = d / D
    kt = 3.000 - 3.140 * r + 3.667 * r ** 2 - 1.527 * r ** 3
    return kt * gross * D / (D - d)                 # Kt applies to the net-section stress


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solved", default=os.path.join("data", "raw_si_cps4_solved"))
    ap.add_argument("--manifest", default=os.path.join("data", "odb_si_cps4", "si_manifest.csv"))
    ap.add_argument("--out", default=os.path.join("data", "si_cps4"))
    ap.add_argument("--val", default="", help="comma-separated shapes to use as validation instead, e.g. E_H3,E_V3")
    ap.add_argument("--skip-quality-gate", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.manifest):
        print("No manifest at %s - it is written by abaqus/generate_si_dataset_cps4.py "
              "in the folder where it ran." % a.manifest)
        return 1
    with open(a.manifest) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "a_mm" not in rows[0]:
        print("%s is not the SI manifest (no a_mm column)." % a.manifest)
        return 1

    # ---- which shapes are validation -------------------------------------------
    names = dict((r["name"], r) for r in rows)
    if a.val:
        wanted = [v.strip().replace("plate_si_", "") for v in a.val.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in names]
        if unknown:
            print("--val: %s not in the manifest. Known shapes: %s" % (", ".join(unknown), ", ".join(sorted(names))))
            return 1
        bad = [v for v in wanted if names[v]["split"] == "train"]
        if bad:
            print("--val: %s are TRAINING shapes - validation must be shapes the model never trains on." % ", ".join(bad))
            return 1
        for r in rows:                                  # the generator's own val shapes are not used then
            if r["split"] == "val":
                r["split"] = "unused"
        for v in wanted:
            names[v]["split"] = "val"
    rows = [r for r in rows if r["split"] != "unused"]
    split_count = {}
    for r in rows:
        split_count[r["split"]] = split_count.get(r["split"], 0) + 1
    if not split_count.get("val"):
        print("No validation shapes: the manifest has no 'val' rows (an older generator?) and --val was not given.")
        print("Either run the new abaqus/generate_si_dataset_cps4.py once more (it only adds the 4 validation")
        print("shapes), or borrow 4 test shapes:  python scripts/organize_si_cps4.py --val E_H3,E_V3,R_W24,R_H24")
        return 1

    # ---- the plan: which file goes where --------------------------------------------
    plan = []                                           # (geometry, load, folder)
    for r in rows:
        g, s = r["geometry"], r["split"]
        if s == "train":
            plan += [(g, L, "train") for L in TRAIN_LOADS] + [(g, L, "unseen_loads") for L in TEST_LOADS]
        elif s == "val":
            plan += [(g, L, "val") for L in TRAIN_LOADS]
        elif s == "test_interp":
            plan += [(g, L, "unseen_interp") for L in ALL_LOADS]
        elif s == "test_extrap":
            plan += [(g, L, "unseen_extrap") for L in ALL_LOADS]
        else:
            print("unknown split '%s' for %s in the manifest" % (s, g))
            return 1

    solved = {}
    for f in glob.glob(os.path.join(a.solved, "*.npz")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if "__" not in stem:
            continue
        geom, load = stem.rsplit("__", 1)
        try:
            solved[(geom, int(float(load)))] = f
        except ValueError:
            pass

    # ---- 1. completeness ----------------------------------------------------------
    needed = sorted(set((g, L) for g, L, _ in plan) | set((r["geometry"], REF_LOAD) for r in rows))
    missing = [k for k in needed if k not in solved]
    if missing:
        print("%d of %d needed solves are missing from %s:" % (len(missing), len(needed), a.solved))
        for g, L in missing[:25]:
            print("   %s__%d" % (g, L))
        if len(missing) > 25:
            print("   ... and %d more" % (len(missing) - 25))
        print("Run the generator again (it resumes), then abaqus python abaqus\\extract_odb.py ... --out %s" % a.solved)
        return 1
    print("completeness  %d geometries, %d files needed, all present" % (len(rows), len(needed)))
    print("              %s" % ", ".join("%s %d" % (k, v) for k, v in sorted(split_count.items())))

    # ---- 2. CPS4 + the triangle split --------------------------------------------
    for r in rows:
        d = np.load(solved[(r["geometry"], REF_LOAD)], allow_pickle=False)
        if "quads" not in d.files:
            print("%s has no 'quads': it was written by the OLD extract_odb.py (triangles only)."
                  % solved[(r["geometry"], REF_LOAD)])
            print("Replace abaqus\\extract_odb.py with the CPS4 version and extract again.")
            return 1
        nq, nt = len(d["quads"]), len(d["tris"])
        if nq == 0 or nt != 2 * nq:
            print("%s: %d quads but %d triangles - expected 2 triangles per quad (CPS4 mesh)."
                  % (r["geometry"], nq, nt))
            return 1
    print("CPS4          every mesh is 4-node quads, each split into 2 triangles for the graph")

    # ---- 3. quality gate ---------------------------------------------------------------
    circ = [r for r in rows if r["name"] == "E_H1"]
    if circ:
        d = np.load(solved[(circ[0]["geometry"], REF_LOAD)], allow_pickle=False)
        dia = 2 * float(circ[0]["a_mm"])
        got, want = float(d["mises"].max()), peterson_peak(dia)
        print("quality gate  circle d = %g mm at %d MPa: peak %.2f MPa vs Peterson %.2f MPa -> %.1f%%"
              % (dia, REF_LOAD, got, want, 100 * got / want))
        if abs(got / want - 1) > 0.05 and not a.skip_quality_gate:
            print("   More than 5% off: the hole mesh is not resolving the peak. Pass --skip-quality-gate to go on anyway.")
            return 1

    # ---- 4. linearity ----------------------------------------------------------------------
    worst, worst_at = 0.0, ""
    for r in rows:
        ref = np.load(solved[(r["geometry"], REF_LOAD)], allow_pickle=False)["mises"].astype(np.float64)
        for L in sorted(set(L for g, L, _ in plan if g == r["geometry"])):
            m = np.load(solved[(r["geometry"], L)], allow_pickle=False)["mises"].astype(np.float64)
            if m.shape != ref.shape:
                print("   %s: mesh at %d MPa differs from %d MPa - all loads must share one mesh"
                      % (r["geometry"], L, REF_LOAD))
                return 1
            dev = float(np.abs(m - ref * L / float(REF_LOAD)).max() / max(np.abs(m).max(), 1e-12))
            if dev > worst:
                worst, worst_at = dev, "%s at %d MPa" % (r["geometry"], L)
    print("linearity     largest deviation from exact scaling: %.2e (%s)" % (worst, worst_at or "-"))
    if worst > 1e-3:
        print("   WARNING: solves are not proportional to the load. Check the model is linear.")

    # ---- write --------------------------------------------------------------------------
    dirs = dict((k, os.path.join(a.out, k)) for k in FOLDERS)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(os.path.join(d, "*.npz")):     # a re-run with another --val must not leave stale copies
            os.remove(old)
    n, n_geo = dict((k, 0) for k in FOLDERS), dict((k, set()) for k in FOLDERS)
    for g, L, where in plan:
        shutil.copy2(solved[(g, L)], dirs[where])
        n[where] += 1
        n_geo[where].add(g)
    man_out = os.path.join(a.out, "si_manifest.csv")
    with open(man_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("")
    print("%-26s %4d files  (%d geometries x %d training loads)"
          % (dirs["train"], n["train"], len(n_geo["train"]), len(TRAIN_LOADS)))
    print("%-26s %4d files  (%d validation geometries x %d training loads: %s)"
          % (dirs["val"], n["val"], len(n_geo["val"]), len(TRAIN_LOADS),
             ", ".join(sorted(g.replace("plate_si_", "") for g in n_geo["val"]))))
    print("%-26s %4d files  (training geometries at %s MPa)"
          % (dirs["unseen_loads"], n["unseen_loads"], " / ".join(str(L) for L in TEST_LOADS)))
    print("%-26s %4d files  (%d interpolation geometries x %d loads)"
          % (dirs["unseen_interp"], n["unseen_interp"], len(n_geo["unseen_interp"]), len(ALL_LOADS)))
    print("%-26s %4d files  (%d extrapolation geometries x %d loads)"
          % (dirs["unseen_extrap"], n["unseen_extrap"], len(n_geo["unseen_extrap"]), len(ALL_LOADS)))
    print("%-26s the manifest with the split used" % man_out)
    print("\nNext: push data/si_cps4 to GitHub, then on Ubuntu:  bash scripts/b1_cps4.sh build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
