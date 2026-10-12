r"""
Train RUN B1 WITH A VALIDATION SET - edge-augmented MGN + anchor features in every
layer + random edges re-drawn every epoch, and every --eval-every epochs the R2 on
the training set AND on the validation set is logged, so R2 vs epoch can be
plotted for both (scripts/check_fit.py).

    bash scripts/b1_cps4.sh train                                   (the easy way, CPS4 SI dataset)
    python scripts/train_b1_val.py                                  defaults = the CPS4 SI dataset
    python scripts/train_b1_val.py --epochs 20 --eval-every 10 --results results_si_cps4_quick
                                                                    timing check first
    python scripts/train_b1_val.py --resume                         continue after a stop

    optional - start from the weights of your inch Run B1 instead of from random:
    python scripts/train_b1_val.py --init-from results_aug20_anchors_redraw/mgn_best.pt --results results_si_cps4_b1_ft

THE MODEL IS RUN B1, UNCHANGED  (mgn/anchor_trainer.py: AnchorMGN)
    20 layers, hidden 64, embedding 16, Adam lr 1e-5, batch 1, 10000 epochs, seed 0,
    20% random edges RE-DRAWN every epoch (seed = crc32(name) + aug_seed + epoch),
    edge flag as the 4th edge feature, load + 14 anchor numbers as the global input
    of every layer. The training step is the paper's own _train_epoch(). Only the
    data and the validation logging are new.

THE VALIDATION SET  (data/si_cps4/val - 4 shapes x 20 training loads, never trained on)
    Every --eval-every epochs (default 50, and at epoch 0 and the last epoch) the model
    predicts every training graph and every validation graph - eval mode, fp32, with
    the stored random-edge draw (seed aug_seed + 0) for both, so the two numbers are
    measured the same way - and logs:
        R2 per file (mean, median), R2 of all nodes pooled,
        loss = MSE on the z-scale, the same scale as the training loss,
        mean |peak error| %
    The TEST folders (unseen loads / interpolation / extrapolation) are never read here.

    How to read the two curves (scripts/check_fit.py draws them):
        both still rising                  -> under-trained, more epochs help
        both high, together, flat          -> converged, fits well (the goal)
        train rising, validation falling   -> over-fitting: mgn_best_val.pt is the epoch to keep
        train itself low                   -> under-fitting

WRITES  (results_si_cps4_b1/)
    loss_log.csv          epoch, loss, seconds                          every epoch (as train_anchor.py)
    r2_log.csv            epoch, train and validation R2, loss, peak error   every --eval-every epochs
    mgn_best_val.pt       weights at the HIGHEST VALIDATION mean R2     <- evaluate this one on the tests
    mgn_best.pt           weights at the lowest training loss           (what train_anchor.py keeps)
    mgn_latest.pt         weights + optimiser + epoch, for --resume (every --ckpt-every epochs)
    anchors_summary.csv   what the anchor features found in each training geometry
    multi_geometry_mgn.pt full model with its normalisation, when the run finishes
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import Case, _read_name, build_topology, load_dataset     # noqa: E402
from mgn.anchor_trainer import AnchorMGN                                    # noqa: E402
from mgn.anchors import ANCHOR_NAMES, anchor_features                       # noqa: E402

DEFAULT_DATA = os.path.join("data", "si_cps4", "dataset_si_cps4_aug20.pt")
DEFAULT_VAL = os.path.join("data", "si_cps4", "val")
DEFAULT_RESULTS = "results_si_cps4_b1"
R2_COLUMNS = ["epoch", "loss",
              "train_r2_mean", "train_r2_median", "train_r2_pooled", "train_loss_eval", "train_peak_err_pct",
              "val_r2_mean", "val_r2_median", "val_r2_pooled", "val_loss", "val_peak_err_pct", "seconds"]


def fmt(seconds):
    s = int(seconds)
    return "%dh%02dm%02ds" % (s // 3600, (s % 3600) // 60, s % 60)


def short(name):
    for p in ("plate_si_", "plate_own_", "plate_"):
        if name.startswith(p):
            return name[len(p):]
    return name


# ------------------------------------------------------------------ data

def read_val(folder, aug):
    """Validation .npz files -> Case objects with the SAME graph recipe as training:
    real mesh edges + aug_perc random edges from seed crc32(name) + aug_seed."""
    cases, topo = [], {}
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        name = (_read_name(d["geometry"]) if "geometry" in d.files
                else os.path.splitext(os.path.basename(f))[0].rsplit("__", 1)[0])
        coords = d["coords"].astype(np.float32)
        key = (name, len(coords))
        if key not in topo:
            ei, nt, fl = build_topology(coords, d["tris"].astype(np.int64), name,
                                        aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"])
            topo[key] = (coords, ei, nt, fl)
        coords, ei, nt, fl = topo[key]
        cases.append(Case(geometry=name, coordinates=coords, edge_index=ei, node_types=nt,
                          von_mises=d["mises"].astype(np.float32), metadata={"load": float(d["load"])},
                          edge_flag=fl))
    return cases


def anchor_summary(mgn, cases, path=None, title="ANCHORS FOUND"):
    """One line per geometry: hole nodes, corners, strengths."""
    seen, rows = set(), []
    for c in cases:
        if c.geometry in seen:
            continue
        seen.add(c.geometry)
        _, info = anchor_features(np.asarray(c.coordinates), mgn._mesh_edges(c), return_info=True)
        rows.append((c.geometry, info["n_hole_loops"], info["n_hole_nodes"], len(info["corners"]),
                     float(np.max(info["hole_strength"])) if info["n_hole_nodes"] else 0.0,
                     float(np.max(info["corner_strength"])) if len(info["corners"]) else 0.0))
    if path:
        with open(path, "w") as fh:
            fh.write("geometry,hole_loops,hole_nodes,corners,max_hole_strength,max_corner_strength\n")
            for r in rows:
                fh.write("%s,%d,%d,%d,%.4f,%.4f\n" % r)
    print("\n%s  (hole strength = log(1 + curvature x H); corner strength = excess turning, rad)" % title)
    print("  %-26s %6s %7s %8s %10s %10s" % ("geometry", "holes", "h.nodes", "corners", "max hole s", "max corn s"))
    for r in rows:
        print("  %-26s %6d %7d %8d %10.2f %10.2f" % r)
    no_hole = [r[0] for r in rows if r[1] == 0]
    if no_hole:
        print("  WARNING: no hole found in %s - is the mesh split into triangles (extract_odb.py, CPS4 version)?"
              % ", ".join(no_hole))


# ------------------------------------------------------------------ evaluation

def evaluate(mgn, cases):
    """R2 per file (mean, median), pooled R2, z-scale MSE and mean |peak error| %.
    Eval mode, fp32, each case's stored random-edge draw. Leaves the model in train mode."""
    y_std = float(mgn._y_std)
    r2s, losses, peaks = [], [], []
    n = s1 = s2 = sres = 0.0
    try:
        with torch.no_grad():
            for c in cases:
                p = np.asarray(mgn.predict(c), dtype=np.float64).ravel()
                t = np.asarray(c.von_mises, dtype=np.float64).ravel()
                e = p - t
                ss = float(((t - t.mean()) ** 2).sum())
                r2s.append(1.0 - float((e ** 2).sum()) / ss if ss > 0 else np.nan)
                losses.append(float(((e / y_std) ** 2).mean()))
                it, ip = int(np.argmax(t)), int(np.argmax(p))
                peaks.append(abs(100.0 * (p[ip] - t[it]) / t[it]))
                n += len(t)
                s1 += float(t.sum())
                s2 += float((t ** 2).sum())
                sres += float((e ** 2).sum())
    finally:
        mgn._model.train()
    sstot = s2 - s1 * s1 / max(n, 1.0)
    return dict(r2_mean=float(np.nanmean(r2s)), r2_median=float(np.nanmedian(r2s)),
                r2_pooled=1.0 - sres / sstot if sstot > 0 else float("nan"),
                loss=float(np.mean(losses)), peak=float(np.mean(peaks)),
                worst=int(np.nanargmin(r2s)), worst_r2=float(np.nanmin(r2s)))


def truncate_log(path, max_epoch):
    """On --resume: drop rows logged after the checkpoint (those epochs are re-done)."""
    if not os.path.isfile(path):
        return
    lines = open(path).read().splitlines()
    if not lines:
        return
    keep = [lines[0]]
    for ln in lines[1:]:
        try:
            if int(float(ln.split(",")[0])) <= max_epoch:
                keep.append(ln)
        except ValueError:
            pass
    open(path, "w").write("\n".join(keep) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA, help="training dataset (.pt, python -m mgn.dataset)")
    ap.add_argument("--val-dir", default=DEFAULT_VAL, help="validation .npz files (scripts/organize_si_cps4.py)")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=50, help="epochs between train / validation R2 evaluations")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="start from these weights (e.g. your inch B1 mgn_best.pt) instead of random")
    ap.add_argument("--no-anchors", action="store_true", help="ablation: switch the anchor features off")
    ap.add_argument("--no-redraw", action="store_true", help="ablation: keep one fixed random-edge draw")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ---- data ---------------------------------------------------------------
    if not os.path.isfile(a.data):
        print("No %s.\nBuild it first:  python -m mgn.dataset --raw data/si_cps4/train --aug-perc 0.2 --out %s"
              % (a.data, a.data))
        return 1
    cases, aug = load_dataset(a.data, return_meta=True)
    if aug["aug_perc"] <= 0 and not a.no_redraw:
        print("%s has no random edges (aug_perc = 0), so there is nothing to re-draw. Build it with "
              "--aug-perc 0.2 or pass --no-redraw." % a.data)
        return 1
    val_cases = read_val(a.val_dir, aug) if os.path.isdir(a.val_dir) else []
    if not val_cases:
        print("No validation files in %s - run scripts/organize_si_cps4.py (Windows) and push data/si_cps4." % a.val_dir)
        return 1
    geoms = sorted(set(c.geometry for c in cases))
    val_geoms = sorted(set(c.geometry for c in val_cases))
    both = sorted(set(geoms) & set(val_geoms))
    if both:
        print("STOP: %s is in the training data AND in the validation folder. Validation shapes must "
              "never be trained on." % ", ".join(both))
        return 1

    anchors, redraw = not a.no_anchors, not a.no_redraw
    RESULTS = a.results
    LATEST, BEST = os.path.join(RESULTS, "mgn_latest.pt"), os.path.join(RESULTS, "mgn_best.pt")
    BEST_VAL = os.path.join(RESULTS, "mgn_best_val.pt")
    LOG, R2LOG = os.path.join(RESULTS, "loss_log.csv"), os.path.join(RESULTS, "r2_log.csv")
    os.makedirs(RESULTS, exist_ok=True)

    y = np.concatenate([c.von_mises for c in cases])
    mgn = AnchorMGN(num_layers=a.layers, hidden_channels=a.hidden, embedding_dim=a.embedding,
                    learning_rate=a.lr, epochs=a.epochs, global_features=["load"],
                    aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"], anchors=anchors, redraw=redraw)
    config = mgn.config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_val_loads = len(val_cases) // max(len(val_geoms), 1)
    print("=" * 74)
    print("MGN TRAINING - RUN B1 WITH VALIDATION")
    print("=" * 74)
    print("training       %d cases  (%d geometries x %d loads)  %s"
          % (len(cases), len(geoms), len(cases) // max(len(geoms), 1), a.data))
    print("validation     %d cases  (%d geometries x %d loads: %s)  %s"
          % (len(val_cases), len(val_geoms), n_val_loads, ", ".join(short(g) for g in val_geoms), a.val_dir))
    print("device         %s%s" % (dev, "  (" + torch.cuda.get_device_name(0) + ")" if dev == "cuda" else ""))
    print("epochs         %d   batch %d   lr %g   layers %d   hidden %d   R2 every %d epochs"
          % (a.epochs, a.batch_size, a.lr, a.layers, a.hidden, a.eval_every))
    print("augmentation   %.0f%% random edges, base seed %d, %s"
          % (100 * aug["aug_perc"], aug["aug_seed"], "RE-DRAWN every epoch" if redraw else "fixed (one draw)"))
    print("anchors        %s" % ("hole + corner, 14 numbers per node, in every layer" if anchors else "off"))
    print("results        %s" % RESULTS)
    print("=" * 74)

    # ---- the paper's preprocessing and model ---------------------------------
    t_prep = time.time()
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    mgn._build_model()
    if anchors:
        anchor_summary(mgn, cases, os.path.join(RESULTS, "anchors_summary.csv"))
        anchor_summary(mgn, val_cases, None, title="ANCHORS FOUND - VALIDATION SHAPES")
        names = ["load"] + ANCHOR_NAMES
        print("\nglobal inputs  %d columns: %s" % (len(names), ", ".join(names)))
    print("parameters     %d" % sum(p.numel() for p in mgn._model.parameters()))
    print("preprocessing  %.0f s" % (time.time() - t_prep))

    optimizer = torch.optim.Adam(mgn._model.parameters(), lr=a.lr)
    loss_fn = torch.nn.MSELoss()
    start_epoch, best, best_val, best_val_epoch = 0, float("inf"), -float("inf"), 0

    if a.resume and os.path.isfile(LATEST):
        ck = torch.load(LATEST, map_location=mgn.device, weights_only=False)
        if ck.get("augmentation") != aug or ck.get("config", {}).get("anchors") != anchors \
                or ck.get("config", {}).get("redraw") != redraw:
            print("\nREFUSING TO RESUME: %s was trained with a different setup (%s, %s)."
                  % (LATEST, ck.get("augmentation"), ck.get("config")))
            return 1
        mgn._model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best = ck["epoch"], ck.get("best", float("inf"))
        best_val, best_val_epoch = ck.get("best_val", -float("inf")), ck.get("best_val_epoch", 0)
        mgn.train_losses = ck.get("train_losses", [])
        np.random.seed(a.seed + start_epoch)             # shuffling continues from a fresh, reproducible state
        truncate_log(LOG, start_epoch)
        truncate_log(R2LOG, start_epoch)
        print("\nRESUMED from epoch %d  (best loss %.6f, best validation R2 %.4f at epoch %d)"
              % (start_epoch, best, best_val, best_val_epoch))
    else:
        if a.resume:
            print("\n--resume given but no %s found; starting fresh." % LATEST)
        if a.init_from:
            ck = torch.load(a.init_from, map_location=mgn.device, weights_only=False)
            state = ck.get("model", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck
            try:
                mgn._model.load_state_dict(state)
            except RuntimeError as exc:
                print("--init-from %s does not fit this model (different layers / hidden / anchors?):\n%s"
                      % (a.init_from, str(exc).splitlines()[0]))
                return 1
            config["init_from"] = a.init_from
            print("initial weights  %s  (the normalisation is this dataset's own)" % a.init_from)

    if start_epoch == 0 or not os.path.isfile(LOG):
        open(LOG, "w").write("epoch,loss,seconds\n")
    if start_epoch == 0 or not os.path.isfile(R2LOG):
        open(R2LOG, "w").write(",".join(R2_COLUMNS) + "\n")

    def state(epoch):
        return {"model": mgn._model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best": best, "best_val": best_val, "best_val_epoch": best_val_epoch,
                "train_losses": mgn.train_losses, "augmentation": aug, "config": config}

    t0 = time.time()

    def score(epoch, loss):
        """Train + validation R2 -> r2_log.csv and, if best so far, mgn_best_val.pt."""
        nonlocal best_val, best_val_epoch
        tr, va = evaluate(mgn, cases), evaluate(mgn, val_cases)
        with open(R2LOG, "a") as fh:
            fh.write(",".join(["%d" % epoch, "" if loss is None else "%.8f" % loss] +
                              ["%.6f" % tr[k] for k in ("r2_mean", "r2_median", "r2_pooled")] +
                              ["%.8f" % tr["loss"], "%.4f" % tr["peak"]] +
                              ["%.6f" % va[k] for k in ("r2_mean", "r2_median", "r2_pooled")] +
                              ["%.8f" % va["loss"], "%.4f" % va["peak"], "%.1f" % (time.time() - t0)]) + "\n")
        mark = ""
        if va["r2_mean"] > best_val:
            best_val, best_val_epoch = va["r2_mean"], epoch
            torch.save({"model": mgn._model.state_dict(), "epoch": epoch,
                        "loss": float("nan") if loss is None else loss, "val_r2": va["r2_mean"],
                        "augmentation": aug, "config": config}, BEST_VAL)
            mark = "  <- best validation so far, saved"
        wv = val_cases[va["worst"]]
        print("   R2 at epoch %d:  train %.4f   validation %.4f   (gap %+.4f, worst val %s @ %g = %.4f)%s"
              % (epoch, tr["r2_mean"], va["r2_mean"], tr["r2_mean"] - va["r2_mean"], short(wv.geometry),
                 wv.metadata["load"], va["worst_r2"], mark))

    if start_epoch == 0:
        print("\nscoring the untrained model (epoch 0) ...")
        score(0, None)

    print("\nepoch      loss        elapsed     eta")
    print("-" * 52)
    interrupted = False
    try:
        for epoch in range(start_epoch, a.epochs):
            mgn.redraw_edges(cases, fem_data, epoch)                  # new random edges for this epoch
            loss = mgn._train_epoch(cases, fem_data, a.batch_size, optimizer, loss_fn)   # the paper's step
            mgn.train_losses.append(loss)
            elapsed = time.time() - t0
            eta = elapsed / (epoch - start_epoch + 1) * (a.epochs - epoch - 1)
            open(LOG, "a").write("%d,%.8f,%.1f\n" % (epoch + 1, loss, elapsed))
            if loss < best:
                best = loss
                torch.save({"model": mgn._model.state_dict(), "epoch": epoch + 1, "loss": loss,
                            "augmentation": aug, "config": config}, BEST)
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print("%6d   %.6f   %s   %s" % (epoch + 1, loss, fmt(elapsed), fmt(eta)))
            if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
                score(epoch + 1, loss)
            if (epoch + 1) % a.ckpt_every == 0:
                torch.save(state(epoch + 1), LATEST)
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted. Saving checkpoint...")

    torch.save(state(len(mgn.train_losses)), LATEST)
    print("\n" + "=" * 74)
    print("%s at epoch %d after %s" % ("Stopped" if interrupted else "Finished", len(mgn.train_losses),
                                      fmt(time.time() - t0)))
    if mgn.train_losses:
        print("final loss            %.6f\nbest loss             %.6f" % (mgn.train_losses[-1], best))
    print("best validation R2    %.4f at epoch %d   -> %s" % (best_val, best_val_epoch, BEST_VAL))
    if interrupted or len(mgn.train_losses) < a.epochs:
        print("\nContinue with:  python scripts/train_b1_val.py --resume --data %s --val-dir %s --results %s%s%s"
              % (a.data, a.val_dir, RESULTS, " --no-anchors" if not anchors else "", " --no-redraw" if not redraw else ""))
    else:
        mgn.save(os.path.join(RESULTS, "multi_geometry_mgn.pt"))
    print("\nR2 vs epoch plot:  python scripts/check_fit.py --curves-only --ckpt %s" % BEST_VAL)
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
