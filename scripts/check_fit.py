r"""
Over- or under-fitting check for an MGN run, with R2-vs-epoch curves for the
training and the validation set when the run logged them (train_b1_val.py).

    CPS4 SI dataset, Run B1 with validation (scripts/train_b1_val.py):
    bash scripts/b1_cps4.sh curves          R2 vs epoch only - fast, works WHILE training runs
    bash scripts/b1_cps4.sh fitcheck        the full check (run bash scripts/b1_cps4.sh eval first)

    the same, written out:
    python scripts/check_fit.py --curves-only --ckpt results_si_cps4_b1/mgn_best_val.pt
    python scripts/check_fit.py --ckpt results_si_cps4_b1/mgn_best_val.pt --draws 8 \
        --train-data data/si_cps4/dataset_si_cps4_aug20.pt --val-dir data/si_cps4/val

    inch Run B1 (trained without a validation set), as before:
    python scripts/check_fit.py --ckpt results_aug20_anchors_redraw/mgn_best.pt --draws 8

Run eval_own_unseen.py with the same --draws first: this script reads its
metrics_all.csv for the three test sets.

WHAT IT DOES
    1. Loss curve (loss_log.csv): has training stopped improving?
    2. R2 vs epoch (r2_log.csv, written by train_b1_val.py every few epochs):
       training and validation R2 on one plot. How to read it:
           both still rising                  -> under-trained: more epochs help
           both high, close together, flat    -> converged, generalises (the goal)
           train rising, validation falling   -> over-fitting (keep mgn_best_val.pt)
           train itself low                   -> under-fitting
    3. Training-set fit of THIS checkpoint, and validation-set fit, averaged
       over --draws random-edge draws, scored exactly like eval_own_unseen.py.
    4. The "fit ladder" - the same numbers on
           train            seen geometry, seen load
           validation       new geometry inside the range, training loads
           unseen_loads     seen geometry, new load
           unseen_interp    new geometry inside the range       (test)
           unseen_extrap    new geometry outside the range      (test)
       and a verdict:
           underfitting  -> the TRAIN level itself is poor
           overfitting   -> train excellent, validation (or interpolation) clearly worse,
                            or the validation curve turned down while train kept rising
           extrapolation -> only the shapes outside the range are poor
                            (a data-coverage problem, not over-fitting)

WRITES  (into <checkpoint folder>/fit_check/)
    r2_vs_epoch.png         R2 vs epoch for train and validation, 1 - R2 on a log scale, and the loss
    fit_check.png           loss curve, R2 vs epoch, R2 and peak error at every level
    fit_ladder.csv          the summary table
    fit_train_metrics.csv   every training file: R2, RMSE, peak error
    fit_val_metrics.csv     every validation file (when --val-dir is given)
"""

import argparse
import csv
import glob
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

LEVELS = [("train", "train  (seen geometry, seen load)"),
          ("validation", "validation  (new geometry, training loads)"),
          ("unseen_loads", "unseen loads  (seen geometry)"),
          ("unseen_interp", "unseen geometry - interpolation"),
          ("unseen_extrap", "unseen geometry - EXTRAPOLATION")]
TICK = dict(train="train", validation="validation", unseen_loads="unseen\nloads",
            unseen_interp="new shape\n(interp)", unseen_extrap="new shape\n(EXTRAP)")

# thresholds used for the verdict (change them if you want a stricter test)
TRAIN_R2_OK = 0.99          # mean R2 on the training set below this -> underfitting
TRAIN_PEAK_OK = 5.0         # mean |peak error| on the training set above this (%) -> underfitting at peaks
GAP_R2 = 0.03               # train R2 - validation (or interp) R2 above this -> generalisation gap (over-fitting)
CURVE_DROP = 0.005          # validation R2 this far below its best, while train R2 rose -> over-fitting trend
STILL_RISING = 0.002        # validation R2 gained more than this over the last 20% of epochs -> under-trained

# colours: the validated categorical slots 1 and 2 (train = blue, validation = orange); text stays in ink
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LEVEL_COLOUR = dict(train=BLUE, validation=ORANGE, unseen_loads=MUTED, unseen_interp=MUTED, unseen_extrap=MUTED)
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "legend.frameon": False})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------- logs

def read_csv_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def loss_report(log_path):
    rows = read_csv_rows(log_path)
    if not rows:
        print("no loss_log.csv at %s - skipping the loss curve" % log_path)
        return None
    ep = np.array([int(r["epoch"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    n = len(loss)
    w = max(1, min(500, n // 5))
    late, before = loss[-w:].mean(), loss[-2 * w:-w].mean() if n >= 2 * w else np.nan
    drop = 100 * (before - late) / before if np.isfinite(before) else np.nan
    print("\nLOSS CURVE  (%s)" % log_path)
    print("  epochs logged              %d" % n)
    for e in (1, 100, 1000, 2000, 5000, n):
        if e <= n:
            print("  loss at epoch %-6d       %.3e   (training fit R2 ~ %.5f)" % (e, loss[e - 1], 1 - loss[e - 1]))
    print("  lowest loss                %.3e at epoch %d" % (loss.min(), ep[int(loss.argmin())]))
    if n < 200:
        print("  only %d epochs logged - too early to judge convergence" % n)
    elif np.isfinite(drop):
        print("  mean of last %d epochs vs the %d before: %.1f %% lower" % (w, w, drop))
        print("  -> %s" % ("STILL IMPROVING: more epochs (or a higher lr) could still help = not fully trained"
                         if drop > 10 else "levelled off: training has converged"))
    return ep, loss, drop


def read_curves(path):
    """r2_log.csv (train_b1_val.py) -> dict of arrays, or None."""
    rows = read_csv_rows(path)
    if not rows:
        return None
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan     # noqa: E731
    keys = [k for k in rows[0].keys() if k != "epoch"]
    c = dict((k, np.array([f(r, k) for r in rows])) for k in keys)
    c["epoch"] = np.array([int(float(r["epoch"])) for r in rows])
    return c


def smooth(v, k=3):
    """Centered moving average over k evaluation points (edges use what exists)."""
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    for i in range(len(v)):
        lo, hi = max(0, i - k // 2), min(len(v), i + k // 2 + 1)
        out[i] = np.nanmean(v[lo:hi])
    return out


def curve_report(c):
    """Print the R2-vs-epoch table and return the curve verdict lines."""
    ep, tr, va = c["epoch"], c["train_r2_mean"], c["val_r2_mean"]
    print("\nR2 VS EPOCH  (train and validation, mean R2 per file; from r2_log.csv)")
    print("  %8s %12s %12s %10s %14s %14s" % ("epoch", "train R2", "val R2", "gap", "train loss", "val loss"))
    idx = sorted(set(list(np.unique(np.linspace(0, len(ep) - 1, min(len(ep), 9)).round().astype(int)))
                     + [int(np.nanargmax(va))]))
    for i in idx:
        print("  %8d %12.5f %12.5f %+10.5f %14.3e %14.3e%s" % (ep[i], tr[i], va[i], tr[i] - va[i],
              c["train_loss_eval"][i], c["val_loss"][i], "   <- best validation" if i == int(np.nanargmax(va)) else ""))
    lines = []
    if len(ep) < 3:
        return ["only %d R2 point(s) logged so far - too early for a verdict." % len(ep)]
    ib = int(np.nanargmax(va))
    st, sv = smooth(tr), smooth(va)
    drop = float(np.nanmax(sv) - sv[-1])
    rose = float(st[-1] - st[int(np.nanargmax(sv))])
    gap = float(tr[-1] - va[-1])
    n20 = max(1, int(round(0.2 * (len(ep) - 1))))
    val_gain = float(sv[-1] - sv[-1 - n20]) if len(ep) > n20 else np.nan
    lines.append("best validation R2 %.4f at epoch %d of %d; last: train %.4f, validation %.4f (gap %+.4f)"
                 % (va[ib], ep[ib], ep[-1], tr[-1], va[-1], gap))
    if tr[-1] < TRAIN_R2_OK:
        lines.append("UNDERFITTING (curve): the training R2 itself ends at %.4f < %.2f." % (tr[-1], TRAIN_R2_OK))
    if drop > CURVE_DROP and rose > 0:
        lines.append("OVERFITTING (curve): validation R2 fell %.4f below its best while train R2 rose %.4f "
                     "- the model started memorising after epoch %d. Use mgn_best_val.pt (that epoch)."
                     % (drop, rose, ep[ib]))
    elif np.isfinite(val_gain) and val_gain > STILL_RISING:
        lines.append("STILL IMPROVING (curve): validation R2 rose %.4f over the last 20%% of the epochs "
                     "- more epochs would still help." % val_gain)
    else:
        lines.append("no clear downturn of the validation curve at the end -> no over-fitting trend.")
    if gap > GAP_R2:
        lines.append("GAP (curve): train - validation = %.4f > %.2f at the end -> generalisation gap." % (gap, GAP_R2))
    return lines


def plot_r2_vs_epoch(c, lr, path, title):
    """(a) R2 vs epoch, (b) 1 - R2 on a log scale, (c) loss on the z-scale."""
    ep, tr, va = c["epoch"], c["train_r2_mean"], c["val_r2_mean"]
    ib = int(np.nanargmax(va))
    best_lab = "best validation: R\u00b2 %.4f at epoch %d" % (va[ib], ep[ib])
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))

    # (a) R2 itself. The untrained model (epoch 0, R2 far below 0) is left out of this panel only.
    ax = axs[0]
    k = ep > 0 if (ep > 0).sum() >= 1 else np.ones(len(ep), bool)
    ax.plot(ep[k], tr[k], color=BLUE, lw=2, marker="o", ms=3.5, label="train  (last %.4f)" % tr[-1])
    ax.plot(ep[k], va[k], color=ORANGE, lw=2, marker="o", ms=3.5, label="validation  (last %.4f)" % va[-1])
    ax.axvline(ep[ib], color=MUTED, lw=1, label=best_lab)
    lo = min(np.nanmin(tr[k]), np.nanmin(va[k]))
    ax.set_ylim(max(lo - 0.02, -0.5), 1.0 + 0.004)
    ax.set_xlabel("epoch")
    ax.set_ylabel("R\u00b2  (mean over files)")
    ax.set_title("(a) R\u00b2 vs epoch" + ("  (epoch 0 = untrained, not shown)" if (ep == 0).any() and k.sum() < len(ep) else "")
                 + ("\n     axis cut at \u22120.5" if lo < -0.5 else ""))
    ax.legend(loc="lower right", fontsize=9)
    style(ax)

    # (b) what is left to learn, on a log scale: near R2 = 1 the gap is only visible here
    ax = axs[1]
    ax.semilogy(ep, np.clip(1 - tr, 1e-6, None), color=BLUE, lw=2, marker="o", ms=3.5, label="train")
    ax.semilogy(ep, np.clip(1 - va, 1e-6, None), color=ORANGE, lw=2, marker="o", ms=3.5, label="validation")
    ax.axvline(ep[ib], color=MUTED, lw=1, label="best validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("1 \u2212 R\u00b2  (log scale, lower is better)")
    ax.set_title("(b) 1 \u2212 R\u00b2: the space between the lines\nis the generalisation gap")
    ax.legend(loc="upper right", fontsize=9)
    style(ax)

    # (c) loss on the z-scale: per-epoch training loss, and train / validation scored the same way
    ax = axs[2]
    if lr:
        e_all, loss, _ = lr
        wdw = max(1, min(200, len(loss) // 20))
        sm = np.convolve(loss, np.ones(wdw) / wdw, mode="valid")
        ax.semilogy(e_all[wdw - 1:], sm, color="#9ec5f4", lw=1.2,
                    label="training loss, every epoch" + (" (%d-epoch average)" % wdw if wdw > 1 else ""))
    ax.semilogy(ep, c["train_loss_eval"], color=BLUE, lw=2, marker="o", ms=3.5, label="train (scored like validation)")
    ax.semilogy(ep, c["val_loss"], color=ORANGE, lw=2, marker="o", ms=3.5, label="validation")
    ax.axvline(ep[ib], color=MUTED, lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss  (MSE of z-scored stress)")
    ax.set_title("(c) loss: validation turning up while train\nkeeps falling = over-fitting")
    ax.legend(loc="upper right", fontsize=9)
    style(ax)

    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  wrote %s" % path)


# ----------------------------------------------------------------- fits

def summarise(rows):
    r2 = np.array([r["r2"] for r in rows], dtype=float)
    pk = np.array([abs(r["peak_err_pct"]) for r in rows], dtype=float)
    nr = np.array([r["rmse"] / r["peak_true"] for r in rows], dtype=float)
    worst = rows[int(np.nanargmin(r2))]
    return dict(n=len(rows), mean_r2=float(np.nanmean(r2)), median_r2=float(np.nanmedian(r2)),
                worst_r2=float(np.nanmin(r2)), worst_case="%s @ %g" % (short(worst["geometry"]), worst["load"]),
                mean_abs_peak=float(np.nanmean(pk)), mean_nrmse=100 * float(np.nanmean(nr)))


def short(name):
    for p in ("plate_own_", "plate_si_", "plate_"):
        if name.startswith(p):
            return name[len(p):]
    return name


def read_test_csv(path):
    out = {}
    for r in read_csv_rows(path):
        rmse_col = [k for k in r if k.startswith("rmse")][0]
        peak_col = [k for k in r if k.startswith("peak_true")][0]
        out.setdefault(r["set"], []).append(dict(
            geometry=r["geometry"], load=float(r["load"]), r2=float(r["r2"]),
            rmse=float(r[rmse_col]), peak_true=float(r[peak_col]),
            peak_err_pct=float(r["peak_err_pct"])))
    return out


def fit_cases(E, mgn, items, draws, Case, build_topology, augment_edges, geometry_seed, torch, what):
    """items: Case objects (training, stored draw = k 0) or dicts from .npz (validation).
    Prediction = mean over `draws` random-edge draws (seeds aug_seed + 0 .. draws-1)."""
    rows = []
    for i, it in enumerate(items):
        preds = []
        for k in range(max(1, draws)):
            if isinstance(it, dict):                       # .npz -> build exactly like eval_own_unseen.py
                ei, nt, fl = build_topology(it["coords"], it["tris"], it["name"], aug_perc=mgn.aug_perc,
                                            aug_seed=mgn.aug_seed + k)[:3]
                case = Case(geometry=it["name"], coordinates=it["coords"], edge_index=ei, node_types=nt,
                            von_mises=np.zeros(len(it["coords"]), dtype=np.float32),
                            metadata={"load": it["load"]}, edge_flag=fl)
            elif k == 0:
                case = it                                  # the stored draw = seed aug_seed + 0
            else:
                mesh = mgn._mesh_edges(it)
                ei, fl = augment_edges(mesh, len(it.coordinates), perc=mgn.aug_perc,
                                       seed=geometry_seed(it.geometry, mgn.aug_seed + k))
                case = Case(geometry=it.geometry, coordinates=it.coordinates, edge_index=torch.from_numpy(ei),
                            node_types=it.node_types, von_mises=it.von_mises, metadata=it.metadata,
                            edge_flag=torch.from_numpy(fl))
            with torch.no_grad():
                preds.append(np.asarray(mgn.predict(case), dtype=np.float64).ravel())
        pred = np.mean(preds, axis=0)
        if isinstance(it, dict):
            truth, geom, load = it["truth"], it["name"], it["load"]
        else:
            truth, geom, load = np.asarray(it.von_mises, dtype=np.float64), it.geometry, float(it.metadata["load"])
        m = E.metrics(truth, pred)
        rows.append(dict(geometry=geom, load=load, r2=m["r2"], rmse=m["rmse"],
                         peak_true=m["peak_true"], peak_pred=m["peak_pred"], peak_err_pct=m["peak_err_pct"]))
        if (i + 1) % 60 == 0 or i + 1 == len(items):
            print("  %s %d/%d" % (what, i + 1, len(items)))
    return rows


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geometry", "load", "r2", "rmse", "peak_true", "peak_pred", "peak_err_pct"])
        for r in sorted(rows, key=lambda r: (r["geometry"], r["load"])):
            w.writerow([r["geometry"], "%g" % r["load"], "%.6f" % r["r2"], "%.4f" % r["rmse"],
                        "%.4f" % r["peak_true"], "%.4f" % r["peak_pred"], "%.4f" % r["peak_err_pct"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join("results_aug20_anchors_redraw", "mgn_best.pt"))
    ap.add_argument("--train-data", default=os.path.join("data", "dataset_own_aug20.pt"))
    ap.add_argument("--val-dir", default=None, help="validation .npz folder, e.g. data/si_cps4/val")
    ap.add_argument("--draws", type=int, default=1, help="average over this many random-edge draws (use the same as for eval_own_unseen.py)")
    ap.add_argument("--test-dir", default=None, help="default: <ckpt folder>/unseen_eval[_drawsK]")
    ap.add_argument("--curves-only", action="store_true", help="only the R2-vs-epoch plot from the logs (no model, fast)")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    a = ap.parse_args()

    folder = os.path.dirname(a.ckpt) or "."
    out = os.path.join(folder, "fit_check")
    os.makedirs(out, exist_ok=True)
    test_dir = a.test_dir or os.path.join(folder, "unseen_eval" + ("_draws%d" % a.draws if a.draws > 1 else ""))

    # ---- 1. loss curve and 2. R2 vs epoch --------------------------------
    lr = loss_report(os.path.join(folder, "loss_log.csv"))
    curves = read_curves(os.path.join(folder, "r2_log.csv"))
    curve_lines = []
    if curves is not None and len(curves["epoch"]) >= 1:
        curve_lines = curve_report(curves)
        plot_r2_vs_epoch(curves, lr, os.path.join(out, "r2_vs_epoch.png"),
                         "Learning curves  -  %s   (validation = shapes the model never trains on)" % folder)
        print("\nCURVE VERDICT")
        for ln in curve_lines:
            print("  " + ln)
    else:
        print("\nno r2_log.csv in %s - this run was trained without a validation set "
              "(scripts/train_b1_val.py writes one)." % folder)
    if a.curves_only:
        return 0

    # ---- 3. fits of this checkpoint ------------------------------------------
    import torch                                                  # noqa: E402  (only needed from here)
    import eval_own_unseen as E                                   # noqa: E402
    from mgn.dataset import Case, build_topology, load_dataset    # noqa: E402
    from mgn.graph import augment_edges, geometry_seed            # noqa: E402

    margs = types.SimpleNamespace(ckpt=a.ckpt, train_data=a.train_data, layers=a.layers,
                                  hidden=a.hidden, embedding=a.embedding)
    mgn, desc, _ = E.load_model(margs)
    print("\nmodel   %s" % desc)
    cases = load_dataset(a.train_data)
    print("train   %d cases from %s,  prediction = mean of %d random-edge draw(s)" % (len(cases), a.train_data, a.draws))
    helpers = (Case, build_topology, augment_edges, geometry_seed, torch)
    sets = {"train": fit_cases(E, mgn, cases, a.draws, *helpers, what="train")}
    write_rows(sets["train"], os.path.join(out, "fit_train_metrics.csv"))
    if a.val_dir:
        val_items = E.read_set(a.val_dir) if os.path.isdir(a.val_dir) else []
        if val_items:
            print("val     %d files from %s" % (len(val_items), a.val_dir))
            sets["validation"] = fit_cases(E, mgn, val_items, a.draws, *helpers, what="validation")
            write_rows(sets["validation"], os.path.join(out, "fit_val_metrics.csv"))
        else:
            print("val     no .npz files in %s - validation level skipped" % a.val_dir)

    # ---- 4. the fit ladder ------------------------------------------------
    tests = read_test_csv(os.path.join(test_dir, "metrics_all.csv"))
    if not tests:
        print("\nNo %s - run eval_own_unseen.py with --draws %d first for the test levels."
              % (os.path.join(test_dir, "metrics_all.csv"), a.draws))
    for k in ("unseen_loads", "unseen_interp", "unseen_extrap"):
        if tests.get(k):
            sets[k] = tests[k]
    summ = dict((k, summarise(sets[k])) for k, _ in LEVELS if sets.get(k))

    print("\nFIT LADDER   (test levels from %s)" % test_dir)
    print("  %-44s %6s %9s %9s %9s  %-24s %13s %12s" % ("level", "files", "mean R2", "median", "worst", "worst case",
                                                       "mean|peak err|", "RMSE/peak"))
    print("  " + "-" * 138)
    with open(os.path.join(out, "fit_ladder.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "files", "mean_r2", "median_r2", "worst_r2", "worst_case", "mean_abs_peak_err_pct",
                    "mean_rmse_over_peak_pct"])
        for k, title in LEVELS:
            if k not in summ:
                continue
            s = summ[k]
            print("  %-44s %6d %9.4f %9.4f %9.4f  %-24s %12.2f%% %11.2f%%" % (title, s["n"], s["mean_r2"], s["median_r2"],
                  s["worst_r2"], s["worst_case"], s["mean_abs_peak"], s["mean_nrmse"]))
            w.writerow([k, s["n"], "%.6f" % s["mean_r2"], "%.6f" % s["median_r2"], "%.6f" % s["worst_r2"], s["worst_case"],
                        "%.4f" % s["mean_abs_peak"], "%.4f" % s["mean_nrmse"]])

    # ---- verdict ----------------------------------------------------------
    t = summ["train"]
    print("\nVERDICT")
    if t["mean_r2"] < TRAIN_R2_OK or t["mean_abs_peak"] > TRAIN_PEAK_OK:
        print("  UNDERFITTING: the model does not even fit its own training data well "
              "(train mean R2 %.4f, mean |peak err| %.1f %%)." % (t["mean_r2"], t["mean_abs_peak"]))
        print("  -> more epochs / capacity / better features; more data will NOT fix this.")
    else:
        print("  training fit is good (train mean R2 %.4f, mean |peak err| %.1f %%) -> no underfitting."
              % (t["mean_r2"], t["mean_abs_peak"]))
    ref = "validation" if "validation" in summ else "unseen_interp" if "unseen_interp" in summ else None
    if ref:
        gap = t["mean_r2"] - summ[ref]["mean_r2"]
        pk_ratio = summ[ref]["mean_abs_peak"] / max(t["mean_abs_peak"], 1e-9)
        if gap > GAP_R2:
            print("  OVERFITTING / MEMORISATION: R2 drops by %.3f from train to new shapes (%s), peak error x%.1f."
                  % (gap, ref, pk_ratio))
            print("  -> more (or more varied) training geometries, or stop earlier (mgn_best_val.pt).")
        else:
            print("  generalisation gap to new shapes inside the range is small (%s: R2 drop %.3f, peak error x%.1f) "
                  "-> no meaningful overfitting." % (ref, gap, pk_ratio))
    for ln in curve_lines[1:]:
        print("  " + ln)
    if "unseen_extrap" in summ and "unseen_interp" in summ:
        if summ["unseen_interp"]["mean_r2"] - summ["unseen_extrap"]["mean_r2"] > GAP_R2:
            print("  EXTRAPOLATION GAP: shapes outside the training range are worse (worst: %s). This is missing "
                  "data coverage, not overfitting." % summ["unseen_extrap"]["worst_case"])

    # ---- figure -----------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
    ax = axs[0, 0]
    if lr:
        ep, loss, _ = lr
        ax.semilogy(ep, loss, color="#9ec5f4", lw=0.6, label="loss each epoch")
        wdw = max(1, min(200, len(loss) // 20))
        sm = np.convolve(loss, np.ones(wdw) / wdw, mode="valid")
        ax.semilogy(ep[wdw - 1:], sm, color=BLUE, lw=2, label="moving average (%d epochs)" % wdw)
        ax.axvline(ep[int(loss.argmin())], color=MUTED, lw=1, label="lowest training loss")
    if curves is not None:
        ax.semilogy(curves["epoch"], curves["val_loss"], color=ORANGE, lw=2, marker="o", ms=3, label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE of z-scored stress)")
    ax.legend(fontsize=9)
    ax.set_title("(a) loss: still falling = under-trained")
    style(ax)

    ax = axs[0, 1]
    if curves is not None:
        ce, ctr, cva = curves["epoch"], curves["train_r2_mean"], curves["val_r2_mean"]
        if (ce > 0).sum() >= 1:                           # epoch 0 = the untrained model, far below 0
            ce, ctr, cva = ce[ce > 0], ctr[ce > 0], cva[ce > 0]
        ax.plot(ce, ctr, color=BLUE, lw=2, marker="o", ms=3, label="train")
        ax.plot(ce, cva, color=ORANGE, lw=2, marker="o", ms=3, label="validation")
        ax.set_ylim(max(min(np.nanmin(ctr), np.nanmin(cva)) - 0.02, -0.5), 1.004)
        ax.axvline(ce[int(np.nanargmax(cva))], color=MUTED, lw=1, label="best validation")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("R² (mean over files)")
    else:
        ax.text(0.5, 0.5, "no r2_log.csv:\nthis run had no validation set", ha="center", va="center",
                color=INK2, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("(b) R² vs epoch: train and validation")
    style(ax)

    keys = [k for k, _ in LEVELS if k in summ]
    for ax, field, ylab, ttl in ((axs[1, 0], "r2", "R² per file (clipped at -1)", "(c) R² at every level"),
                                 (axs[1, 1], "peak", "|peak error| per file (%)", "(d) peak-stress error at every level")):
        for i, k in enumerate(keys):
            if field == "r2":
                v = np.clip([r["r2"] for r in sets[k]], -1, 1)
            else:
                v = np.array([abs(r["peak_err_pct"]) for r in sets[k]])
            x = i + np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=9, color=LEVEL_COLOUR[k], alpha=0.5, linewidths=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK, lw=2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([TICK[k] for k in keys], fontsize=9.5)
        ax.set_ylabel(ylab)
        ax.set_title(ttl + "  (black line = median)")
        style(ax)
        if field == "r2":
            ax.set_ylim(-1.05, 1.02)
        else:
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(bottom=0)
    fig.suptitle("Fit check  -  %s" % a.ckpt, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_check.png"), dpi=130)
    plt.close(fig)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
