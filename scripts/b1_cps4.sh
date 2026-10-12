#!/usr/bin/env bash
# Run B1 on the CPS4 SI dataset - every step is one short command (Ubuntu, from the repo folder):
#
#   bash scripts/b1_cps4.sh build      compile the training graphs (once, after git pull)
#   bash scripts/b1_cps4.sh quick      20-epoch timing run -> results_si_cps4_quick/ (delete it afterwards)
#   bash scripts/b1_cps4.sh train      the full 10000-epoch run in the background (safe to close SSH)
#   bash scripts/b1_cps4.sh status     is it running? last lines of both logs
#   bash scripts/b1_cps4.sh curves     R2 vs epoch plot, train + validation (works while training runs)
#   bash scripts/b1_cps4.sh resume     continue a stopped run, in the background
#   bash scripts/b1_cps4.sh eval       the three tests: unseen loads / interpolation / extrapolation, 8 edge draws
#   bash scripts/b1_cps4.sh fitcheck   over/under-fitting verdict + plots (after eval)
#
# Activate the environment first:   source ~/mgn/.venv/bin/activate

set -e
cd "$(dirname "$0")/.."

D=data/si_cps4
DATA=$D/dataset_si_cps4_aug20.pt
RES=results_si_cps4_b1
CKPT=$RES/mgn_best_val.pt

if [ -z "$1" ] || [ "$1" = "help" ]; then sed -n '2,13p' "$0"; exit 0; fi
if ! python -c "import torch" 2>/dev/null; then
    echo "python cannot import torch - activate the environment first:  source ~/mgn/.venv/bin/activate"
    exit 1
fi
PAT='^[^ ]*python[^ ]* (-u )?scripts/train_b1_val\.py'          # a python process running the trainer, nothing else
running() { pgrep -f "$PAT" > /dev/null; }

case "$1" in
  build)
    python -m mgn.dataset --raw $D/train --aug-perc 0.2 --out $DATA ;;
  quick)
    python scripts/train_b1_val.py --data $DATA --val-dir $D/val --epochs 20 --eval-every 10 --results results_si_cps4_quick ;;
  train|resume)
    if running; then echo "A training run is already going (bash scripts/b1_cps4.sh status). Not starting a second one."; exit 1; fi
    mkdir -p $RES
    EXTRA=""; [ "$1" = "resume" ] && EXTRA="--resume"
    nohup python -u scripts/train_b1_val.py --data $DATA --val-dir $D/val --results $RES $EXTRA >> $RES/train.out 2>&1 &
    sleep 5
    if running; then echo "started (PID $!). Output: $RES/train.out   Check with: bash scripts/b1_cps4.sh status";
    else echo "it stopped straight away - the error:"; tail -n 20 $RES/train.out; exit 1; fi ;;
  status)
    if running; then echo "RUNNING:"; pgrep -af "$PAT"; else echo "not running"; fi
    echo "--- last epochs: epoch, loss, seconds"
    if [ -f $RES/loss_log.csv ]; then tail -n 3 $RES/loss_log.csv; else echo "none yet"; fi
    echo "--- last R2 check: epoch, train R2, validation R2"
    if [ -f $RES/r2_log.csv ]; then tail -n 2 $RES/r2_log.csv | cut -d, -f1,3,8; else echo "none yet"; fi ;;
  curves)
    python scripts/check_fit.py --curves-only --ckpt $CKPT
    echo "plot: $RES/fit_check/r2_vs_epoch.png" ;;
  eval)
    python scripts/eval_own_unseen.py --ckpt $CKPT --train-data $DATA --loads-dir $D/unseen_loads \
        --interp-dir $D/unseen_interp --extrap-dir $D/unseen_extrap --manifest $D/si_manifest.csv --units MPa --draws 8 ;;
  fitcheck)
    python scripts/check_fit.py --ckpt $CKPT --train-data $DATA --val-dir $D/val --draws 8 ;;
  *)
    echo "unknown step '$1'"; sed -n '2,13p' "$0"; exit 1 ;;
esac
