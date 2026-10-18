#!/bin/sh
# Train one FTPO variant on two B70s and merge it into the MXFP4 checkpoint (round 2 variant B by default).
# Run it where both cards are visible and nothing else uses them (stop every vLLM server first):
#   ONEAPI_DEVICE_SELECTOR=level_zero:0,1 sh run_train.sh
# Variables: PY (python with torch-xpu, transformers, peft, trl), CONFIG, BASE (stock MXFP4 dir), ADAPTER (the
# config's output_dir), OUT (merged model dir, must not exist). Output goes to stdout and train.log.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
CONFIG=${CONFIG:-configs/xpu_train_r2b_noheads.yaml}
BASE=${BASE:-/models/gpt-oss-20b}
ADAPTER=${ADAPTER:-runs/ftpo_lora_r2b}
OUT=${OUT:-/models/gpt-oss-20b-antidoom-r2b}
{
  "$PY" -c "import torch; n=torch.xpu.device_count(); print('XPUs', n, [torch.xpu.get_device_name(i) for i in range(n)], flush=True)"
  # data + config check first, so a bad file fails before any card is touched
  "$PY" xpu_train.py -c "$CONFIG" --check || { echo "CHECK FAILED $CONFIG"; exit 1; }
  echo "=== TRAIN $CONFIG ==="
  if "$PY" xpu_train.py -c "$CONFIG"; then
    echo "TRAIN DONE"
    if "$PY" merge_into_mxfp4.py --base "$BASE" --adapter "$ADAPTER" --out "$OUT"; then
      echo "MERGE DONE -> $OUT"
    else
      echo "MERGE FAILED"
    fi
  else
    echo "TRAIN FAILED"
  fi
} 2>&1 | tee train.log
