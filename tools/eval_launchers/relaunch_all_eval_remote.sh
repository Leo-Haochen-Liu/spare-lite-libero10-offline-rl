#!/bin/bash
set -euo pipefail
mkdir -p /root/autodl-tmp/SpaRe-lite/results/eval_runtime/archive
stamp=$(date +%Y%m%d_%H%M%S)
for f in /root/autodl-tmp/SpaRe-lite/results/sparelite-eval-*.eval.log; do
  [ -f "$f" ] || continue
  cp "$f" "/root/autodl-tmp/SpaRe-lite/results/eval_runtime/archive/$(basename "$f").$stamp"
  : > "$f"
done
pkill -f 'sparelite-smoke-libero10-r1|main_ppo|run_openvla_oft_libero_eval_min|sparelite-eval-libero' || true
sleep 5
nohup bash /root/autodl-tmp/SpaRe-lite/tools/eval_launchers/gpu0_eval_libero10_r1.sh >/root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu0.current.out 2>&1 & echo $! > /root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu0.current.pid
nohup bash /root/autodl-tmp/SpaRe-lite/tools/eval_launchers/gpu1_eval_libero10_r1r2.sh >/root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu1.current.out 2>&1 & echo $! > /root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu1.current.pid
nohup bash /root/autodl-tmp/SpaRe-lite/tools/eval_launchers/gpu2_eval_libero_spatial_r1.sh >/root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu2.current.out 2>&1 & echo $! > /root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu2.current.pid
nohup bash /root/autodl-tmp/SpaRe-lite/tools/eval_launchers/gpu3_eval_libero_spatial_r1r2.sh >/root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu3.current.out 2>&1 & echo $! > /root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu3.current.pid
sleep 3
cat /root/autodl-tmp/SpaRe-lite/results/eval_runtime/gpu*.current.pid
