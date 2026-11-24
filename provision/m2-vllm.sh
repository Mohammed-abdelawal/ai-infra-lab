#!/usr/bin/env bash
# On-start script for milestone 2: vLLM serving one model, plus this repo cloned to /root/lab.
# Meant for the vllm/vllm-openai:v0.33.0 image (CUDA 13.0: host driver must support CUDA >= 13.0).
# Runs unattended as root on every instance start, so it must be safe to run twice.
set -euo pipefail

LOG=/var/log/provision.log
exec > >(tee -a "$LOG") 2>&1
echo "[provision] start $(date -Is)"

export DEBIAN_FRONTEND=noninteractive
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
MAX_LEN="${MAX_LEN:-4096}"            # longest conversation (prompt + reply) one request may have
GPU_UTIL="${GPU_UTIL:-0.90}"          # share of VRAM vLLM may take: weights + KV cache + overhead
REPO="${REPO:-https://github.com/Mohammed-abdelawal/ai-infra-lab}"
LAB=/root/lab

# Tools the image may not have.
missing=""
for tool in git curl; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done
if [ -n "$missing" ]; then
  apt-get update -qq
  apt-get install -y -qq $missing ca-certificates
fi

# The repo: clone on the first start, fast-forward on later ones. Read-only, no token on this box.
if [ -d "$LAB/.git" ]; then
  git -C "$LAB" pull --ff-only
else
  git clone --depth 1 "$REPO" "$LAB"
fi

# vLLM listens on localhost only: it has no authentication. Reach it over SSH, never an open port.
if curl -sf http://127.0.0.1:8000/health > /dev/null; then
  echo "[provision] vLLM already running, skipping start"
else
  echo "[provision] starting vLLM: $MODEL, max-model-len $MAX_LEN, gpu-memory-utilization $GPU_UTIL"
  nohup vllm serve "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    >> /var/log/vllm.log 2>&1 &
fi

# First start downloads the model (~5.6 GB) and captures CUDA graphs, so allow 15 minutes.
tries=0
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
  tries=$((tries + 1))
  if [ "$tries" -ge 900 ]; then
    echo "[provision] ERROR: vLLM not healthy after 15 min, see /var/log/vllm.log" >&2
    exit 1
  fi
  sleep 1
done

# vLLM's own capacity estimate: how many full-length requests fit in the KV cache at once.
grep -E "KV cache|Maximum concurrency" /var/log/vllm.log | tail -n 3 || true
echo "[provision] done $(date -Is)"
