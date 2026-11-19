#!/usr/bin/env bash
# On-start script for milestone 1: fresh Vast.ai instance -> Ollama serving one model.
# Runs unattended as root inside a Docker container (no systemd), on every instance start,
# so it must be safe to run twice.
set -euo pipefail

LOG=/var/log/provision.log
exec > >(tee -a "$LOG") 2>&1
echo "[provision] start $(date -Is)"


# choose a 7-8B model tag at 4-bit quantization.
MODEL="${MODEL:-qwen2.5:7b-instruct-q4_K_M}"
export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

# install curl, ca-certificates and zstd
apt-get update
apt-get install -y curl ca-certificates zstd python3


# install Ollama, but only if it is not installed already.
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh

# start the Ollama server in the background.
if curl -sf http://127.0.0.1:11434/  > /dev/null; then
  echo "[provision] ollama already running, skipping start"
else
  echo "[provision] starting ollama server"
  nohup ollama serve >> /var/log/ollama.log 2>&1 &
fi

# wait until the server answers before going on
tries=0
until curl -sf http://127.0.0.1:11434/ > /dev/null; do
  tries=$((tries + 1))
  if [ "$tries" -ge 60 ]; then
    echo "[provision] ERROR: ollama not answering after 60s, see /var/log/ollama.log" >&2
    exit 1
  fi
  sleep 1
done



# pull $MODEL.
ollama pull "$MODEL"
echo "[provision] done $(date -Is)"
