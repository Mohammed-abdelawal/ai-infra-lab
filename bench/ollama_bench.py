#!/usr/bin/env python3
"""Benchmark Ollama across models and CPU thread settings.

For each model it measures the cold load time and the VRAM used. For each thread
setting it measures generation speed, prompt speed, time to first token, and how long
the kernel froze the container for going over its CPU quota.

Run it on the GPU box, inside the same container as Ollama, so the throttling numbers
belong to Ollama's own CPU quota. It uses only the Python standard library.

Example:
  python3 bench/ollama_bench.py \
      --models qwen2.5:7b-instruct-q4_K_M,qwen2.5:7b-instruct-q8_0 \
      --threads default,4,8 --runs 3
"""
import argparse
import csv
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PROMPT = (
    "Explain how a database index works, step by step, with a concrete example. "
    "Write at least 500 words."
)


def compute_metrics(resp: dict) -> dict:
    """turn one /api/generate response into the numbers we report.

    Return a dict with three keys:
      gen_tps     generated tokens per second
      prompt_tps  prompt tokens processed per second
      ttft_s      seconds until the first generated token

    Fields available in resp: eval_count, eval_duration, prompt_eval_count,
    prompt_eval_duration, load_duration, total_duration.
    Check the unit of the *_duration fields in Ollama's API docs before dividing.

    For ttft_s: with stream=false you never see the first token arrive, so build it
    from the durations. Ask yourself what has to finish before the first token exists.
    """

    return {
        "gen_tps": resp["eval_count"] / (resp["eval_duration"] / 1e9),
        "prompt_tps": resp["prompt_eval_count"] / (resp["prompt_eval_duration"] / 1e9),
        "ttft_s": (resp["load_duration"] + resp["prompt_eval_duration"]) / 1e9,
    }


# ---------------------------------------------------------------- Ollama API

def call(host: str, path: str, payload: dict | None = None, timeout: int = 900) -> dict:
    """GET when payload is None, otherwise POST it as JSON."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(host + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(host: str, model: str, threads: int | None, num_predict: int, num_ctx: int) -> dict:
    # A unique first line per request defeats Ollama's prompt cache. Without it, runs 2 and 3
    # reuse the KV cache from run 1 and the prompt speed is meaningless (the 777-of-796
    # "prompt eval cached" you saw in your first session).
    prompt = f"Request {uuid.uuid4().hex[:8]}.\n{PROMPT}"
    # Fixed context and a fixed seed, so every run does the same amount of work.
    options = {"temperature": 0, "seed": 42, "num_predict": num_predict, "num_ctx": num_ctx}
    if threads is not None:
        options["num_thread"] = threads
    return call(host, "/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": options})


def loaded_models(host: str) -> dict:
    return {m["name"]: m for m in call(host, "/api/ps").get("models", [])}


def unload(host: str, model: str, wait_s: int = 30) -> None:
    call(host, "/api/generate", {"model": model, "keep_alive": 0})
    deadline = time.time() + wait_s
    while model in loaded_models(host) and time.time() < deadline:
        time.sleep(0.5)


def cold_load_s(host: str, model: str, num_ctx: int) -> float:
    """Wall-clock time to load the model from disk into VRAM. An empty prompt only loads it."""
    unload(host, model)
    t0 = time.perf_counter()
    call(host, "/api/generate", {"model": model, "options": {"num_ctx": num_ctx}})
    return time.perf_counter() - t0


# ---------------------------------------------------------------- machine facts

def gpu() -> tuple[str | None, int | None, int | None]:
    """GPU name, MiB used and MiB total, from nvidia-smi. All None when there is no GPU."""
    try:
        line = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        name, used, total = (x.strip() for x in line.split(","))
        return name, int(used), int(total)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None, None


def cpu_quota() -> float | None:
    """CPUs this container may use (cgroup v2 cpu.max). None when unlimited or unknown."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else round(int(quota) / int(period), 2)
    except (OSError, ValueError):
        return None


def throttled_s() -> float | None:
    """Total seconds this cgroup has been frozen for going over its CPU quota (cgroup v2)."""
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            key, value = line.split()
            if key == "throttled_usec":
                return int(value) / 1e6
    except (OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------- main

def parse_threads(text: str) -> list[int | None]:
    """'default,4,8' -> [None, 4, 8]. None means: let Ollama pick."""
    return [None if t.strip() == "default" else int(t) for t in text.split(",")]


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--models", required=True, help="comma-separated Ollama tags, already pulled")
    ap.add_argument("--threads", default="default,4", help="comma-separated; 'default' lets Ollama pick")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=256, help="max tokens to generate per run")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    threads = parse_threads(args.threads)
    gpu_name, _, vram_total = gpu()
    meta = {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "gpu": gpu_name or "none (CPU only)",
        "vram_total_mib": vram_total,
        "ollama": call(args.host, "/api/version").get("version"),
        "visible_cpus": os.cpu_count(),
        "cpu_quota": cpu_quota(),
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }
    print("# " + " | ".join(f"{k}: {v}" for k, v in meta.items()), flush=True)

    rows, summary = [], []
    for model in models:
        load_s = cold_load_s(args.host, model, args.num_ctx)
        _, vram_used, _ = gpu()
        ps = loaded_models(args.host).get(model, {})
        on_gpu = f"{100 * ps['size_vram'] / ps['size']:.0f}%" if ps.get("size") else "?"
        print(f"\n{model}: cold load {load_s:.1f}s, VRAM used {vram_used} MiB, on GPU {on_gpu}", flush=True)

        for t in threads:
            # Warm-up: a new thread count can make Ollama restart the model runner.
            # This keeps that reload out of the measured runs.
            generate(args.host, model, t, 16, args.num_ctx)
            for run in range(1, args.runs + 1):
                before = throttled_s()
                resp = generate(args.host, model, t, args.num_predict, args.num_ctx)
                after = throttled_s()
                try:
                    m = compute_metrics(resp)
                except NotImplementedError:
                    print("\ncompute_metrics() is still a TODO. Write it, then run again.", file=sys.stderr)
                    return 2
                row = {
                    "model": model,
                    "threads": "default" if t is None else t,
                    "run": run,
                    "gen_tps": round(m["gen_tps"], 2),
                    "prompt_tps": round(m["prompt_tps"], 2),
                    "ttft_s": round(m["ttft_s"], 3),
                    "tokens_generated": resp.get("eval_count"),
                    "throttled_s": None if before is None else round(after - before, 2),
                }
                rows.append(row)
                print(f"  threads={row['threads']:<7} run {run}: {row['gen_tps']} tok/s gen, "
                      f"{row['prompt_tps']} tok/s prompt, TTFT {row['ttft_s']}s, "
                      f"throttled {row['throttled_s']}s", flush=True)

            mine = [r for r in rows if r["model"] == model and r["threads"] == ("default" if t is None else t)]
            summary.append({
                "model": model,
                "threads": "default" if t is None else t,
                "gen_tps": mean([r["gen_tps"] for r in mine]),
                "prompt_tps": mean([r["prompt_tps"] for r in mine]),
                "ttft_s": mean([r["ttft_s"] for r in mine]),
                "throttled_s": mean([r["throttled_s"] for r in mine]),
                "cold_load_s": round(load_s, 1),
                "vram_mib": vram_used,
                "on_gpu": on_gpu,
            })
        unload(args.host, model)  # free VRAM so the next model's numbers are its own

    print("\n| " + " | ".join(summary[0]) + " |")
    print("|" + " --- |" * len(summary[0]))
    for s in summary:
        print("| " + " | ".join(str(v) for v in s.values()) + " |")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    label = (gpu_name or "cpu").replace(" ", "-")
    path = out / f"{stamp}-{label}.csv"
    with path.open("w", newline="") as f:
        f.write("# " + json.dumps(meta) + "\n")
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw runs saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
