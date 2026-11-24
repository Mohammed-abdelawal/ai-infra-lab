#!/usr/bin/env python3
"""Load-test an OpenAI-compatible streaming endpoint (vLLM) with N users at once.

For each concurrency level it starts N requests at the same instant and records, for
every request, when it was sent, when its first token arrived, when its last token
arrived and how many tokens came back. Then it summarizes the wave: time to first token,
the speed each user sees, and total throughput. Standard library only.

  python3 bench/load_test.py --url http://127.0.0.1:8000/v1 \
      --model Qwen/Qwen2.5-7B-Instruct-AWQ --users 1,4,8,16,32 --max-tokens 256

Run it on the GPU box, next to the server, so network latency doesn't blur the numbers.
"""
import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPT = "Explain how a database index works, step by step, with a concrete example."


@dataclass
class Result:
    t_start: float          # request sent (time.perf_counter() seconds)
    t_first: float | None   # first content token arrived
    t_end: float            # stream finished or failed
    tokens: int             # tokens generated
    error: str | None = None


def summarize(results: list[Result]) -> dict:
    """turn one wave of simultaneous requests into the numbers we report.

    Return a dict with these keys:
      ttft_p50_s, ttft_p95_s   time to first token: median and 95th percentile
      user_tps_p50             generation speed one user sees (median across users)
      total_tps                tokens per second across all users together
      errors                   how many requests failed

    Each Result has t_start, t_first, t_end (seconds from time.perf_counter()),
    tokens and error.

    Hints:
      - A user's generation speed should not include their wait for the first token.
      - total_tps describes the whole wave. Which start and which end bound the wave?
      - statistics.quantiles(data, n=100) returns 99 cut points; index 94 is p95.
        It needs at least two values, so handle the one-user case.
      - Leave failed requests (error is not None) out of the timing numbers.
    """
    ttft_p50_s = ttft_p95_s = user_tps_p50 = total_tps = errors = None
    
    ttft_p50_s = statistics.median([r.t_first - r.t_start for r in results if r.error is None])
    ttft_p95_s = statistics.quantiles([r.t_first - r.t_start for r in results if r.error is None], n=100)[94] if len([r.t_first - r.t_start for r in results if r.error is None]) > 1 else None 
    user_tps_p50 = statistics.median([(r.tokens / (r.t_end - r.t_first)) for r in results if r.error is None])
    total_tps = sum(r.tokens for r in results if r.error is None) / (max(r.t_end for r in results if r.error is None) - min(r.t_start for r in results if r.error is None))
    errors = sum(1 for r in results if r.error is not None)
    return {
        "ttft_p50_s": ttft_p50_s,
        "ttft_p95_s": ttft_p95_s,
        "user_tps_p50": user_tps_p50,
        "total_tps": total_tps,
        "errors": errors
    }


# ---------------------------------------------------------------- one streaming request

def one_request(url: str, model: str, max_tokens: int, timeout: int) -> Result:
    # A unique first line per request defeats vLLM's prefix cache, so every request pays
    # for reading its own prompt. That's the worst case. A real chat with a shared system
    # prompt would do better.
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"Request {uuid.uuid4().hex[:8]}.\n{PROMPT}"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,          # vLLM extension: always generate max_tokens, so waves compare fairly
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_start = time.perf_counter()
    t_first, chunks, usage_tokens = None, 0, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:                       # server-sent events: one "data: {...}" per line
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                for choice in event.get("choices", []):
                    if choice.get("delta", {}).get("content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
        return Result(t_start, t_first, time.perf_counter(), usage_tokens or chunks)
    except Exception as exc:  # a failed request is data too: count it, don't crash the wave
        return Result(t_start, t_first, time.perf_counter(), chunks, f"{type(exc).__name__}: {exc}")


def run_wave(users: int, url: str, model: str, max_tokens: int, timeout: int) -> list[Result]:
    """Start `users` requests at the same instant, wait for all of them."""
    gate = threading.Barrier(users)

    def user() -> Result:
        gate.wait()                                # everyone leaves the gate together
        return one_request(url, model, max_tokens, timeout)

    with ThreadPoolExecutor(max_workers=users) as pool:
        return list(pool.map(lambda _: user(), range(users)))


# ---------------------------------------------------------------- main

def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return out.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="model name as the server reports it in /v1/models")
    ap.add_argument("--users", default="1,4,8,16,32", help="comma-separated concurrency levels")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    levels = [int(u) for u in args.users.split(",")]
    meta = {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "gpu": gpu_name() or "none",
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
    }
    print("# " + " | ".join(f"{k}: {v}" for k, v in meta.items()), flush=True)

    one_request(args.url, args.model, 8, args.timeout)   # warm-up: first request pays one-time costs

    rows, raw = [], []
    for users in levels:
        results = run_wave(users, args.url, args.model, args.max_tokens, args.timeout)
        try:
            summary = summarize(results)
        except NotImplementedError:
            print("\nsummarize() is still a TODO. Write it, then run again.", file=sys.stderr)
            return 2
        row = {"users": users, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}}
        rows.append(row)
        raw += [{"users": users, **asdict(r)} for r in results]
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    print("\n| " + " | ".join(rows[0]) + " |")
    print("|" + " --- |" * len(rows[0]))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row.values()) + " |")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    path = out / f"{stamp}-load-{(meta['gpu']).replace(' ', '-')}.csv"
    with path.open("w", newline="") as f:
        f.write("# " + json.dumps(meta) + "\n")
        writer = csv.DictWriter(f, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    print(f"\nRaw requests saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
