# Progress

Shown at the start of every session. A box is ticked only when the work is finished and measured.

Budget: about $30–50 for the project. Spent so far: $0.15 (Vast.ai credit $10.00 → $9.85 on 2026-11-20)

## Setup

- [x] Repo created with layout, `.gitignore` and README
- [x] Level check answered
- [x] Roadmap agreed (8 milestones; locked in on 2026-11-17 after no objection, still open to change)
- [x] Vast.ai account funded ($10 credit loaded), SSH key added
- [x] First offer chosen: on-demand vs interruptible understood (on-demand for interactive sessions; bandwidth price matters because we re-download every session)
- [x] First on-start script in `provision/` (local Docker test, then a real RTX 3060 run on 2026-11-19)

## Milestones

- [x] 1. Run a small model with llama.cpp or Ollama on a 3060; measure tokens/sec and VRAM across quantization levels (write-up in `milestones/1-first-model/`; pushed 2026-11-20; LinkedIn draft kept for a later combined post)
- [ ] 2. Serve an LLM with vLLM; call it from Django with streaming; measure throughput under concurrent requests
- [ ] 3. Tool-calling chat assistant over a mock API on the self-hosted model; eval set; measure answer and tool-call accuracy
- [ ] 4. Speech-to-text and text-to-speech as HTTP services
- [ ] 5. Stream both over WebSocket; measure latency per stage
- [ ] 6. Realtime voice agent with Pipecat or LiveKit Agents (interruptions, turn-taking)
- [ ] 7. QLoRA fine-tune of the assistant on tool-call examples: debug on 7–8B, compare on the eval set, then decide on scaling up
- [ ] 8. Production packaging: Docker, health checks, logs, cost per job

## Concepts covered

- VRAM for weights = parameters × bits per weight / 8 (2026-11-17)
- KV cache: the per-token memory that grows with context length (2026-11-17)
- Time-to-first-token vs tokens/sec, and which workload cares about which (2026-11-17)
- KV cache size = 2 × layers × KV heads × head size × bytes per value, per token; grouped-query attention shrinks it (2026-11-19)
- Generation is memory-bound: tokens/sec ceiling ≈ memory bandwidth ÷ model size; prompt reading and batching are compute-bound (2026-11-19)
- Cheapest per hour is not cheapest per token: filter hosts by memory bandwidth, not just price (2026-11-19)
- Cold vs warm model load, and prompt caching across chat turns (`prompt eval cached`) (2026-11-19)
- More VRAM buys capacity (bigger models, more concurrent users), not per-reply speed (2026-11-19)
- Batching: one read of the weights serves every user in the batch; each extra user costs its own KV cache (2026-11-19)
- CPU quota vs visible CPUs: too many threads in a CPU-limited container get throttled and starve the GPU (2026-11-19)
- Batching puts the idle compute to work: weight reads are shared, so total throughput grows until compute saturates; KV cache reads are per user and stay memory-bound (2026-11-19)

Each milestone is done when it has: a write-up in `milestones/`, measured numbers in the README, a demo idea, a LinkedIn draft, and new lines under "Skills I can now prove".

## Session log

| Date | What we did | GPU | Cost |
| --- | --- | --- | --- |
| 2026-11-17 | Repo set up, level check answered, roadmap locked, VRAM math, Vast.ai account and SSH key ready, provisioning script tested in local Docker | none | $0.00 |
| 2026-11-19 | First real instance: provisioning worked; KV cache math; slow generation traced to CPU thread oversubscription (72 visible CPUs, 8.64 CPU quota); three models loaded side by side; instance destroyed | RTX 3060, $0.064/hr | $0.07 |
| 2026-11-20 | Milestone 1 benchmark: Q4 vs Q8, threads default/4/8, results CSV copied home; instance destroyed | RTX 3060, $0.068/hr | $0.09 |

## Raw measurements (milestone 1)

RTX 3060 12GB, Vast.ai host in the US, Xeon E5-2686 v4, container quota 8.64 CPUs. Model qwen2.5:7b-instruct-q4_K_M, Ollama 0.40.16, context 4096, all 29/29 layers on GPU. Date 2026-11-19.

| Setting | Prompt tok/s | Generation tok/s | Note |
| --- | --- | --- | --- |
| Default threads (36), first run | 0.93 | 3.54 | Learner's first `ollama run --verbose` |
| Default threads (36), warm | 582 | 39.8 | cgroup throttled ~54 s during one 128-token request |
| `num_thread` 4 | 344 | 65.1 | Zero throttling |

VRAM in `nvidia-smi` with the model loaded: 4,673 MiB. Ollama's breakdown: weights 4,168 MiB + KV cache 224 MiB + compute buffer 143 MiB. KV cache at 4,096 tokens = 224 MiB, matching 2 × 28 layers × 4 KV heads × 128 head_dim × 2 bytes = 56 KiB per token.

### Second box, 2026-11-19 22:26 UTC (benchmark script)

RTX 3060 12GB, Vast.ai host in South Korea, 20 visible CPUs, CPU quota 19.2, Ollama 0.40.16, $0.068/hr. Context 4096, 256 generated tokens, 3 runs per setting, unique prompt per run (no prompt cache). Script: `bench/ollama_bench.py`.

| Model | Threads | Gen tok/s | Prompt tok/s | TTFT (server) | Throttled | Cold load | VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:7b-instruct-q4_K_M | default / 4 / 8 | 63.7 / 63.9 / 63.5 | 2,010 / 2,028 / 2,052 | 0.03 s | 0 s | 21.9 s | 4,673 MiB |
| qwen2.5:7b-instruct-q8_0 | default / 4 / 8 | 42.0 / 41.8 / 41.8 | 1,676 / 1,711 / 1,652 | 0.04 s | 0 s | 15.1 s | 7,669 MiB |

Thread count made no difference here: no throttling with a 19.2-CPU quota. On the first box (8.64-CPU quota) default threads cost 40%. A fixed low thread count is free insurance.

## Predictions vs reality

Written down before running. The gap is the lesson.

| Date | What | Predicted | Actual | Why the gap |
| --- | --- | --- | --- | --- |
| 2026-11-17 | RTX 3060 price per hour on Vast.ai | $0.03 | $0.042–0.061 | Home page shows a "from" price; interruptible bids are cheaper than on-demand; bandwidth and storage are billed on top |
| 2026-11-17 | Download of qwen2.5:7b-instruct-q4_K_M on an ~840 Mbps host | 3.8 s | not measured yet (best case ~45 s) | Mbps is megabits (÷8); the real file is ~4.7 GB, not the 3.26 GiB weights estimate |
| 2026-11-17 | Second run of the provisioning script (local Docker test) | skips install and pull | 2.8 s; install skipped, pull ran but downloaded nothing | `ollama pull` re-checks the registry; only missing layers are downloaded |
| 2026-11-17 | First run of the provisioning script (local Docker test, home connection) | no prediction made | 7m30s | Mostly downloads over a home link: apt packages, the Ollama bundle with CUDA libs, a 397 MB model |
| 2026-11-19 | VRAM with qwen2.5:7b-instruct-q4_K_M loaded, 4,096 context | ~5.4 GB (4.7 weights + 0.22 KV + ~0.5 overhead); first guess was 11.7 GB assuming 128k context | 4.9 GB peak (RTX 3060, 2026-11-19) | First guess assumed the model's max context; Ollama allocates a much smaller default, and Qwen2.5-7B's native max is 32k |
| 2026-11-17 | Generation speed, same model, 3060 at ~317 GB/s | 67.5 tok/s | 39.96 tok/s with default threads; 65.1 tok/s (96% of ceiling) with `num_thread` 4 (2026-11-19) | Not the GPU: Ollama started 36 CPU threads on a container limited to 8.64 CPUs, the kernel throttled it, and the GPU waited. Fixed thread count reached the bandwidth ceiling |
| 2026-11-19 | Concurrent 4k-context users that fit on an RTX 3060 with qwen2.5 7B Q4 loaded (KV cache space) | ~33 (7.5 GB free ÷ 224 MiB per user) | pending (milestone 2, vLLM) | |
| 2026-11-19 | Speed per user with 33 users at full 4k context on an RTX 3060 (7B Q4) | ~26 tok/s each, ~865 tok/s total (ceiling: 317 GB/s ÷ (4.7 GB weights + 7.4 GB KV)); first instinct was 65 ÷ 33 | pending (milestone 2, vLLM) | |
| 2026-11-20 | Generation speed, qwen2.5:7b-instruct-q8_0 (8.1 GB) on an RTX 3060 | ~37 tok/s (317 GB/s ÷ 8.1 GB × 96%) | 41.9 tok/s (RTX 3060, South Korea host, 2026-11-19) | Above the file-size ceiling (39.1): the ~0.6 GB token-embedding table is not read per token (llama.cpp keeps it off the GPU and looks up one row). Real bytes per token ≈ 7.5 GB (matches VRAM), ceiling 42.2, so 99%. Q4 reached ~88% of its real ceiling (4.37 GB → 72.5): unpacking 4-bit K-quants costs more compute |

## Quiz log

| Date | Topics | Result |
| --- | --- | --- |
| 2026-11-19 | Serving 20 users on one GPU: queueing, TTFT, batching, KV cache per user | Half right: queueing and TTFT correct; assumed the model must be loaded once per user. Corrected the same day |
