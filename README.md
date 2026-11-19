# AI Infra Lab

Hands-on work running, serving and fine-tuning open models on rented GPUs, and wiring them into a backend. Every number in this repo was measured on a real instance, not copied from a model card.

## Layout

Folders are created when the first file lands in them.

| Folder | What goes there |
| --- | --- |
| `provision/` | On-start scripts that turn a fresh Vast.ai instance into a working box, so instances can be destroyed after every session |
| `bench/` | Measurement scripts and their results: tokens/sec, VRAM, latency per stage |
| `agents/` | Agent experiments and their tests |
| `behaviour/` | Prompts, eval cases and notes on how each model actually behaves |
| `milestones/` | One folder per milestone: write-up, measured numbers, demo notes |

Progress and spend are tracked in [PROGRESS.md](PROGRESS.md).

## Roadmap

It gets revised as the work shows what is realistic.

1. Run a small model with llama.cpp or Ollama on an RTX 3060; measure tokens/sec and VRAM across quantization levels.
2. Serve an LLM with vLLM and call it from Django with streaming; measure throughput with many requests at once.
3. Build a tool-calling chat assistant over a mock API on the self-hosted model; build an eval set and measure answer and tool-call accuracy.
4. Run speech-to-text and text-to-speech as HTTP services.
5. Stream them over WebSocket; measure the latency of each stage.
6. Build a realtime voice agent with Pipecat or LiveKit Agents (interruptions, turn-taking).
7. Fine-tune the assistant with QLoRA on tool-call examples: debug on a 7–8B model, compare against the eval set, then decide whether scaling up is worth the cost.
8. Package for production: Docker, health checks, logs, cost per job.

## Measured numbers

| Milestone | Model | GPU | VRAM | Speed / latency | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 (2026-11-19) | Qwen2.5-7B-Instruct, Q4_K_M (Ollama) | RTX 3060 12GB, Vast.ai | 4,673 MiB | 63.8 tok/s generation, ~2,000 tok/s prompt, cold load 21.9 s | $0.068/hr |
| 1 (2026-11-19) | Qwen2.5-7B-Instruct, Q8_0 (Ollama) | RTX 3060 12GB, Vast.ai | 7,669 MiB | 41.9 tok/s generation, ~1,700 tok/s prompt, cold load 15.1 s | $0.068/hr |

## Skills I can now prove

Each line is added only after the milestone that backs it is finished.

- Estimate an LLM's VRAM from parameters, quantization and context length (weights + KV cache); predicted within 10% on a real GPU ([milestone 1](milestones/1-first-model/))
- Predict single-user generation speed from memory bandwidth; measured 63.8 tok/s (Q4) and 41.9 tok/s (Q8) on an RTX 3060, at 88–99% of the computed ceiling
- Provision disposable GPU instances with idempotent on-start scripts; a full benchmark session costs under $0.10
- Diagnose CPU-quota throttling in containers that cut GPU inference speed by 40%

## Rules of the lab

- Cheapest GPU that does the job: RTX 3060 (12GB) for learning, RTX 3090 (24GB) for real work, A6000 (48GB) only when a task needs it.
- Instances are disposable. Setup lives in `provision/`; nothing important lives only on a rented disk.
- No private data on marketplace hosts.
- No secrets in git. Tokens go in `.env`, which is ignored.
