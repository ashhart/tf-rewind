#!/usr/bin/env python3
"""Interleaved head-to-head benchmark comparing clean-room m4lm vs Apple mlx_lm.

Runs both engines back-to-back on the exact same model checkpoint and prompt so shared
machine load / thermal state affects both fairly.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

M4LM_BIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "zig-out", "bin", "m4lm")
)


def find_snapshot(pattern: str) -> str:
    base = os.path.expanduser("~/.cache/huggingface/hub")
    snaps = sorted(glob.glob(f"{base}/{pattern}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot found for {pattern}")
    return snaps[-1]


def bench_m4lm(model_dir: str, prompt: str, max_tokens: int, warmup: int = 16) -> dict:
    cmd = [
        M4LM_BIN,
        "bench",
        model_dir,
        "--prompt-len",
        "128",
        "--tokens",
        str(max_tokens),
        "--warmup",
        str(warmup),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = res.stderr + res.stdout
    return {"raw": out.strip()}


def bench_mlx_lm(model_dir: str, max_tokens: int) -> dict:
    code = f"""
import time, mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
model, tok = load({model_dir!r})
prompt = mx.array([(i * 131 + 42) % 30000 for i in range(128)])
# Warmup
for _, _ in zip(generate_step(prompt, model), range(16)):
    pass
mx.eval()
# Timed run
t0 = time.perf_counter()
gen = generate_step(prompt, model)
first_tok, _ = next(gen)
mx.eval(first_tok)
pp_ms = (time.perf_counter() - t0) * 1000.0
t1 = time.perf_counter()
n = 0
for t, _ in gen:
    mx.eval(t)
    n += 1
    if n >= {max_tokens}:
        break
tg_ms = (time.perf_counter() - t1) * 1000.0
print(f"Prefill: 128 tok in {{pp_ms:.2f}} ms ({{128.0 / (pp_ms / 1e3):.1f}} tok/s)")
print(f"Decode : {{n}} tok in {{tg_ms:.2f}} ms ({{n / (tg_ms / 1e3):.1f}} tok/s, {{tg_ms / n:.2f}} ms/tok)")
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return {"raw": (res.stdout + res.stderr).strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["0.8b", "2b", "27b"], default="0.8b")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    pats = {
        "0.8b": "models--mlx-community--Qwen3.5-0.8B-4bit",
        "2b": "models--mlx-community--Qwen3.5-2B-4bit",
        "27b": "models--TensorFold--Qwen3.8-27B-MLX-4bit",
    }
    snap = find_snapshot(pats[args.model])
    print(f"=== Interleaved Benchmark: {args.model} ({snap}) ===")
    for r in range(args.rounds):
        print(f"\n--- Round {r + 1}: m4lm (Metal 4 + Zig) ---")
        print(bench_m4lm(snap, "", args.tokens)["raw"])
        print(f"\n--- Round {r + 1}: Apple mlx_lm (stock) ---")
        print(bench_mlx_lm(snap, args.tokens)["raw"])


if __name__ == "__main__":
    main()
