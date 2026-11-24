#!/usr/bin/env python3
"""Head-to-head interleaved benchmark: Clean-room m4lm vs Stock Apple mlx_lm.

Measures identical greedy prompts across open prose, greenfield code, and copy/refactor
on Qwen3.5-0.8B-4bit, Qwen3.5-2B-4bit, and Qwen3.8-27B-MLX-4bit (+ DFlash2 on m4lm).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

M4LM_BIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "zig-out", "bin", "m4lm")
)


PROMPTS = {
    "open_prose": (
        "Explain how a GPU tensor core multiplies two matrices in clear, concise technical terms.",
        128,
    ),
    "greenfield_code": (
        "Write a clean, idiomatic TypeScript ring buffer class with push, pop, peek, length, and iterator support.",
        160,
    ),
    "copy_passage": (
        "Repeat the following technical passage word for word without changing a single word or punctuation mark:\n\n"
        "Modern Apple Silicon GPUs execute matrix multiplications on dedicated hardware tensor units arranged in sixteen-row "
        "blocks across forty compute cores. When a language model decodes one token at a time, fifteen of those sixteen "
        "hardware rows remain completely idle while the entire weight matrix streams from unified memory. By verifying "
        "sixteen to sixty-four candidate tokens simultaneously in a single forward pass, a bare Metal engine converts wasted "
        "hardware rows directly into lossless token throughput without altering a single output bit.",
        110,
    ),
}


def find_snap(pat: str) -> str:
    base = os.path.expanduser("~/.cache/huggingface/hub")
    snaps = sorted(glob.glob(f"{base}/{pat}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(pat)
    return snaps[-1]


def run_mlx_lm(model_dir: str, cases: list[str]) -> dict:
    payload = {k: PROMPTS[k] for k in cases}
    code = f"""
import json, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

model, tok = load({model_dir!r})
sampler = make_sampler(temp=0.0)
cases = {payload!r}

# Warmup
w_prompt = tok.apply_chat_template([{{"role":"user","content":"Hi"}}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
for _, _ in zip(generate_step(mx.array(tok.encode(w_prompt, add_special_tokens=False)), model, sampler=sampler), range(8)):
    pass
mx.eval()

out = {{}}
for name, (text, max_toks) in cases.items():
    rendered = tok.apply_chat_template([{{"role":"user","content":text}}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(rendered, add_special_tokens=False)
    t0 = time.perf_counter()
    gen = generate_step(mx.array(ids), model, sampler=sampler)
    first_tok, first_lp = next(gen)
    mx.eval(first_lp)
    pp_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    n = 1
    for t, lp in gen:
        mx.eval(lp)
        ti = int(t)
        if ti in (248044, 248046):
            break
        n += 1
        if n >= max_toks:
            break
    tg_ms = (time.perf_counter() - t1) * 1000.0
    out[name] = {{"toks": n, "pp_ms": round(pp_ms, 2), "tg_ms": round(tg_ms, 2), "tps": round((n - 1) / (tg_ms / 1e3), 1) if n > 1 else 0.0}}
print("JSON:" + json.dumps(out))
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        if line.startswith("JSON:"):
            import json
            return json.loads(line[5:])
    raise RuntimeError(res.stdout + res.stderr)


def run_m4lm_case(model_dir: str, draft_dir: str | None, prompt: str, max_toks: int) -> dict:
    cmd = [M4LM_BIN, "generate", model_dir, "--prompt", prompt, "--max-tokens", str(max_toks)]
    if draft_dir:
        cmd += ["--draft", draft_dir]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    m = re.search(r"Gen:\s+(\d+)\s+tok\s+\(([\d.]+)\s+tok/s,\s+([\d.]+)\s+tok/rnd", res.stderr)
    if not m:
        raise RuntimeError(f"Could not parse m4lm stderr:\n{res.stderr}")
    return {"toks": int(m.group(1)), "tps": float(m.group(2)), "tpr": float(m.group(3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["0.8b", "2b", "27b"], default="27b")
    args = ap.parse_args()

    pats = {
        "0.8b": ("models--mlx-community--Qwen3.5-0.8B-4bit", None),
        "2b": ("models--mlx-community--Qwen3.5-2B-4bit", None),
        "27b": ("models--TensorFold--Qwen3.8-27B-MLX-4bit", "models--z-lab--Qwen3.8-27B-DFlash2"),
    }
    mpat, dpat = pats[args.model]
    mdir = find_snap(mpat)
    ddir = find_snap(dpat) if dpat else None

    cases = ["open_prose", "greenfield_code", "copy_passage"]
    print(f"=== Head-to-Head: m4lm vs Stock Apple mlx_lm ({args.model}) ===")
    mlx_res = run_mlx_lm(mdir, cases)
    for c in cases:
        text, max_toks = PROMPTS[c]
        m4_res = run_m4lm_case(mdir, ddir, text, max_toks)
        mx_tps = mlx_res[c]["tps"]
        ratio = m4_res["tps"] / mx_tps if mx_tps > 0 else 0.0
        print(f"  {c:<16} | mlx_lm: {mx_tps:6.1f} tok/s | m4lm: {m4_res['tps']:7.1f} tok/s ({m4_res['tpr']:.2f} tok/rnd) | Speedup: {ratio:5.2f}x")


if __name__ == "__main__":
    main()
