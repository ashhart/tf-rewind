#!/usr/bin/env python3
"""Bidirectional ReplaySSM + TF-Rewind: Unplanned Prefix-Cache Splitting Benchmark.

Demonstrates TF-Rewind as a drop-in bidirectional add-on to NVIDIA/Dao Lab's ReplaySSM
for multi-tenant RadixAttention serving where incoming requests diverge at UNPLANNED
mid-block positions P in [a_k, a_{k+1}] (e.g., coding-agent file edits, RAG chunk swaps).

Policy comparison across random unplanned split positions inside a cached block [0 .. B]:
  1. Forward-Only ReplaySSM (checkpoints every B tokens):
     Must always replay forward from a_k -> P (avg B/2 steps, worst-case B steps, sequential).
  2. Bidirectional ReplaySSM + TF-Rewind (Theorem 5 Single-GEMM + Certified Fast Heads):
     - If P <= B/2: forward ReplaySSM from a_k (bit-exact, <= B/2 steps).
     - If P >  B/2: backward TF-Rewind (Cert) from a_{k+1} via ONE Single-GEMM for slow heads
       + certified zero-start / anchor replay for fast & non-stationary heads (<= B/2 steps!).
     Result:
       - At EQUAL checkpoint spacing B: cuts max/avg distance 2.0x AND runs significantly faster
         because slow heads (60%+) execute as ONE parallel GEMM instead of a sequential scan!
       - At EQUAL worst-case distance: cuts ReplaySSM checkpoint RAM in HALF (2.0x fewer ckpts)!

Run on any CPU or CUDA GPU:
    python3 examples/bidirectional_radix_bench.py
"""
from __future__ import annotations

import math
import time
import torch

from tf_rewind_grad_torch import backward_ct_gemm


def forward_gdn_range(
    S_init: torch.Tensor,  # [H, D, D]
    k: torch.Tensor,       # [T, H, D]
    v: torch.Tensor,       # [T, H, D]
    alpha: torch.Tensor,   # [T, H]
    beta: torch.Tensor,    # [T, H]
    start: int,
    end: int,
) -> torch.Tensor:
    """Standard sequential forward GDN replay from start -> end."""
    S = S_init.clone()
    for t in range(start, end):
        a = alpha[t].view(-1, 1, 1)
        b = beta[t].view(-1, 1)
        S_dec = a * S
        c = b * (v[t] - torch.einsum("hij,hj->hi", S_dec, k[t]))
        S = S_dec + torch.einsum("hi,hj->hij", c, k[t])
    return S


def bidirectional_recover(
    S_lo: torch.Tensor,    # checkpoint at block start a_k
    S_hi: torch.Tensor,    # checkpoint at block end   a_{k+1}
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    c_tape: torch.Tensor,
    log_a_cpu: torch.Tensor,
    a_k: int,
    a_next: int,
    P: int,
    u_eff: float = 5e-7,
    eps_tol: float = 5e-5,
) -> tuple[torch.Tensor, str, float]:
    """Bidirectional ReplaySSM + TF-Rewind (Theorem 5 GEMM + Theorem 4 Cert).

    Guarantees <= eps_tol error on 100% of heads while terminating non-stationary
    heads cleanly at anchor a_k (O(1) bounded memory in total context length T!).
    """
    H = S_lo.shape[0]
    fwd_dist = P - a_k
    bwd_dist = a_next - P

    # First half of block: standard forward ReplaySSM from a_k (bit-exact, <= B/2 steps!)
    if fwd_dist <= bwd_dist:
        return forward_gdn_range(S_lo, k, v, alpha, beta, a_k, P), "FWD(ReplaySSM)", float(fwd_dist)

    # Second half of block: Backward TF-Rewind (Cert + Theorem 5 Single-GEMM) from a_{k+1}!
    log_u = math.log(u_eff)
    log_tol = math.log(eps_tol)
    log_ext = -11.5

    bwd_heads: list[int] = []
    zero_heads: list[int] = []
    zero_starts: list[int] = []
    anchor_heads: list[int] = []

    total_steps = 0
    for h in range(H):
        log_err_back = log_u - float(log_a_cpu[P:a_next, h].sum())
        if log_err_back <= log_tol:
            bwd_heads.append(h)
            total_steps += bwd_dist
        else:
            # Fast or non-stationary head: walk backward toward a_k for zero-start contraction
            opt_start = P
            acc = 0.0
            while opt_start > a_k and acc > log_ext:
                opt_start -= 1
                acc += float(log_a_cpu[opt_start, h])
            if acc <= log_tol:
                zero_heads.append(h)
                zero_starts.append(opt_start)
                total_steps += (P - opt_start)
            else:
                # Adversarial non-stationary head hits anchor a_k -> exact anchor fallback (Mode 3)!
                anchor_heads.append(h)
                total_steps += fwd_dist

    S_out = torch.empty_like(S_hi)
    if bwd_heads:
        bi = torch.tensor(bwd_heads, device=k.device, dtype=torch.long)
        # ONE parallel Tensor-Core GEMM jumps ALL bwd_dist tokens at once!
        S_out[bi] = backward_ct_gemm(S_hi[bi], k[P:a_next, bi], alpha[P:a_next, bi], c_tape[P:a_next, bi])
    if zero_heads:
        zi = torch.tensor(zero_heads, device=k.device, dtype=torch.long)
        S_z = torch.zeros(len(zero_heads), S_lo.shape[1], S_lo.shape[2], device=k.device, dtype=torch.float32)
        min_s = min(zero_starts)
        st_t = torch.tensor(zero_starts, device=k.device, dtype=torch.long)
        for t in range(min_s, P):
            active = (st_t <= t).view(-1, 1, 1)
            a = alpha[t, zi].view(-1, 1, 1)
            b = beta[t, zi].view(-1, 1)
            S_dec = a * S_z
            c = b * (v[t, zi] - torch.einsum("hij,hj->hi", S_dec, k[t, zi]))
            S_z = torch.where(active, S_dec + torch.einsum("hi,hj->hij", c, k[t, zi]), S_z)
        S_out[zi] = S_z
    if anchor_heads:
        ai = torch.tensor(anchor_heads, device=k.device, dtype=torch.long)
        S_out[ai] = forward_gdn_range(S_lo[ai], k[:, ai], v[:, ai], alpha[:, ai], beta[:, ai], a_k, P)

    return S_out, "BWD(TF-Rewind)", total_steps / float(H)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, D = 256, 48, 128  # Matches 1 full linear layer of Qwen3.8-27B (48 heads, D=128)
    g = torch.Generator(device="cpu").manual_seed(2026)

    # Prefix block [0 .. B] preceded by warm prior state S_lo != 0
    S_lo = torch.randn(H, D, D, generator=g, device=device) * 0.1
    k = torch.randn(B, H, D, generator=g, device=device)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(B, H, D, generator=g, device=device)

    alpha = torch.empty(B, H)
    beta = torch.empty(B, H).uniform_(0.05, 0.55, generator=g)
    # Realistic Qwen3.8-27B spectrum: 60% slow, 30% med, 10% fast + beta=0.999 spikes
    n_slow, n_med = int(H * 0.60), int(H * 0.30)
    alpha[:, :n_slow].uniform_(0.96, 0.999, generator=g)
    alpha[:, n_slow:n_slow + n_med].uniform_(0.72, 0.94, generator=g)
    alpha[:, n_slow + n_med:].uniform_(0.05, 0.45, generator=g)
    for h in range(int(H * 0.80)):
        ts = torch.randint(0, B, (4,), generator=g)
        beta[ts, h] = torch.tensor([0.95, 0.98, 0.995, 0.999])
    alpha, beta = alpha.to(device), beta.to(device)
    log_a_cpu = torch.log(torch.clamp(alpha, min=1e-12)).cpu()

    # Run recorded forward across block [0 .. B] saving ground-truth states at probe split points
    split_points = [32, 96, 160, 192, 224, 240, 252]  # Unplanned mid-block edits across [0 .. 256]
    gt_states: dict[int, torch.Tensor] = {}
    c_tape = torch.empty(B, H, D, device=device)
    S = S_lo.clone()
    for t in range(B):
        a = alpha[t].view(H, 1, 1)
        b = beta[t].view(H, 1)
        S_dec = a * S
        c = b * (v[t] - torch.einsum("hij,hj->hi", S_dec, k[t]))
        c_tape[t] = c
        S = S_dec + torch.einsum("hi,hj->hij", c, k[t])
        if (t + 1) in split_points:
            gt_states[t + 1] = S.clone()
    S_hi = S

    # Warmup
    _ = forward_gdn_range(S_lo, k, v, alpha, beta, 0, 16)
    _ = bidirectional_recover(S_lo, S_hi, k, v, alpha, beta, c_tape, log_a_cpu, 0, B, 240)

    print(f"=== Bidirectional ReplaySSM + TF-Rewind on Unplanned Mid-Block Splits (B={B}, H={H}, D={D}) ===")
    print(f"  {'Split P':>7} | {'Fwd-Only ReplaySSM':>20} | {'Bidirectional (+TF-Rewind)':>28} | {'Speedup':>8} | {'MaxRelErr':>10} | {'H<1e-3':>6}")

    tot_fwd_ms = 0.0
    tot_bi_ms = 0.0
    for P in split_points:
        t0 = time.perf_counter()
        _ = forward_gdn_range(S_lo, k, v, alpha, beta, 0, P)
        ms_fwd = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        S_bi, path, avg_stp = bidirectional_recover(S_lo, S_hi, k, v, alpha, beta, c_tape, log_a_cpu, 0, B, P)
        ms_bi = (time.perf_counter() - t1) * 1000.0

        tot_fwd_ms += ms_fwd
        tot_bi_ms += ms_bi

        ref = gt_states[P]
        err = torch.linalg.matrix_norm(S_bi - ref, ord="fro") / torch.linalg.matrix_norm(ref, ord="fro").clamp(min=1e-12)
        mx = float(err.max())
        ok = int((err < 1e-3).sum())
        assert ok == H and mx < 1e-4, f"Split P={P} violated cert: ok={ok}/{H}, max={mx}"
        print(f"  P = {P:3d} | {P:3d} stp ({ms_fwd:6.2f} ms) | {path:<14} {avg_stp:5.1f} stp ({ms_bi:5.2f} ms) | {ms_fwd / max(ms_bi, 1e-6):6.2f}x | {mx:10.2e} | {ok:2d}/{H}")

    print(f"\n  Mean speedup across unplanned splits : {tot_fwd_ms / tot_bi_ms:.2f}x faster wall-clock time!")
    print("  Checkpoint RAM savings at equal worst-case replay distance (27B, 144 MiB/ckpt):")
    print("    • 4K context  (max 128 tok): Forward ReplaySSM = 4.50 GiB (32 ckpt) -> Bidirectional = 2.25 GiB (16 ckpt, 2.0x less!)")
    print("    • 32K context (max 128 tok): Forward ReplaySSM = 36.0 GiB (256 ckpt) -> Bidirectional = 18.0 GiB (128 ckpt, 2.0x less!)")
    print("\nPASS: Bidirectional ReplaySSM + TF-Rewind certified 100% of heads (< 1e-4) across every unplanned split!")


if __name__ == "__main__":
    main()
