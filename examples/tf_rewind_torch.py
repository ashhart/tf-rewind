#!/usr/bin/env python3
"""TF-Rewind: Standalone PyTorch Reference & Verification of Theorems 1-4.

Verifies Singularity-Free c_t-Inverse, Non-Stationary Minimax Duality (kappa),
and Runtime-Certified Adaptive Replay on realistic Gated DeltaNet heads —
including write-gate spikes (beta -> 1.0) and adversarial non-stationary decay
flushes (pre-remember / post-flush, kappa << 1) — with ZERO checkpoints.

Runs in ~2 seconds on CPU, CUDA, or MPS:
    python3 examples/tf_rewind_torch.py
"""
from __future__ import annotations

import math
import torch


def gdn_forward_record(
    k: torch.Tensor,      # [T, H, D] unit-norm keys
    v: torch.Tensor,      # [T, H, D] values
    alpha: torch.Tensor,  # [T, H]    decay gate in (0, 1]
    beta: torch.Tensor,   # [T, H]    write gate in [0, 1]
    target_pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward Gated DeltaNet scan recording innovation vector c_t (zero extra FLOPs).

    Forward recurrence:
        c_t = beta_t * (v_t - alpha_t * S_{t-1} k_t)
        S_t = alpha_t * S_{t-1} + c_t k_t^T
    """
    T, H, D = k.shape
    S = torch.zeros(H, D, D, device=k.device, dtype=torch.float32)
    c_tape = torch.empty(T, H, D, device=k.device, dtype=torch.float32)
    S_ref = None

    for t in range(T):
        a = alpha[t].view(H, 1, 1)
        b = beta[t].view(H, 1)
        kt, vt = k[t], v[t]
        S_dec = a * S
        sk = torch.einsum("hij,hj->hi", S_dec, kt)
        ct = b * (vt - sk)               # Already computed by forward GDN!
        c_tape[t] = ct
        S = S_dec + torch.einsum("hi,hj->hij", ct, kt)
        if t + 1 == target_pos:
            S_ref = S.clone()

    assert S_ref is not None
    return S, S_ref, c_tape


def backward_ct_inverse(
    S_T: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    c_tape: torch.Tensor,
    target_pos: int,
) -> torch.Tensor:
    """Theorem 1: Singularity-free backward inverse S_{t-1} = (S_t - c_t k_t^T) / alpha_t."""
    S = S_T.clone()
    T = k.shape[0]
    for t in range(T - 1, target_pos - 1, -1):
        inv_a = (1.0 / alpha[t]).view(-1, 1, 1)
        S = (S - torch.einsum("hi,hj->hij", c_tape[t], k[t])) * inv_a
    return S


def backward_sm_inverse(
    S_T: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target_pos: int,
) -> torch.Tensor:
    """Ablation: Naive Sherman-Morrison inverse (explodes when beta_t * ||k_t||^2 -> 1)."""
    S = S_T.clone()
    T = k.shape[0]
    for t in range(T - 1, target_pos - 1, -1):
        kt, vt = k[t], v[t]
        kn2 = (kt * kt).sum(dim=-1)
        gamma = (beta[t] / torch.clamp(1.0 - beta[t] * kn2, min=1e-7)).view(-1, 1)
        sk = torch.einsum("hij,hj->hi", S, kt)
        inv_a = (1.0 / alpha[t]).view(-1, 1, 1)
        S = (S + torch.einsum("hi,hj->hij", gamma * (sk - vt), kt)) * inv_a
    return S


def forward_zero_replay(
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    starts: list[int],
    target_pos: int,
) -> torch.Tensor:
    """Zero-start forward replay from per-head start index starts[h] to target_pos."""
    _, H, D = k.shape
    S = torch.zeros(H, D, D, device=k.device, dtype=torch.float32)
    min_start = min(starts)
    start_t = torch.tensor(starts, device=k.device, dtype=torch.long)
    for t in range(min_start, target_pos):
        active = (start_t <= t).view(H, 1, 1)
        a = alpha[t].view(H, 1, 1)
        b = beta[t].view(H, 1)
        S_dec = a * S
        ct = b * (v[t] - torch.einsum("hij,hj->hi", S_dec, k[t]))
        S_next = S_dec + torch.einsum("hi,hj->hij", ct, k[t])
        S = torch.where(active, S_next, S)
    return S


def tf_rewind(
    S_T: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    c_tape: torch.Tensor,
    target_pos: int,
    certified: bool = True,
    base_window: int = 64,
    u_eff: float = 5e-7,
    eps_tol: float = 5e-5,
) -> tuple[torch.Tensor, int, int, int]:
    """Theorems 3-4: Per-head Forget-Invert Duality + Runtime-Certified Non-Stationary Replay."""
    T, H, _ = k.shape
    delta = T - target_pos
    eff_w = max(base_window, 2 * delta)
    log_a = torch.log(torch.clamp(alpha, min=1e-12)).cpu()
    log_u = math.log(u_eff)
    log_tol = math.log(eps_tol)
    log_ext = -11.5  # exp(-11.5) ~ 1e-5 margin when extending non-stationary heads

    bwd_heads: list[int] = []
    fwd_heads: list[int] = []
    fwd_starts: list[int] = []
    n_extended = 0

    for h in range(H):
        log_err_back = log_u - float(log_a[target_pos:T, h].sum())
        start_w = max(0, target_pos - eff_w)

        opt_start = target_pos
        acc_log = 0.0
        while opt_start > start_w:
            opt_start -= 1
            acc_log += float(log_a[opt_start, h])
            if acc_log <= -14.0:
                break
        log_err_fwd = -100.0 if opt_start == 0 else acc_log

        if certified:
            if log_err_back <= log_err_fwd and log_err_back <= log_tol:
                bwd_heads.append(h)
            elif log_err_fwd <= log_tol:
                fwd_heads.append(h)
                fwd_starts.append(opt_start)
            else:
                # Non-stationary head detected (kappa << 1): extend ONLY this head until certified!
                while opt_start > 0 and acc_log > log_ext:
                    opt_start -= 1
                    acc_log += float(log_a[opt_start, h])
                fwd_heads.append(h)
                fwd_starts.append(opt_start)
                n_extended += 1
        else:
            if log_err_back <= log_err_fwd:
                bwd_heads.append(h)
            else:
                fwd_heads.append(h)
                fwd_starts.append(opt_start)

    S_out = torch.empty_like(S_T)
    if bwd_heads:
        idx = torch.tensor(bwd_heads, device=k.device, dtype=torch.long)
        S_out[idx] = backward_ct_inverse(S_T[idx], k[:, idx], alpha[:, idx], c_tape[:, idx], target_pos)
    if fwd_heads:
        idx = torch.tensor(fwd_heads, device=k.device, dtype=torch.long)
        S_out[idx] = forward_zero_replay(k[:, idx], v[:, idx], alpha[:, idx], beta[:, idx], fwd_starts, target_pos)

    return S_out, len(bwd_heads), len(fwd_heads), n_extended


def rel_errors(S_got: torch.Tensor, S_ref: torch.Tensor) -> tuple[float, float, int]:
    diff = torch.linalg.matrix_norm(S_got - S_ref, ord="fro")
    ref = torch.linalg.matrix_norm(S_ref, ord="fro").clamp(min=1e-12)
    err = torch.where(torch.isfinite(diff / ref), diff / ref, torch.full_like(diff, 1e9))
    return float(err.median()), float(err.max()), int((err < 1e-3).sum())


def make_realistic_gdn_trace(T: int, H: int, D: int, P_adv: int, device: torch.device):
    """Synthesize realistic Qwen3.5/3.8 GDN spectrum + adversarial non-stationary flushes."""
    g = torch.Generator(device="cpu").manual_seed(42)
    k = torch.randn(T, H, D, generator=g)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(T, H, D, generator=g)

    alpha = torch.empty(T, H)
    beta = torch.empty(T, H).uniform_(0.05, 0.55, generator=g)

    n_slow = int(H * 0.60)
    n_med = int(H * 0.25)
    n_fast = int(H * 0.10)
    # Remaining 5% of heads are ADVERSARIAL NON-STATIONARY (kappa << 1: retain before P, flush after P!)
    alpha[:, :n_slow].uniform_(0.96, 0.999, generator=g)
    alpha[:, n_slow:n_slow + n_med].uniform_(0.70, 0.94, generator=g)
    alpha[:, n_slow + n_med:n_slow + n_med + n_fast].uniform_(0.05, 0.45, generator=g)
    adv_start = n_slow + n_med + n_fast
    alpha[:P_adv, adv_start:].uniform_(0.990, 0.999, generator=g)   # strongly remembers before P
    alpha[P_adv:, adv_start:].uniform_(0.15, 0.35, generator=g)     # aggressively flushes after P!

    # Inject realistic beta >= 0.95 write-gate spikes into 80% of heads (including beta = 0.999)
    for h in range(int(H * 0.80)):
        spike_ts = torch.randint(0, T, (4,), generator=g)
        beta[spike_ts, h] = torch.tensor([0.95, 0.98, 0.995, 0.999])

    return k.to(device), v.to(device), alpha.to(device), beta.to(device), H - adv_start


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, H, D = 512, 64, 128
    print(f"=== TF-Rewind PyTorch Verification (device={device}, T={T}, H={H}, D={D}, ZERO checkpoints) ===")

    for delta in (4, 32, 128):
        P = T - delta
        k, v, alpha, beta, n_adv = make_realistic_gdn_trace(T, H, D, P, device)
        S_T, S_ref, c_tape = gdn_forward_record(k, v, alpha, beta, P)

        print(f"\n--- Rewind Delta = {delta:3d} (T={T} -> P={P}, incl. {n_adv} adversarial kappa<<1 heads & beta=0.999 spikes) ---")
        print(f"  {'Policy':<22} | {'B/F (Ext)':<12} | {'MedRelErr':>10} | {'MaxRelErr':>10} | {'Heads < 1e-3':>12}")

        rows = [
            ("Tail-Replay (W=64)", forward_zero_replay(k, v, alpha, beta, [max(0, P - 64)] * H, P), f"0/{H} (0)"),
            ("SM-Inverse Only", backward_sm_inverse(S_T, k, v, alpha, beta, P), f"{H}/0 (0)"),
            ("c_t-Inverse Only", backward_ct_inverse(S_T, k, alpha, c_tape, P), f"{H}/0 (0)"),
        ]
        S_2d, nb, nf, _ = tf_rewind(S_T, k, v, alpha, beta, c_tape, P, certified=False)
        rows.append(("TF-Rewind (2D)", S_2d, f"{nb}/{nf} (0)"))
        S_cert, nb, nf, ext = tf_rewind(S_T, k, v, alpha, beta, c_tape, P, certified=True)
        rows.append(("TF-Rewind (Cert)", S_cert, f"{nb}/{nf} ({ext})"))

        for name, S_got, split in rows:
            med, mx, ok = rel_errors(S_got, S_ref)
            print(f"  {name:<22} | {split:<12} | {med:10.2e} | {mx:10.2e} | {ok:3d}/{H}")

        _, max_cert, ok_cert = rel_errors(S_cert, S_ref)
        assert ok_cert == H and max_cert < 1e-4, f"Certificate violated: {ok_cert}/{H}, max={max_cert}"

    print("\nPASS: TF-Rewind (Cert) certified 100% of heads (< 1e-4) across all Delta — including adversarial kappa << 1 heads & beta = 0.999 spikes!")


if __name__ == "__main__":
    main()
