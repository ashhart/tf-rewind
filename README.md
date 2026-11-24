# TF-Rewind — Checkpoint-Free Arbitrary-Boundary State Rewind for Hybrid Linear-Attention LLMs

**TensorFold Research** · Paper: [paper/TF_REWIND.md](paper/TF_REWIND.md) · Bare-Metal Engine: `m4lm` (Zig + direct Metal, zero Apple MLX, zero external dependencies)

**TF-Rewind** introduces **Forget–Invert Duality** — a mathematically optimal, **zero-checkpoint** algorithm that lets hybrid linear-attention / Gated DeltaNet (GDN) LLMs (`Qwen3.5`, `Qwen3.8`, Mamba-2, RWKV-7, GLA, KDA) **rewind ("Ctrl-Z") their recurrent state to any arbitrary past token position** in milliseconds with exact output parity, eliminating the multi-second full re-prefill penalty whenever an agent retries a tool call, branches a conversation, edits a prompt, or rolls back rejected speculative tokens.

---

## Why Hybrid LLMs Couldn't "Ctrl-Z" — Until Now

In a pure Transformer, rewinding from token $T$ back to any earlier token $P = T - \Delta$ is free: just truncate the KV cache index (`pos = P`).

In a **hybrid Gated DeltaNet (GDN) + GQA model** (such as `Qwen3.5` and `Qwen3.8`), the linear-attention layers compress all past tokens into a fixed-size matrix state $S_t \in \mathbb{R}^{d_v \times d_k}$ (`144 MiB` on `Qwen3.8-27B`) updated via:

$$S_t = \alpha_t S_{t-1}\left(I - \beta_t k_t k_t^\top\right) + \beta_t v_t k_t^\top$$

Because $S_t$ overwrites $S_{t-1}$ in place, rewinding to an un-checkpointed position $P$ previously forced an impossible choice:
- **Save checkpoints everywhere** (`144 MiB` per snapshot on 27B) — unsustainable across arbitrary token boundaries.
- **Full re-prefill from token 0** — `7+ seconds` on 27B for a single 1-line edit or tool retry.
- **2026 SOTA `Tail-Replay`** (zero-start forward replay over a short window) — breaks on the **`59%–65%` of real trained GDN heads with slow decay ($\alpha \ge 0.95$)**, causing state drift and hallucinations.
- **Naive Sherman–Morrison algebraic inversion** — explodes to `NaN` because **`76%–90%` of real trained GDN heads hit $\beta_t \ge 0.90$**, making $(1 - \beta_t \|k_t\|_2^2)^{-1}$ singular.

---

## The Breakthrough: Forget–Invert Duality

TF-Rewind solves both failure modes simultaneously with **zero state checkpoints**:

### 1. Singularity-Free $c_t$-Inverse (Zero Extra FLOPs, `63.6×–76.3×` Smaller Tape)
During the normal forward scan, every GDN kernel **already computes** the innovation vector $c_t = \beta_t (v_t - \alpha_t S_{t-1} k_t) \in \mathbb{R}^{d_v}$ in registers before writing $S_t = \alpha_t S_{t-1} + c_t k_t^\top$. Recording $c_t$ into a tiny ring tape (`1.93 MiB` vs `144 MiB` full state on 27B) makes exact backward inversion **unconditionally singularity-free** and dot-product-free:

$$S_{t-1} = \frac{S_t - c_t k_t^\top}{\alpha_t}$$

No $\beta_t$ appears anywhere in the inverse — completely immune to $\beta_t \to 1$.

### 2. Universal Minimax Duality (Per-Head Optimal Routing)
The only remaining error axis is the decay rate $\alpha_t \in (0, 1]$ — where Backward $c_t$-Inverse and Forward Zero-Start Tail-Replay have **exact opposite stability regimes**:
- **Slow-decay heads ($\alpha \approx 1$):** Backward $c_t$-Inverse has error $E_{\text{back}} \approx u_{\text{eff}} / \prod_{t=P}^{T-1}\alpha_t \approx 10^{-7}$ (where Tail-Replay fails catastrophically).
- **Fast-decay heads ($\alpha \ll 1$):** Forward Zero-Start Tail-Replay over window $W = r\Delta$ has error $E_{\text{fwd}} \le \prod_{t=P-W}^{P-1}\alpha_t \ll 10^{-6}$ (where backward division by $\alpha_t$ would amplify rounding).

Choosing per head whichever mode minimizes $\min(E_{\text{back}}, E_{\text{fwd}})$ with window $W = r\Delta$ yields a **universal minimax error bound completely independent of $\alpha$, $\beta$, and $\Delta$**:

$$\sup_{\bar{\alpha} \in (0, 1]} \min\Big(E_{\text{back}}(\bar{\alpha}),\; E_{\text{fwd}}(\bar{\alpha})\Big) = u_{\text{eff}}^{\,\frac{r}{r+1}}$$

At $r = 2$ ($W = 2\Delta$) in fp32 ($u_{\text{eff}} \approx 5\times 10^{-7}$), worst-case relative state error is bounded by $\le 6 \times 10^{-5}$ across **all heads** with **zero checkpoints**.

---

## Verified Results (Apple M5 Max, 128 GB)

### Controlled Ablation on `Qwen3.8-27B` (48 Linear Layers, 2,304 GDN Heads, Zero Checkpoints)

Rewinding from $T = 487$ to un-checkpointed positions $P = T - \Delta$:

| Rewind $\Delta$ | Method (0 checkpoints) | Split B/F | GPU Time | Med Rel Err | p90 Rel Err | Heads $< 10^{-3}$ | $\text{KL}(P_{\text{ref}} \| P_{\text{got}})$ | Token Parity |
| :---: | :--- | :---: | ---: | ---: | ---: | :---: | ---: | :---: |
| **4** | `Tail-Replay (2026 SOTA)` | `0 / 2304` | `23.8 ms` | `4.91e-2` | `6.46e-1` | `834 / 2304` | `1.77e-2` | `5 / 32` |
| **4** | `SM-Inverse Only` | `2304 / 0` | `4.4 ms` | `2.09e-7` | `1.35e-3` | `2066 / 2304` | `9.51e+0` | `5 / 32` |
| **4** | **`TF-Rewind (2D, Ours)`** | `1715 / 589` | **`15.5 ms`** | **`1.05e-7`** | **`3.20e-7`** | **`2304 / 2304`** | **`1.28e-5`** | **`32 / 32`** |
| **16** | `Tail-Replay (2026 SOTA)` | `0 / 2304` | `19.6 ms` | `3.00e-2` | `5.64e-1` | `875 / 2304` | `8.74e-3` | `18 / 32` |
| **16** | **`TF-Rewind (2D, Ours)`** | `1661 / 643` | **`15.5 ms`** | **`2.44e-7`** | **`6.47e-7`** | **`2304 / 2304`** | **`1.72e-4`** | **`32 / 32`** |
| **32** | `Tail-Replay (2026 SOTA)` | `0 / 2304` | `17.8 ms` | `1.40e-2` | `4.89e-1` | `908 / 2304` | `6.19e-4` | `30 / 32` |
| **32** | **`TF-Rewind (2D, Ours)`** | `1586 / 718` | **`10.9 ms`** | **`3.53e-7`** | **`1.84e-6`** | **`2304 / 2304`** | **`5.29e-5`** | **`32 / 32`** |
| **64** | `Tail-Replay (2026 SOTA)` | `0 / 2304` | `25.2 ms` | `5.89e-3` | `5.08e-1` | `984 / 2304` | `5.32e-4` | `27 / 32` |
| **64** | **`TF-Rewind (2D, Ours)`** | `1399 / 905` | **`22.3 ms`** | **`5.39e-7`** | **`5.58e-6`** | **`2302 / 2304`** | **`4.91e-7`** | **`32 / 32`** |
| **128** | `Tail-Replay (2026 SOTA)` | `0 / 2304` | `56.1 ms` | `3.36e-1` | `8.69e-1` | `563 / 2304` | `5.88e+0` | `5 / 32` |
| **128** | **`TF-Rewind (2D, Ours)`** | `1274 / 1030` | **`25.0 ms`** | **`1.28e-6`** | **`2.33e-4`** | **`2178 / 2304`** | **`4.20e-6`** | **`32 / 32`** |

- **`160 / 160` (100% exact continuation parity)** across every $\Delta \in \{4, 16, 32, 64, 128\}$ on 27B with **zero checkpoints**.
- **`1,400,000×` lower KL divergence** than 2026 SOTA `Tail-Replay` at $\Delta = 128$ (`4.20e-6` vs `5.88`).
- Full cross-scale results on `Qwen3.5-0.8B`, `Qwen3.5-2B`, and `Qwen3.8-27B` are documented in [paper/TF_REWIND.md](paper/TF_REWIND.md).

### Live End-to-End "Needle-in-a-Haystack Ctrl-Z" Demo (`m4lm tf-rewind-demo`)

An 816-token specification manual containing a secret key (`ORION-8391-ZETA`) near token ~18 is read, followed by Turn A (49 tokens). With **zero checkpoints saved**, the engine rewinds $\Delta = 49$ tokens to un-checkpointed position $P = 816$ and answers a brand-new query requiring exact secret recall:

| Model | Cold Re-Prefill TTFT | `Tail-Replay` (2026 SOTA) | **`TF-Rewind` (Ours, 0 ckpt)** |
| :--- | :---: | :--- | :--- |
| **`Qwen3.5-0.8B`** | `546.1 ms` | `53.7 ms` (`10.2×`), **`10/33` exact** — *hallucinated: "eliminates the CPU and GPU"* | **`31.8 ms` (`17.2×`), `33/33` (100% exact)** |
| **`Qwen3.5-2B`** | `849.0 ms` | `58.2 ms` (`14.6×`), `25/25` exact | **`60.6 ms` (`14.0×`, `3.1 ms` rewind), `25/25` (100% exact)** |
| **`Qwen3.8-27B`** | `7,083.6 ms` | `432.0 ms` (`16.4×`), `21/23` exact (drift) | **`457.8 ms` (`15.5×`, `16.9 ms` rewind), `23/23` (100% exact)** |

---

## Quickstart: Build & Reproduce TF-Rewind

Requires macOS on Apple Silicon (Metal) and Zig `0.17+`. Zero external libraries, zero Apple MLX.

```bash
zig build --release=fast
# Outputs:
#   zig-out/bin/m4lm          (~1.1 MB standalone CLI + OpenAI HTTP/SSE server)
#   zig-out/lib/libm4lm.dylib (~840 KB C ABI shared library)

# 1. Run the full 0-checkpoint Forget-Invert Duality ablation (Delta = 4, 16, 32, 64, 128)
./zig-out/bin/m4lm tf-rewind MODEL_DIR --window 64 --gen 64

# 2. Run the live Needle-in-a-Haystack Ctrl-Z demo (Cold Re-Prefill vs Tail-Replay vs TF-Rewind)
./zig-out/bin/m4lm tf-rewind-demo MODEL_DIR --repeats 14
```

Core TF-Rewind implementation files:
- Metal GPU kernels (`gdn_scan_record_corr`, `gdn_rewind_dual`): [kernels/gdn_ops.metal](kernels/gdn_ops.metal#L616-L702)
- Host planner, minimax mode selector, ablations & Ctrl-Z demo: [src/dual_rewind.zig](src/dual_rewind.zig)
- Full technical paper & proofs: [paper/TF_REWIND.md](paper/TF_REWIND.md)

---

## The Engine Underneath: `m4lm` Clean-Room Direct-Metal Runtime

TF-Rewind was discovered and built inside **`m4lm`**, a clean-room LLM inference engine written from scratch in **Zig + direct Metal** for Apple Silicon (5 verified models across 2 architectures, zero Apple MLX):

| Architecture | Models Verified | Weight Format | Highlights |
| :--- | :--- | :--- | :--- |
| **Hybrid Gated DeltaNet + GQA** | `Qwen3.5-0.8B-4bit`, `Qwen3.5-2B-4bit`, `Qwen3.8-27B-MLX-4bit` (+ `z-lab/Qwen3.8-27B-DFlash2`) | MLX 4-bit group-64 | **TF-Rewind** zero-checkpoint Ctrl-Z, ping-pong SRAM-resident `gdn_fused_step_1`, bit-exact tree FlashAttention, DFlash2 + cross-model + copy speculation, zero-copy KV+GDN prefix cache & 16-stream prefix forking |
| **Standard Dense Transformer** | `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-7B` | **Raw upstream HF BF16** (quantized on the fly to 4-bit `qk` tiles in parallel via GCD) | Zero pre-conversion, `HD=128` full split-half RoPE + fused QKV bias, branching tree verify, adaptive copy lookahead, zero-copy prefix cache |

### Engine Benchmarks & Parity (Apple M5 Max, 128 GB)

| Model & Capability | Parity / Exactness | `m4lm` Result | Speedup vs `mlx_lm` / Baseline |
| :--- | :--- | ---: | ---: |
| `Qwen3.5-0.8B` prefill / `c=1` / copy | `30/30`, `128/128`, `92/92` exact | **16,989** / **561** / **5,883–6,754 tok/s** | **4.9×** / **1.36–1.58×** / **12.2–14.3×** vs `mlx_lm` |
| `Qwen3.5-0.8B` 16-stream forked wave | `79/79` bit-exact | **2,456 tok/s** agg (`1.68 ms` fork) | **8.14×** solo (`2,595` cached tok) |
| `Qwen3.5-0.8B` hybrid prefix cache | `48/48` bit-exact | **14.13 ms** TTFT (`0.21 ms` GDN restore) | **16.30×** TTFT (`1,368` cached tok) |
| `Qwen3.5-2B` `c=1` / copy | bit-exact + `21/21` tree / `92/92` | **325.6** / **3,240–4,515 tok/s** | **1.13×** / **11.4–15.8×** vs `mlx_lm` |
| `Qwen3.5-2B` 16-stream forked wave | `128/128` bit-exact | **2,179–2,600 tok/s** agg (`1.42 ms` fork) | **7.8–9.2×** solo (`2,595` cached tok) |
| `Qwen3.5-2B` hybrid prefix cache | `48/48` bit-exact | **8.13 ms** TTFT (`0.26 ms` GDN restore) | **47.51×** TTFT (`1,368` cached tok) |
| `Qwen3.8-27B` + `DFlash2` prose / code / copy | `96/96`, `64/64`, `90/90` exact | **109.2** / **126.3** / **462.3 tok/s** | **3.62×** / **4.20×** / **16.93×** vs `mlx_lm` |
| `Qwen3.8-27B` + `0.8B` cross-draft (greedy / `T=0.7`) | `128/128` greedy; 200k-draw TVD `0.00150` | **53.4–136.8 tok/s** / **2.78 tok/rnd** | **2.1–6.7×** greedy / **1.48×** at `T=0.7` |
| `Qwen3.8-27B` 16-stream forked wave | `64/64` bit-exact | **294.0 tok/s** agg (`7.65 ms` fork) | **12.22×** solo (`1.15×` wall time of 1 stream) |
| `Qwen3.8-27B` hybrid prefix cache | `48/48` bit-exact | **123.4 ms** TTFT (`0.78 ms` GDN restore) | **37.70×** TTFT (`1,028` cached tok) |
| `Qwen3.8-27B` hardware roofline `c=1` | `21/21` tree, `92/92` copy | **223.4 GB/s** mean / **339.7 GB/s** p10 | **85.7% live mean DRAM** / **101.8% copy peak** |
| `Qwen2.5-1.5B` (raw BF16 → 4-bit on-the-fly) | `21/21` tree, `110/110` copy, `48/48` cache | **1.06 s** warm quantize+load | **58.08×** TTFT (`1,015` cached tok) |

### Full CLI, OpenAI Server & Drop-In Python Usage

```bash
# Single-stream generation (greedy adaptive copy lookahead, or sampled with --temp)
./zig-out/bin/m4lm generate MODEL_DIR --prompt "Explain GPU tensor cores." --max-tokens 256
./zig-out/bin/m4lm generate MODEL_DIR --prompt "Write a metaphor." --temp 0.8 --top-p 0.9 --min-p 0.05 --seed 42

# Qwen3.8-27B with DFlash2 (3.6-4.2x prose/code, 16.9x copy) or 0.8B cross-draft (greedy or temp > 0)
./zig-out/bin/m4lm generate MODEL_27B_DIR --draft DRAFTER_DIR --prompt "Write a TypeScript ring buffer."
./zig-out/bin/m4lm generate MODEL_27B_DIR --cross-draft MODEL_08B_DIR --temp 0.7 --top-p 0.9 --prompt "Explain FlashAttention."

# Native OpenAI HTTP/SSE + /v1/stream-parallel server (with automatic Hybrid KV+GDN Prefix Cache)
./zig-out/bin/m4lm serve MODEL_27B_DIR --draft DRAFTER_DIR --port 8080 --max-streams 16

# Built-in Verification, Roofline & Profiling Suite
./zig-out/bin/m4lm roofline     MODEL_DIR --tokens 64
./zig-out/bin/m4lm timeline     MODEL_DIR
./zig-out/bin/m4lm cache-bench  MODEL_DIR --repeats 12
./zig-out/bin/m4lm wave-bench   MODEL_DIR --streams 16 --tokens 128
./zig-out/bin/m4lm verify       MODEL_08B_DIR bench/mlx_lm_ref.json qwen35_0.8b
```

```python
# Drop-in mlx_lm-compatible Python API (PYTHONPATH=python, zero Apple MLX imported)
import m4lm

model, tokenizer = m4lm.load(
    "TensorFold/Qwen3.8-27B-MLX-4bit",
    draft_model="z-lab/Qwen3.8-27B-DFlash2",
)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain how M5 tensor units accelerate LLMs."}],
    add_generation_prompt=True,
)
print(m4lm.generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True))
```

---

## Citation

If you use **TF-Rewind** or **Forget–Invert Duality** in your research or inference systems, please cite:

```bibtex
@misc{tensorfold2026tfrewind,
  title  = {TF-Rewind: Exact Checkpoint-Free State Rewind for Hybrid Linear-Attention LLMs via Forget-Invert Duality},
  author = {TensorFold Research},
  year   = {2026},
  url    = {https://github.com/ashhart/tf-rewind}
}
```

## License

MIT — Copyright (c) 2026 TensorFold. See [LICENSE](LICENSE).
