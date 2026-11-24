#!/usr/bin/env python3
"""Generate the publication-grade 2-panel Hero Figure for TF-Rewind (TensorFold Research).

Outputs:
  assets/tf_rewind_hero.png  (high-DPI PNG for X/Twitter, GitHub README & social preview)
  assets/tf_rewind_hero.svg  (vector SVG)
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Palette — Obsidian Dark Research Theme (pops on both X dark/light feeds & GitHub)
BG = "#0A0E17"
CARD = "#111827"
BORDER = "#1F2937"
GRID = "#1E293B"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
CYAN = "#22D3EE"
EMERALD = "#34D399"
INDIGO = "#818CF8"
CORAL = "#F43F5E"
AMBER = "#F59E0B"


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "assets")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": BORDER,
        "axes.linewidth": 1.1,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "mathtext.fontset": "dejavusans",
    })

    fig = plt.figure(figsize=(15.5, 8.4), dpi=200, facecolor=BG)
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[0.13, 0.72, 0.15],
        width_ratios=[1.0, 1.08],
        hspace=0.24,
        wspace=0.22,
        left=0.06,
        right=0.96,
        top=0.95,
        bottom=0.04,
    )

    # ── Top Header Banner ───────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(BG)
    ax_hdr.axis("off")

    ax_hdr.text(
        0.0, 0.72,
        "TF-Rewind",
        fontsize=26, fontweight="bold", color=TEXT, va="center",
    )
    ax_hdr.text(
        0.162, 0.72,
        "  ·  TensorFold Research",
        fontsize=15.5, fontweight="bold", color=CYAN, va="center",
    )
    ax_hdr.text(
        0.0, 0.14,
        "Exact Arbitrary-Boundary State Rewind for Hybrid Linear-Attention LLMs  (Qwen3.8-27B · 2,304 GDN Heads · Zero Checkpoints)",
        fontsize=12.0, color=MUTED, va="center",
    )
    # Crisp right-aligned pill aligned cleanly with the title row (zero subtitle collision)
    pill = patches.FancyBboxPatch(
        (0.742, 0.48), 0.258, 0.44,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        facecolor="#0E2230", edgecolor=CYAN, linewidth=1.2,
        transform=ax_hdr.transAxes, clip_on=False,
    )
    ax_hdr.add_patch(pill)
    ax_hdr.text(
        0.871, 0.70,
        "Apple M5 Max · Bare Metal (Zig + Metal)",
        fontsize=10.2, fontweight="bold", color=CYAN,
        ha="center", va="center", transform=ax_hdr.transAxes,
    )

    # ── Panel A: Forget–Invert Duality Mechanism ────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(CARD)

    alpha = np.linspace(0.58, 0.9995, 500)
    delta_t = 32.0
    u_eff = 5e-7
    r = 2.0
    e_back = u_eff * np.power(alpha, -delta_t)
    e_fwd = np.power(alpha, r * delta_t)
    e_dual = np.minimum(e_back, e_fwd)
    alpha_star = (u_eff ** (1.0 / (r + 1.0))) ** (1.0 / delta_t)
    e_star = u_eff ** (r / (r + 1.0))

    # Shaded regime regions
    ax1.axvspan(0.58, alpha_star, color="#172554", alpha=0.28, zorder=0)
    ax1.axvspan(alpha_star, 1.003, color="#064E3B", alpha=0.22, zorder=0)
    ax1.axvline(alpha_star, color=MUTED, linestyle=":", linewidth=1.1, alpha=0.75, zorder=1)

    ax1.plot(alpha, e_fwd, color=CORAL, linestyle="--", linewidth=2.2,
             label=r"Forward Tail-Replay  $E_{\mathrm{fwd}} = \bar{\alpha}^{2\Delta}$", zorder=2)
    ax1.plot(alpha, e_back, color=AMBER, linestyle="--", linewidth=2.2,
             label=r"Backward $c_t$-Inverse  $E_{\mathrm{back}} = u / \bar{\alpha}^{\Delta}$", zorder=2)

    # Subtle glow + bold envelope for TF-Rewind
    ax1.plot(alpha, e_dual, color=CYAN, linewidth=7.5, alpha=0.16, zorder=3)
    ax1.plot(alpha, e_dual, color=CYAN, linewidth=3.0,
             label=r"TF-Rewind  $\min(E_{\mathrm{back}}, E_{\mathrm{fwd}})$", zorder=4)

    # Runtime certificate ceiling
    ax1.axhline(5e-5, color=EMERALD, linestyle="-.", linewidth=1.4, alpha=0.85, zorder=2)
    ax1.text(0.588, 7.6e-5, r"Runtime Cert $\epsilon_{\mathrm{tol}} = 5\times 10^{-5}$",
             fontsize=9.2, color=EMERALD, fontweight="bold")

    # Minimax crossing point annotation placed cleanly in open mid-left area (no line overlap!)
    ax1.scatter([alpha_star], [e_star], color=CYAN, edgecolor=TEXT, s=72, linewidth=1.4, zorder=6)
    ax1.annotate(
        "Minimax Crossing\n" + r"$u^{\frac{r\kappa}{r\kappa+1}} \approx 6.3\times 10^{-5}$" + "\n(Cert extends " + r"$\kappa \ll 1$" + " heads)",
        xy=(alpha_star, e_star),
        xytext=(0.592, 9.0e-4),
        fontsize=9.0, color=TEXT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#0F172A", edgecolor=CYAN, lw=1.1),
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.5, connectionstyle="arc3,rad=0.15"),
        zorder=7,
    )

    # Regime labels
    ax1.text(0.695, 2.2e-7, "Fast Decay Regime\nForward Replay (0 → P)",
             fontsize=9.2, color="#93C5FD", ha="center", va="bottom", fontweight="bold")
    ax1.text(0.930, 2.2e-7, "Slow Memory (59%)\nBackward $c_t$-Inverse",
             fontsize=9.2, color="#6EE7B7", ha="center", va="bottom", fontweight="bold")

    ax1.set_yscale("log")
    ax1.set_xlim(0.58, 1.003)
    ax1.set_ylim(1e-7, 2.5e1)
    ax1.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.8)
    ax1.set_title("A. Forget–Invert Duality (Singularity-Free Error Envelope)",
                  fontsize=13, fontweight="bold", color=TEXT, pad=12, loc="left")
    ax1.set_xlabel(r"Per-Head Effective Decay Rate $\bar{\alpha}$   (Fast Forget $\leftarrow$   $\rightarrow$ Slow Memory)",
                   fontsize=10.5, color=MUTED, labelpad=8)
    ax1.set_ylabel("Relative State Error (log scale)", fontsize=10.5, color=MUTED, labelpad=8)
    leg1 = ax1.legend(loc="upper center", bbox_to_anchor=(0.55, 0.985), frameon=True,
                      facecolor="#0B1220", edgecolor=BORDER, fontsize=8.8, ncol=1)
    for t in leg1.get_texts():
        t.set_color(TEXT)

    # ── Panel B: Empirical Qwen3.8-27B (2,304 Heads) MaxRelErr vs Delta ─────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor(CARD)

    deltas = np.array([4, 16, 32, 64, 128])
    x = np.arange(len(deltas))

    # Exact verified Apple M5 Max Qwen3.8-27B (2,304 GDN heads, zero checkpoints)
    tail_max = np.array([9.67e-1, 8.96e-1, 9.08e-1, 9.02e-1, 9.78e-1])
    w64_max = np.array([1.15e-6, 2.18e-5, 8.79e-5, 3.36e-3, 5.15e-1])
    d2_max = np.array([1.15e-6, 2.18e-5, 8.79e-5, 2.05e-3, 6.47e-2])
    cert_max = np.array([1.15e-6, 2.18e-5, 2.23e-5, 2.28e-5, 7.78e-5])

    ax2.plot(x, tail_max, color=CORAL, marker="s", markersize=7.5, linewidth=2.5,
             label="Tail-Replay (2026 SOTA · 5/32 parity at Δ=4, 128)", zorder=4)
    ax2.plot(x, w64_max, color="#64748B", marker="^", markersize=6.2, linestyle=":", linewidth=1.9,
             label="TF-Rewind Fixed W=64 (12/32 at Δ=128)", zorder=3)
    ax2.plot(x, d2_max, color=INDIGO, marker="D", markersize=6.6, linestyle="--", linewidth=2.2,
             label="TF-Rewind (2D, W=2Δ · 2,178/2,304 at Δ=128)", zorder=4)

    # Glow + bold curve for Certified TF-Rewind
    ax2.plot(x, cert_max, color=CYAN, linewidth=8.0, alpha=0.16, zorder=5)
    ax2.plot(x, cert_max, color=CYAN, marker="o", markersize=8.5, markeredgecolor=TEXT,
             markeredgewidth=1.3, linewidth=3.2,
             label="TF-Rewind (Cert, Ours · 2,304/2,304 & 160/160 exact!)", zorder=6)

    ax2.axhline(1e-3, color=MUTED, linestyle=":", linewidth=1.1, alpha=0.65, zorder=1)
    ax2.text(-0.22, 1.55e-3, "10⁻³ exactness threshold", fontsize=8.7, color=MUTED)

    # Callout at Delta=128 showing 832x drop from 2D -> Cert (cleanly inside open triangle!)
    ax2.annotate(
        "", xy=(4, cert_max[4] * 1.38), xytext=(4, d2_max[4] * 0.70),
        arrowprops=dict(arrowstyle="<->", color=EMERALD, lw=1.9), zorder=7,
    )
    ax2.text(
        3.90, 5.5e-4,
        "832× lower max err\n(447 non-stat heads\ncaught at runtime)",
        fontsize=8.6, fontweight="bold", color=EMERALD, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#08241C", edgecolor=EMERALD, lw=1.0),
        zorder=8,
    )

    # KL divergence badge cleanly above Tail-Replay at right (zero overlap with legend or line!)
    ax2.text(
        4.22, 7.5e0,
        "Δ=128 KL(P∥Q)\nTail-Replay: 5.88\nTF-Rewind: 3.9e-6\n(1.51M× lower)",
        fontsize=8.4, fontweight="bold", color="#FDA4AF", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#2A0D16", edgecolor=CORAL, lw=1.0),
        zorder=8,
    )

    # Inset callout box: ReplaySSM Boundary Cliff cleanly BELOW every single curve!
    inset = patches.FancyBboxPatch(
        (0.245, 0.025), 0.735, 0.215,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor="#0B1220", edgecolor=AMBER, linewidth=1.15, alpha=0.96,
        transform=ax2.transAxes, zorder=8,
    )
    ax2.add_patch(inset)
    ax2.text(
        0.265, 0.205,
        "vs. ReplaySSM Boundary Cliff (27B, Δ = 4, Anchor B = 243):",
        fontsize=8.4, fontweight="bold", color=AMBER, transform=ax2.transAxes, zorder=9,
    )
    ax2.text(
        0.265, 0.148,
        "• ReplaySSM (Full):  792.9 ms  (240 tok re-prefill + 144 MiB/ckpt)",
        fontsize=8.0, color=MUTED, transform=ax2.transAxes, zorder=9,
    )
    ax2.text(
        0.265, 0.096,
        "• ReplaySSM (Tape):   20.4 ms  (240 steps/head + 144 MiB/ckpt)",
        fontsize=8.0, color=MUTED, transform=ax2.transAxes, zorder=9,
    )
    ax2.text(
        0.265, 0.044,
        "• TF-Rewind (Cert):    3.95 ms  (201× / 5.2× faster · ZERO ckpts)",
        fontsize=8.2, fontweight="bold", color=CYAN, transform=ax2.transAxes, zorder=9,
    )

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Δ = {d}" for d in deltas], fontsize=10.2, fontweight="bold", color=TEXT)
    ax2.set_xlim(-0.35, 4.35)
    ax2.set_ylim(1.2e-8, 2.5e2)
    ax2.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.8)
    ax2.set_title("B. Qwen3.8-27B Worst-Case Head Error Across All 2,304 Heads",
                  fontsize=13, fontweight="bold", color=TEXT, pad=12, loc="left")
    ax2.set_xlabel("Arbitrary Un-Checkpointed Rewind Distance Δ  (T = 487 → P = T − Δ)",
                   fontsize=10.5, color=MUTED, labelpad=8)
    ax2.set_ylabel("Worst-Case Head Rel. Error (Max across 2,304 heads)", fontsize=10.2, color=MUTED, labelpad=8)
    leg2 = ax2.legend(loc="upper left", bbox_to_anchor=(0.012, 0.985), frameon=True,
                      facecolor="#0B1220", edgecolor=BORDER, fontsize=8.15)
    for t in leg2.get_texts():
        t.set_color(TEXT)

    # ── Bottom KPI Summary Cards ────────────────────────────────────────────
    ax_ftr = fig.add_subplot(gs[2, :])
    ax_ftr.set_facecolor(BG)
    ax_ftr.axis("off")

    kpis = [
        ("2,304 / 2,304", "100% of 27B GDN heads certified\nMaxRelErr ≤ 7.8×10⁻⁵ (Δ = 4..128)", CYAN),
        ("160 / 160", "100% exact continuation parity\nacross all Δ with zero checkpoints", EMERALD),
        ("201× faster", "vs. ReplaySSM (Full) at Δ = 4\n3.95 ms vs 792.9 ms (5.2× vs Tape)", AMBER),
        ("1,510,000×", "lower KL divergence than 2026 SOTA\nTail-Replay at Δ = 128 (3.9e-6 vs 5.88)", INDIGO),
    ]
    n = len(kpis)
    gap = 0.018
    w = (1.0 - gap * (n - 1)) / n
    for idx, (val, desc, col) in enumerate(kpis):
        x0 = idx * (w + gap)
        card = patches.FancyBboxPatch(
            (x0, 0.05), w, 0.90,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=CARD, edgecolor=BORDER, linewidth=1.1,
            transform=ax_ftr.transAxes,
        )
        ax_ftr.add_patch(card)
        bar = patches.Rectangle(
            (x0 + 0.005, 0.16), 0.0045, 0.68,
            facecolor=col, edgecolor="none",
            transform=ax_ftr.transAxes,
        )
        ax_ftr.add_patch(bar)
        ax_ftr.text(x0 + 0.020, 0.68, val, fontsize=16.5, fontweight="bold",
                    color=col, va="center", transform=ax_ftr.transAxes)
        ax_ftr.text(x0 + 0.020, 0.28, desc, fontsize=8.8, color=MUTED,
                    va="center", linespacing=1.28, transform=ax_ftr.transAxes)

    png_path = os.path.join(out_dir, "tf_rewind_hero.png")
    svg_path = os.path.join(out_dir, "tf_rewind_hero.svg")
    fig.savefig(png_path, dpi=200, facecolor=BG)
    fig.savefig(svg_path, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {png_path} and {svg_path}")


if __name__ == "__main__":
    main()
