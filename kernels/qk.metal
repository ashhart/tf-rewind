// Clean-room M5 hardware tensor-unit 4-bit affine matrix multiplication (qk).
//
//   Y[m, n] = sum_{g=0}^{G-1} ( scale[n, g] * (X[m, g] . codes[n, g]) + bias[n, g] * SUMS[g, m] )
//
// Packed weight block layout per (tile of NT=32 output cols, group of GS=64 input cols):
//   [0 .. 1024):   NT * 32 bytes of packed 4-bit unsigned codes (uint4b_format)
//   [1024 .. 1152): NT * 4 bytes of interleaved {scale_bf16 (low16), bias_bf16 (high16)}
//
// Instantiates ROWS = 16, 32, 48 in a single Metal 4 library so one kernel family covers
// single-token decode, multi-row tree verification, and chunked prompt prefill.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int GS = 64;
constant constexpr int NT = 32;
constant constexpr int SK = 4;
constant constexpr int CODES = NT * GS / 2;   // 1024 bytes
constant constexpr int BLOCK = CODES + NT * 4; // 1152 bytes
constant constexpr int SETS = NT / 16;         // 2 column sets of 4 outputs per lane

struct QkParams {
  int M;           // active rows in this dispatch (<= ROWS)
  int K;           // input width (multiple of 64)
  int N;           // output width (multiple of 32)
  int G;           // K / 64
  int first_row;   // starting row index in X, SUMS, Y
  int sums_stride; // stride between groups in SUMS (multiple of 16)
  uint tile_first; // first 32-col tile handled by this dispatch
};

inline int opPlace(int k, int s, int j) { return (s * 2 + k) * 4 + j; }
inline int accIdx(int k, int s, int j) { return (k * SETS + s) * 4 + j; }

template <int ROWS>
kernel void qk_kernel(const device bfloat* X [[buffer(0)]],
                      const device uchar* W [[buffer(1)]],
                      const device float* SUMS [[buffer(2)]],
                      device bfloat* Y [[buffer(3)]],
                      constant QkParams& p [[buffer(4)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint slice [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr int RUNS = ROWS / 8;
  constexpr int VALUES = RUNS * SETS * 4;

  const uint tile = p.tile_first + group;
  const device uchar* blocks = W + tile * (size_t)p.G * BLOCK;
  const int first = p.G * (int)slice / SK;
  const int last = p.G * (int)(slice + 1) / SK;
  const int end_row = p.first_row + p.M;

  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> xs((device bfloat*)X, dextents<int32_t, 2>(p.K, end_row));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)W, dextents<int32_t, 2>(GS, NT));
  auto LX = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
  auto P = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(LX)>, decltype(b0), float>();

  const auto home = P.get_multidimensional_index(ushort(0));
  const int c0 = home[0];
  const int r0 = home[1];

  float C[VALUES];
  UNROLL for (int i = 0; i < VALUES; i++) C[i] = 0.0f;

  // Prefetch first group's interleaved {scale, bias} words.
  uint4 next[SETS];
  UNROLL for (int s = 0; s < SETS; s++) {
    next[s] = *(const device uint4*)(blocks + (size_t)first * BLOCK + CODES + (c0 + 16 * s) * 4);
  }

  for (int g = first; g < last; g++) {
    float scale[SETS][4];
    float bias[SETS][4];
    UNROLL for (int s = 0; s < SETS; s++) {
      UNROLL for (int j = 0; j < 4; j++) {
        scale[s][j] = as_type<float>(next[s][j] << 16);
        bias[s][j] = as_type<float>(next[s][j] & 0xFFFF0000u);
      }
      next[s] = *(const device uint4*)(blocks + (size_t)(g + 1) * BLOCK + CODES + (c0 + 16 * s) * 4);
    }

    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)blocks + (size_t)g * BLOCK, dextents<int32_t, 2>(GS, NT));

    UNROLL for (int part = 0; part < RUNS / 2; part++) {
      auto a = xs.slice(g * GS, p.first_row + part * 16);
      LX.load(a);
      op.run(LX, b, P);
      const device float* gs = SUMS + (size_t)g * p.sums_stride + p.first_row + part * 16 + r0;
      const float xs0 = gs[0];
      const float xs1 = gs[8];
      UNROLL for (int s = 0; s < SETS; s++) {
        UNROLL for (int j = 0; j < 4; j++) {
          C[accIdx(part * 2, s, j)] = fma(scale[s][j], P[opPlace(0, s, j)], fma(bias[s][j], xs0, C[accIdx(part * 2, s, j)]));
          C[accIdx(part * 2 + 1, s, j)] = fma(scale[s][j], P[opPlace(1, s, j)], fma(bias[s][j], xs1, C[accIdx(part * 2 + 1, s, j)]));
        }
      }
    }
  }

  // Combine K-slices across the 4 simdgroups in threadgroup memory.
  threadgroup float partial[(SK - 1) * VALUES * 32];
  if (slice > 0) {
    UNROLL for (int i = 0; i < VALUES; i++) {
      partial[((slice - 1) * VALUES + i) * 32 + lane] = C[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (slice > 0) return;

  UNROLL for (int s = 0; s < SK - 1; s++) {
    UNROLL for (int i = 0; i < VALUES; i++) {
      C[i] += partial[(s * VALUES + i) * 32 + lane];
    }
  }

  const int n0 = (int)tile * NT + c0;
  UNROLL for (int k = 0; k < RUNS; k++) {
    const int m = p.first_row + r0 + 8 * k;
    if (m < end_row) {
      device bfloat* row = Y + (size_t)m * p.N;
      UNROLL for (int s = 0; s < SETS; s++) {
        const vec<bfloat, 4> four = vec<bfloat, 4>(float4(
            C[accIdx(k, s, 0)], C[accIdx(k, s, 1)], C[accIdx(k, s, 2)], C[accIdx(k, s, 3)]));
        *(device vec<bfloat, 4>*)(row + n0 + 16 * s) = four;
      }
    }
  }
}

template [[host_name("qk_16")]] kernel void qk_kernel<16>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint, uint, uint);
template [[host_name("qk_32")]] kernel void qk_kernel<32>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint, uint, uint);
template [[host_name("qk_48")]] kernel void qk_kernel<48>(
    const device bfloat*, const device uchar*, const device float*, device bfloat*, constant QkParams&, uint, uint, uint);

// Dedicated 1-row 4-bit quantized GEMV reading the exact same packed qk weight layout.
//
// 1 threadgroup = 4 simdgroups (128 threads, SK=4 K-slices) computing one full 32-col qk tile.
// Each lane processes 2 output columns (col_in_half and col_in_half + 16), reusing loaded X
// values 2x in registers and issuing two 128-bit uint4 coalesced weight bursts per group.
kernel void qmv_1(const device bfloat* X [[buffer(0)]],
                  const device uchar* W [[buffer(1)]],
                  device bfloat* Y [[buffer(2)]],
                  constant QkParams& p [[buffer(3)]],
                  uint tile [[threadgroup_position_in_grid]],
                  uint slice [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const uint col0 = lane >> 1; // 0..15
  const uint col1 = col0 + 16; // 16..31
  const uint half_idx = lane & 1u;

  const device uchar* tile_w = W + (size_t)tile * (size_t)p.G * BLOCK;
  const int g_first = p.G * (int)slice / SK;
  const int g_last = p.G * (int)(slice + 1) / SK;

  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int g = g_first; g < g_last; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float4 dot0 = float4(0.0f);
    float4 dot1 = float4(0.0f);
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;

      const uint wa = w4_0[wi];
      const float4 qa0 = float4(float(wa & 0xFu), float((wa >> 4) & 0xFu), float((wa >> 8) & 0xFu), float((wa >> 12) & 0xFu));
      const float4 qb0 = float4(float((wa >> 16) & 0xFu), float((wa >> 20) & 0xFu), float((wa >> 24) & 0xFu), float((wa >> 28) & 0xFu));
      dot0 = fma(qa0, xa, fma(qb0, xb, dot0));

      const uint wb = w4_1[wi];
      const float4 qa1 = float4(float(wb & 0xFu), float((wb >> 4) & 0xFu), float((wb >> 8) & 0xFu), float((wb >> 12) & 0xFu));
      const float4 qb1 = float4(float((wb >> 16) & 0xFu), float((wb >> 20) & 0xFu), float((wb >> 24) & 0xFu), float((wb >> 28) & 0xFu));
      dot1 = fma(qa1, xa, fma(qb1, xb, dot1));
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    const float d0 = (dot0[0] + dot0[1]) + (dot0[2] + dot0[3]);
    const float d1 = (dot1[0] + dot1[1]) + (dot1[2] + dot1[3]);

    acc0 = fma(as_type<float>(sb0 << 16), d0, fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), d1, fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  threadgroup float partial[SK][NT];
  if (half_idx == 0) {
    partial[slice][col0] = acc0;
    partial[slice][col1] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slice == 0) {
    const float total = (partial[0][lane] + partial[1][lane]) + (partial[2][lane] + partial[3][lane]);
    Y[base_col + lane] = bfloat(total);
  }
}

// Fused 1-row Gate+Up 4-bit GEMV + SiLU(gate)*up activation directly to act_buf.
// Reuses loaded X 4x across (gate[col0], gate[col1], up[col0], up[col1]) in registers,
// eliminating proj_big_buf DRAM round-trip and all gu_activate_sums dispatches/barriers!
// Grid: threadgroups (inter / 32, 1, 1) of 128 threads (SK = 4).
// p.N is `inter` (half of gate_up.n); up weights start at tile + (inter / 32).
kernel void qmv_1_silu_mul(const device bfloat* X [[buffer(0)]],
                           const device uchar* W [[buffer(1)]],
                           device bfloat* Y [[buffer(2)]],
                           constant QkParams& p [[buffer(3)]],
                           uint tile [[threadgroup_position_in_grid]],
                           uint slice [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const uint col0 = lane >> 1; // 0..15
  const uint col1 = col0 + 16; // 16..31
  const uint half_idx = lane & 1u;

  const size_t tile_bytes = (size_t)p.G * BLOCK;
  const uint up_tile = tile + (uint)(p.N / NT);
  const device uchar* gate_w = W + (size_t)tile * tile_bytes;
  const device uchar* up_w = W + (size_t)up_tile * tile_bytes;

  const int g_first = p.G * (int)slice / SK;
  const int g_last = p.G * (int)(slice + 1) / SK;

  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;

  float g_acc0 = 0.0f, g_acc1 = 0.0f;
  float u_acc0 = 0.0f, u_acc1 = 0.0f;

  for (int g = g_first; g < g_last; g++) {
    const device uchar* gb = gate_w + (size_t)g * BLOCK;
    const device uchar* ub = up_w + (size_t)g * BLOCK;

    const uint4 wg0 = *(const device uint4*)(gb + off0);
    const uint4 wg1 = *(const device uint4*)(gb + off1);
    const uint4 wu0 = *(const device uint4*)(ub + off0);
    const uint4 wu1 = *(const device uint4*)(ub + off1);

    const uint gsb0 = *(const device uint*)(gb + sb_off0);
    const uint gsb1 = *(const device uint*)(gb + sb_off1);
    const uint usb0 = *(const device uint*)(ub + sb_off0);
    const uint usb1 = *(const device uint*)(ub + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float4 gd0 = float4(0.0f), gd1 = float4(0.0f);
    float4 ud0 = float4(0.0f), ud1 = float4(0.0f);
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;

      const uint ga = wg0[wi];
      gd0 = fma(float4(float(ga & 0xFu), float((ga >> 4) & 0xFu), float((ga >> 8) & 0xFu), float((ga >> 12) & 0xFu)), xa,
                fma(float4(float((ga >> 16) & 0xFu), float((ga >> 20) & 0xFu), float((ga >> 24) & 0xFu), float((ga >> 28) & 0xFu)), xb, gd0));

      const uint gb_w = wg1[wi];
      gd1 = fma(float4(float(gb_w & 0xFu), float((gb_w >> 4) & 0xFu), float((gb_w >> 8) & 0xFu), float((gb_w >> 12) & 0xFu)), xa,
                fma(float4(float((gb_w >> 16) & 0xFu), float((gb_w >> 20) & 0xFu), float((gb_w >> 24) & 0xFu), float((gb_w >> 28) & 0xFu)), xb, gd1));

      const uint ua = wu0[wi];
      ud0 = fma(float4(float(ua & 0xFu), float((ua >> 4) & 0xFu), float((ua >> 8) & 0xFu), float((ua >> 12) & 0xFu)), xa,
                fma(float4(float((ua >> 16) & 0xFu), float((ua >> 20) & 0xFu), float((ua >> 24) & 0xFu), float((ua >> 28) & 0xFu)), xb, ud0));

      const uint ub_w = wu1[wi];
      ud1 = fma(float4(float(ub_w & 0xFu), float((ub_w >> 4) & 0xFu), float((ub_w >> 8) & 0xFu), float((ub_w >> 12) & 0xFu)), xa,
                fma(float4(float((ub_w >> 16) & 0xFu), float((ub_w >> 20) & 0xFu), float((ub_w >> 24) & 0xFu), float((ub_w >> 28) & 0xFu)), xb, ud1));
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    g_acc0 = fma(as_type<float>(gsb0 << 16), (gd0[0] + gd0[1]) + (gd0[2] + gd0[3]), fma(as_type<float>(gsb0 & 0xFFFF0000u), xsum, g_acc0));
    g_acc1 = fma(as_type<float>(gsb1 << 16), (gd1[0] + gd1[1]) + (gd1[2] + gd1[3]), fma(as_type<float>(gsb1 & 0xFFFF0000u), xsum, g_acc1));
    u_acc0 = fma(as_type<float>(usb0 << 16), (ud0[0] + ud0[1]) + (ud0[2] + ud0[3]), fma(as_type<float>(usb0 & 0xFFFF0000u), xsum, u_acc0));
    u_acc1 = fma(as_type<float>(usb1 << 16), (ud1[0] + ud1[1]) + (ud1[2] + ud1[3]), fma(as_type<float>(usb1 & 0xFFFF0000u), xsum, u_acc1));
  }

  g_acc0 += simd_shuffle_xor(g_acc0, 1);
  g_acc1 += simd_shuffle_xor(g_acc1, 1);
  u_acc0 += simd_shuffle_xor(u_acc0, 1);
  u_acc1 += simd_shuffle_xor(u_acc1, 1);

  threadgroup float pg[SK][NT];
  threadgroup float pu[SK][NT];
  if (half_idx == 0) {
    pg[slice][col0] = g_acc0;
    pg[slice][col1] = g_acc1;
    pu[slice][col0] = u_acc0;
    pu[slice][col1] = u_acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slice == 0) {
    const float g_val = float(bfloat((pg[0][lane] + pg[1][lane]) + (pg[2][lane] + pg[3][lane])));
    const float u_val = float(bfloat((pu[0][lane] + pu[1][lane]) + (pu[2][lane] + pu[3][lane])));
    const float act = (g_val / (1.0f + exp(-g_val))) * u_val;
    Y[base_col + lane] = bfloat(act);
  }
}

// Fused 1-row LM Head 4-bit GEMV + Stage-1 Argmax across each 32-vocab tile.
// Completely eliminates writing & reading 496 KB of bf16 logits and the argmax_stage1 dispatch!
// Grid: threadgroups (vocab_tiles, 1, 1) of 128 threads (SK = 4).
kernel void qmv_1_argmax(const device bfloat* X [[buffer(0)]],
                         const device uchar* W [[buffer(1)]],
                         device float* chunk_val [[buffer(2)]],
                         device uint* chunk_idx [[buffer(3)]],
                         constant QkParams& p [[buffer(4)]],
                         uint tile [[threadgroup_position_in_grid]],
                         uint slice [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint base_col = tile * NT;
  if ((int)base_col >= p.N) return;

  const uint col0 = lane >> 1;
  const uint col1 = col0 + 16;
  const uint half_idx = lane & 1u;

  const device uchar* tile_w = W + (size_t)tile * (size_t)p.G * BLOCK;
  const int g_first = p.G * (int)slice / SK;
  const int g_last = p.G * (int)(slice + 1) / SK;

  const size_t off0 = (size_t)col0 * 32 + (size_t)half_idx * 16;
  const size_t off1 = (size_t)col1 * 32 + (size_t)half_idx * 16;
  const size_t sb_off0 = CODES + (size_t)col0 * 4;
  const size_t sb_off1 = CODES + (size_t)col1 * 4;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int g = g_first; g < g_last; g++) {
    const device uchar* blk = tile_w + (size_t)g * BLOCK;
    const uint4 w4_0 = *(const device uint4*)(blk + off0);
    const uint4 w4_1 = *(const device uint4*)(blk + off1);
    const uint sb0 = *(const device uint*)(blk + sb_off0);
    const uint sb1 = *(const device uint*)(blk + sb_off1);

    const device vec<bfloat, 4>* xp = (const device vec<bfloat, 4>*)(X + g * GS + half_idx * 32);
    float4 dot0 = float4(0.0f);
    float4 dot1 = float4(0.0f);
    float4 sum4 = float4(0.0f);

    UNROLL for (int wi = 0; wi < 4; wi++) {
      const float4 xa = float4(xp[wi * 2]);
      const float4 xb = float4(xp[wi * 2 + 1]);
      sum4 += xa + xb;

      const uint wa = w4_0[wi];
      dot0 = fma(float4(float(wa & 0xFu), float((wa >> 4) & 0xFu), float((wa >> 8) & 0xFu), float((wa >> 12) & 0xFu)), xa,
                 fma(float4(float((wa >> 16) & 0xFu), float((wa >> 20) & 0xFu), float((wa >> 24) & 0xFu), float((wa >> 28) & 0xFu)), xb, dot0));

      const uint wb = w4_1[wi];
      dot1 = fma(float4(float(wb & 0xFu), float((wb >> 4) & 0xFu), float((wb >> 8) & 0xFu), float((wb >> 12) & 0xFu)), xa,
                 fma(float4(float((wb >> 16) & 0xFu), float((wb >> 20) & 0xFu), float((wb >> 24) & 0xFu), float((wb >> 28) & 0xFu)), xb, dot1));
    }

    const float xsum = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    acc0 = fma(as_type<float>(sb0 << 16), (dot0[0] + dot0[1]) + (dot0[2] + dot0[3]), fma(as_type<float>(sb0 & 0xFFFF0000u), xsum, acc0));
    acc1 = fma(as_type<float>(sb1 << 16), (dot1[0] + dot1[1]) + (dot1[2] + dot1[3]), fma(as_type<float>(sb1 & 0xFFFF0000u), xsum, acc1));
  }

  acc0 += simd_shuffle_xor(acc0, 1);
  acc1 += simd_shuffle_xor(acc1, 1);

  threadgroup float partial[SK][NT];
  if (half_idx == 0) {
    partial[slice][col0] = acc0;
    partial[slice][col1] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (slice == 0) {
    const uint idx = base_col + lane;
    const float raw = float(bfloat((partial[0][lane] + partial[1][lane]) + (partial[2][lane] + partial[3][lane])));
    float best_v = ((int)idx < p.N && isfinite(raw)) ? raw : -INFINITY;
    uint best_i = ((int)idx < p.N && isfinite(raw)) ? idx : 0xFFFFFFFFu;

    UNROLL for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(best_v, off);
      const uint oi = simd_shuffle_down(best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0) {
      chunk_val[tile] = best_v;
      chunk_idx[tile] = best_i;
    }
  }
}
