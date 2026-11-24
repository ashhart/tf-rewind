// Clean-room fused embedding, RMSNorm + group-sums, Residual-Add + RMSNorm + group-sums,
// and GDN per-head RMSNorm + SiLU gate + group-sums kernels.
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

struct EmbedParams {
  uint rows;
  uint hidden;
};

// Dequantize 4-bit affine group-64 embeddings for `rows` tokens.
// Grid: threads (hidden / 8, rows) — each thread unpacks 8 elements (one uint32 word).
kernel void embed_lookup(const device uint* tokens [[buffer(0)]],
                         const device uint* codes [[buffer(1)]],
                         const device bfloat* scales [[buffer(2)]],
                         const device bfloat* biases [[buffer(3)]],
                         device bfloat* out [[buffer(4)]],
                         constant EmbedParams& p [[buffer(5)]],
                         uint2 tid [[thread_position_in_grid]]) {
  const uint word_idx = tid.x;
  const uint row = tid.y;
  const uint words_per_row = p.hidden >> 3;
  if (word_idx >= words_per_row || row >= p.rows) return;

  const uint tok = tokens[row];
  const uint g = word_idx >> 3; // 8 words = 64 elements = 1 quant group
  const uint groups = p.hidden >> 6;
  const uint packed = codes[(size_t)tok * words_per_row + word_idx];
  const float scale = float(scales[(size_t)tok * groups + g]);
  const float bias = float(biases[(size_t)tok * groups + g]);

  device bfloat* dst = out + (size_t)row * p.hidden + word_idx * 8;
  UNROLL for (uint i = 0; i < 8; i++) {
    const uint c = (packed >> (i * 4)) & 0xFu;
    dst[i] = bfloat(fma(float(c), scale, bias));
  }
}

struct NormParams {
  uint rows;
  uint hidden;
  uint sums_stride;
  float eps;
};

// Fused RMSNorm + per-64-group x-sums (`SUMS[g * sums_stride + row]`).
// 1 threadgroup (256 threads = 8 simdgroups) per row.
kernel void rms_norm_sums(const device bfloat* x [[buffer(0)]],
                          const device bfloat* weight [[buffer(1)]],
                          device bfloat* out [[buffer(2)]],
                          device float* sums [[buffer(3)]],
                          constant NormParams& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  const device bfloat* xr = x + (size_t)row * p.hidden;
  device bfloat* yr = out + (size_t)row * p.hidden;
  const uint groups = p.hidden >> 6;

  float sq = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const float v0 = float(xr[base]);
    const float v1 = float(xr[base + 32]);
    sq = fma(v0, v0, fma(v1, v1, sq));
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float total_sq = 0.0f;
  if (sg == 0) {
    float v = lane < 8 ? sg_sq[lane] : 0.0f;
    total_sq = simd_sum(v);
    if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.hidden) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat y0 = bfloat(float(xr[base]) * inv * float(weight[base]));
    const bfloat y1 = bfloat(float(xr[base + 32]) * inv * float(weight[base + 32]));
    yr[base] = y0;
    yr[base + 32] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
  }
}

// Fused in-place Residual Add (`h += delta`) + RMSNorm + per-64-group x-sums.
// Single-pass register-resident path for hidden <= 8192 eliminates reloading `h` from memory!
kernel void add_rms_norm_sums(device bfloat* h [[buffer(0)]],
                              const device bfloat* delta [[buffer(1)]],
                              const device bfloat* weight [[buffer(2)]],
                              device bfloat* out [[buffer(3)]],
                              device float* sums [[buffer(4)]],
                              constant NormParams& p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  device bfloat* hr = h + (size_t)row * p.hidden;
  const device bfloat* dr = delta + (size_t)row * p.hidden;
  device bfloat* yr = out + (size_t)row * p.hidden;
  const uint groups = p.hidden >> 6;

  // Fast register-resident single-pass path (covers hidden <= 8192: 0.8B/2B/27B/70B!)
  if (groups <= 128) {
    float2 reg[16]; // at most 128/8 = 16 groups per simdgroup, 2 elements per lane per group
    float sq = 0.0f;
    uint k = 0;
    for (uint g = sg; g < groups; g += 8, k++) {
      const uint base = g * 64 + lane;
      const bfloat v0 = bfloat(float(hr[base]) + float(dr[base]));
      const bfloat v1 = bfloat(float(hr[base + 32]) + float(dr[base + 32]));
      hr[base] = v0;
      hr[base + 32] = v1;
      const float f0 = float(v0);
      const float f1 = float(v1);
      reg[k] = float2(f0, f1);
      sq = fma(f0, f0, fma(f1, f1, sq));
    }
    sq = simd_sum(sq);
    threadgroup float sg_sq[8];
    if (lane == 0) sg_sq[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      float v = lane < 8 ? sg_sq[lane] : 0.0f;
      float total_sq = simd_sum(v);
      if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.hidden) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = sg_sq[0];

    k = 0;
    for (uint g = sg; g < groups; g += 8, k++) {
      const uint base = g * 64 + lane;
      const bfloat y0 = bfloat(reg[k].x * inv * float(weight[base]));
      const bfloat y1 = bfloat(reg[k].y * inv * float(weight[base + 32]));
      yr[base] = y0;
      yr[base + 32] = y1;
      if (p.sums_stride != 0) {
        const float gs = simd_sum(float(y0) + float(y1));
        if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
      }
    }
    return;
  }

  float sq = 0.0f;
  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat v0 = bfloat(float(hr[base]) + float(dr[base]));
    const bfloat v1 = bfloat(float(hr[base + 32]) + float(dr[base + 32]));
    hr[base] = v0;
    hr[base + 32] = v1;
    const float f0 = float(v0);
    const float f1 = float(v1);
    sq = fma(f0, f0, fma(f1, f1, sq));
  }
  sq = simd_sum(sq);
  threadgroup float sg_sq[8];
  if (lane == 0) sg_sq[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float v = lane < 8 ? sg_sq[lane] : 0.0f;
    float total_sq = simd_sum(v);
    if (lane == 0) sg_sq[0] = rsqrt(total_sq / float(p.hidden) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sg_sq[0];

  for (uint g = sg; g < groups; g += 8) {
    const uint base = g * 64 + lane;
    const bfloat y0 = bfloat(float(hr[base]) * inv * float(weight[base]));
    const bfloat y1 = bfloat(float(hr[base + 32]) * inv * float(weight[base + 32]));
    yr[base] = y0;
    yr[base + 32] = y1;
    if (p.sums_stride != 0) {
      const float gs = simd_sum(float(y0) + float(y1));
      if (lane == 0) sums[(size_t)g * p.sums_stride + row] = gs;
    }
  }
}

struct GdnNormParams {
  uint rows;
  uint nv;
  uint dv;             // 128
  uint in_proj_stride; // stride of fused in_proj_all buffer
  uint z_offset;       // starting column of Z inside in_proj_all
  uint sums_stride;
  float eps;
};

// Fused GDN per-head RMSNorm (DV=128) + SiLU(z) output gate + out_proj group sums.
// Grid: threadgroups (nv, rows) of 32 threads (1 simdgroup per head/row, zero TG barriers).
kernel void gdn_norm_gate_sums(const device bfloat* scan_out [[buffer(0)]],
                               const device bfloat* in_proj_all [[buffer(1)]],
                               const device bfloat* norm_w [[buffer(2)]],
                               device bfloat* out [[buffer(3)]],
                               device float* sums [[buffer(4)]],
                               constant GdnNormParams& p [[buffer(5)]],
                               uint2 grp [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.nv || row >= p.rows) return;

  // Each lane holds 2 elements from group 0 ([0..63]) and 2 elements from group 1 ([64..127]).
  const uint base = (row * p.nv + head) * 128;
  const uint z_base = row * p.in_proj_stride + p.z_offset + head * 128;

  const float x0 = float(scan_out[base + lane]);
  const float x1 = float(scan_out[base + 32 + lane]);
  const float x2 = float(scan_out[base + 64 + lane]);
  const float x3 = float(scan_out[base + 96 + lane]);

  const float sq = simd_sum(fma(x0, x0, fma(x1, x1, fma(x2, x2, x3 * x3))));
  const float inv = rsqrt(sq / 128.0f + p.eps);

  const float z0 = float(in_proj_all[z_base + lane]);
  const float z1 = float(in_proj_all[z_base + 32 + lane]);
  const float z2 = float(in_proj_all[z_base + 64 + lane]);
  const float z3 = float(in_proj_all[z_base + 96 + lane]);

  const float s0 = z0 / (1.0f + exp(-z0));
  const float s1 = z1 / (1.0f + exp(-z1));
  const float s2 = z2 / (1.0f + exp(-z2));
  const float s3 = z3 / (1.0f + exp(-z3));

  const bfloat y0 = bfloat((x0 * inv * float(norm_w[lane])) * s0);
  const bfloat y1 = bfloat((x1 * inv * float(norm_w[32 + lane])) * s1);
  const bfloat y2 = bfloat((x2 * inv * float(norm_w[64 + lane])) * s2);
  const bfloat y3 = bfloat((x3 * inv * float(norm_w[96 + lane])) * s3);

  out[base + lane] = y0;
  out[base + 32 + lane] = y1;
  out[base + 64 + lane] = y2;
  out[base + 96 + lane] = y3;

  const float gs0 = simd_sum(float(y0) + float(y1));
  const float gs1 = simd_sum(float(y2) + float(y3));
  if (lane == 0) {
    sums[(size_t)(head * 2) * p.sums_stride + row] = gs0;
    sums[(size_t)(head * 2 + 1) * p.sums_stride + row] = gs1;
  }
}
