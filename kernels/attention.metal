// Clean-room fused GQA Attention kernels for Qwen3.5 / Qwen3.8 (head_dim = 256, rotary_dim = 64).
//
// Lane-to-head-dim mapping inside each 32-lane simdgroup:
//   For group g in {0, 1, 2, 3}: lane l holds (g*64 + l, g*64 + 32 + l).
//   - Coalesced 64-byte reads/writes across all 32 lanes.
//   - Group 0 (l, l+32) IS the exact split-half partial RoPE pair (zero shuffles!).
//   - Each group g (0..3) IS an exact 64-element o_proj quant group (zero shuffles for group sums!).
#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint HD = 256;

struct AttnPrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride; // 2*H*256 + 2*KVH*256
  float eps;
};

// Fused Q/K RMSNorm + split-half partial RoPE (dims 0..64) + KV cache store.
// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_qk_rope_store(const device bfloat* qkv_buf [[buffer(0)]],
                               const device bfloat* q_norm [[buffer(1)]],
                               const device bfloat* k_norm [[buffer(2)]],
                               const device float* inv_freq [[buffer(3)]], // [32] f32 precomputed from f64
                               const device uint* positions [[buffer(4)]],
                               const device uint* kv_slots [[buffer(5)]],
                               device bfloat* q_out [[buffer(6)]],
                               device bfloat* k_cache [[buffer(7)]],
                               device bfloat* v_cache [[buffer(8)]],
                               constant AttnPrepParams& p [[buffer(9)]],
                               uint2 grp [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;
  const float pos = float(positions[row]);
  const float angle = pos * inv_freq[lane];
  const float c = cos(angle);
  const float s = sin(angle);

  if (idx < p.num_heads) {
    // Query head h: Q occupies first 256 of each 512-element [Q, gate] head slice.
    const device bfloat* src = row_ptr + (size_t)idx * (2 * HD);
    float v[8];
    float sq = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      const float a = float(src[g * 64 + lane]);
      const float b = float(src[g * 64 + 32 + lane]);
      v[g * 2] = a;
      v[g * 2 + 1] = b;
      sq = fma(a, a, fma(b, b, sq));
    }
    const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
    UNROLL for (uint g = 0; g < 4; g++) {
      v[g * 2] = v[g * 2] * inv * float(q_norm[g * 64 + lane]);
      v[g * 2 + 1] = v[g * 2 + 1] * inv * float(q_norm[g * 64 + 32 + lane]);
    }
    // Partial RoPE on group 0 (0..63)
    const float r0 = v[0] * c - v[1] * s;
    const float r1 = v[1] * c + v[0] * s;
    v[0] = r0;
    v[1] = r1;

    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * HD;
    UNROLL for (uint g = 0; g < 4; g++) {
      dst[g * 64 + lane] = bfloat(v[g * 2]);
      dst[g * 64 + 32 + lane] = bfloat(v[g * 2 + 1]);
    }
    return;
  }

  // KV head kh: normalize + RoPE K, and copy V directly into KV cache at kv_slots[row].
  const uint kh = idx - p.num_heads;
  const uint slot = kv_slots[row];
  const size_t k_off = (size_t)(2 * p.num_heads) * HD + (size_t)kh * HD;
  const size_t v_off = k_off + (size_t)p.num_kv_heads * HD;
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float kv[8];
  float sq = 0.0f;
  UNROLL for (uint g = 0; g < 4; g++) {
    const float a = float(k_src[g * 64 + lane]);
    const float b = float(k_src[g * 64 + 32 + lane]);
    kv[g * 2] = a;
    kv[g * 2 + 1] = b;
    sq = fma(a, a, fma(b, b, sq));
  }
  const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
  UNROLL for (uint g = 0; g < 4; g++) {
    kv[g * 2] = kv[g * 2] * inv * float(k_norm[g * 64 + lane]);
    kv[g * 2 + 1] = kv[g * 2 + 1] * inv * float(k_norm[g * 64 + 32 + lane]);
  }
  const float rk0 = kv[0] * c - kv[1] * s;
  const float rk1 = kv[1] * c + kv[0] * s;
  kv[0] = rk0;
  kv[1] = rk1;

  const size_t cache_base = (size_t)(slot * p.num_kv_heads + kh) * HD;
  device bfloat* k_dst = k_cache + cache_base;
  device bfloat* v_dst = v_cache + cache_base;
  UNROLL for (uint g = 0; g < 4; g++) {
    k_dst[g * 64 + lane] = bfloat(kv[g * 2]);
    k_dst[g * 64 + 32 + lane] = bfloat(kv[g * 2 + 1]);
    v_dst[g * 64 + lane] = v_src[g * 64 + lane];
    v_dst[g * 64 + 32 + lane] = v_src[g * 64 + 32 + lane];
  }
}

struct AttnFlashParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint sums_stride;
  float scale; // 1 / sqrt(256) = 0.0625
  uint splits; // used by split-K path
  uint base_pos; // committed KV prefix length; tree rows start at base_pos
  uint is_tree;  // 1 when tree_mask filters slots >= base_pos
};

// Single-stage fused Flash GQA + sigmoid(gate) + o_proj group sums.
// Supports both causal chains (is_tree=0) and arbitrary speculative trees up to 128 rows (is_tree=1).
// Grid: threadgroups (num_heads, rows) of 128 threads.
kernel void attn_flash_fused(const device bfloat* q_in [[buffer(0)]],
                             const device bfloat* qkv_buf [[buffer(1)]],
                             const device bfloat* k_cache [[buffer(2)]],
                             const device bfloat* v_cache [[buffer(3)]],
                             const device uint* seq_lens [[buffer(4)]], // inclusive KV scan bound per row
                             const device uint4* tree_masks [[buffer(5)]], // 128-bit ancestor bitmask per row
                             device bfloat* out [[buffer(6)]],
                             device float* sums [[buffer(7)]],
                             constant AttnFlashParams& p [[buffer(8)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const uint kv_len = seq_lens[row];
  const uint4 tmask = (p.is_tree != 0) ? tree_masks[row] : uint4(0xFFFFFFFFu);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * HD;

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = float(q_ptr[g * 64 + lane]) * p.scale;
    q[g * 2 + 1] = float(q_ptr[g * 64 + 32 + lane]) * p.scale;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;

  const uint prefix_end = (p.is_tree != 0) ? min(kv_len, p.base_pos) : kv_len;

  // Phase 1: Committed prefix [0 .. prefix_end), assigned identically to sequential decode (t % 4 == sg).
  for (uint t = sg; t < prefix_end; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  // Phase 2 (trees only): Walk topological ancestors in exact depth order, assigning depth d to
  // simdgroup ((base_pos + d) & 3) so fp32 online-softmax reduction is 100% bit-identical to c=1 decode!
  if (p.is_tree != 0 && kv_len > p.base_pos) {
    const uint rel_limit = kv_len - p.base_pos;
    uint depth = 0;
    for (uint rel = 0; rel < rel_limit; rel++) {
      if (((tmask[rel >> 5] >> (rel & 31u)) & 1u) == 0u) continue;
      if (((p.base_pos + depth) & 3u) == sg) {
        const uint t = p.base_pos + rel;
        const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
        float dot_part = 0.0f;
        UNROLL for (uint g = 0; g < 4; g++) {
          dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
          dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
        }
        const float score = simd_sum(dot_part);
        const float new_m = max(m_val, score);
        const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
        const float prob = exp(score - new_m);
        l_val = fma(l_val, alpha, prob);
        m_val = new_m;

        const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
        UNROLL for (uint g = 0; g < 4; g++) {
          acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
          acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
        }
      }
      depth++;
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg != 0) return;

  float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  // Apply sigmoid(gate) from qkv_buf[row, head*512 + 256 .. +512] and compute 4 group sums.
  const device bfloat* gate_ptr = qkv_buf + (size_t)row * p.qkv_stride + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    o0 *= inv_l;
    o1 *= inv_l;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    const bfloat y0 = bfloat(o0 * g0);
    const bfloat y1 = bfloat(o1 * g1);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 4 + g) * p.sums_stride + row] = gs;
    }
  }
}


// Split-K Stage 1 for long contexts: Grid threadgroups (splits, num_heads, rows) of 128 threads.
kernel void attn_flash_split(const device bfloat* q_in [[buffer(0)]],
                             const device bfloat* k_cache [[buffer(1)]],
                             const device bfloat* v_cache [[buffer(2)]],
                             const device uint* seq_lens [[buffer(3)]],
                             device float* partial_acc [[buffer(4)]], // [rows][H][splits][256]
                             device float2* partial_ml [[buffer(5)]], // [rows][H][splits]
                             constant AttnFlashParams& p [[buffer(6)]],
                             uint3 grp [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint split = grp.x;
  const uint head = grp.y;
  const uint row = grp.z;
  if (split >= p.splits || head >= p.num_heads || row >= p.rows) return;

  const uint kv_len = seq_lens[row];
  const uint chunk = (kv_len + p.splits - 1) / p.splits;
  const uint t0 = split * chunk;
  const uint t1 = min(kv_len, t0 + chunk);

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * HD;

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = float(q_ptr[g * 64 + lane]) * p.scale;
    q[g * 2 + 1] = float(q_ptr[g * 64 + 32 + lane]) * p.scale;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;

  for (uint t = t0 + sg; t < t1; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }

  const size_t slot = ((size_t)row * p.num_heads + head) * p.splits + split;
  if (lane == 0) partial_ml[slot] = float2(gm, gl);
  device float* dst = partial_acc + slot * HD;
  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    dst[i0] = o0;
    dst[i1] = o1;
  }
}

// Split-K Stage 2: merge splits + sigmoid(gate) + o_proj group sums.
// Grid: threadgroups (num_heads, rows) of 32 threads.
kernel void attn_merge_gate_sums(const device float* partial_acc [[buffer(0)]],
                                 const device float2* partial_ml [[buffer(1)]],
                                 const device bfloat* qkv_buf [[buffer(2)]],
                                 device bfloat* out [[buffer(3)]],
                                 device float* sums [[buffer(4)]],
                                 constant AttnFlashParams& p [[buffer(5)]],
                                 uint2 grp [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const size_t base_slot = ((size_t)row * p.num_heads + head) * p.splits;
  float gm = -INFINITY;
  for (uint s = 0; s < p.splits; s++) {
    const float2 ml = partial_ml[base_slot + s];
    if (ml.y > 0.0f) gm = max(gm, ml.x);
  }
  float gl = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  for (uint s = 0; s < p.splits; s++) {
    const float2 ml = partial_ml[base_slot + s];
    if (ml.y <= 0.0f) continue;
    const float w = exp(ml.x - gm);
    gl = fma(ml.y, w, gl);
    const device float* pa = partial_acc + (base_slot + s) * HD;
    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(pa[g * 64 + lane], w, acc[g * 2]);
      acc[g * 2 + 1] = fma(pa[g * 64 + 32 + lane], w, acc[g * 2 + 1]);
    }
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  const device bfloat* gate_ptr = qkv_buf + (size_t)row * p.qkv_stride + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    const bfloat y0 = bfloat((acc[g * 2] * inv_l) * g0);
    const bfloat y1 = bfloat((acc[g * 2 + 1] * inv_l) * g1);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 4 + g) * p.sums_stride + row] = gs;
    }
  }
}


struct KvCompactParams {
  uint base_pos;
  uint keep_len;
  uint row_u4; // (num_kv_heads * 256 * 2) / 16 = num_kv_heads * 32
};

// Stage 1: Gather accepted non-contiguous tree KV rows into contiguous tmp buffer.
// Grid: threads (row_u4, keep_len).
kernel void kv_compact_gather(const device uint4* k_cache [[buffer(0)]],
                              const device uint4* v_cache [[buffer(1)]],
                              const device uint* keep_rows [[buffer(2)]],
                              device uint4* k_tmp [[buffer(3)]],
                              device uint4* v_tmp [[buffer(4)]],
                              constant KvCompactParams& p [[buffer(5)]],
                              uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= p.row_u4 || tid.y >= p.keep_len) return;
  const uint src_slot = p.base_pos + keep_rows[tid.y];
  const size_t src_off = (size_t)src_slot * p.row_u4 + tid.x;
  const size_t dst_off = (size_t)tid.y * p.row_u4 + tid.x;
  k_tmp[dst_off] = k_cache[src_off];
  v_tmp[dst_off] = v_cache[src_off];
}

// Stage 2: Write contiguous tmp rows back to k_cache/v_cache at [base_pos .. base_pos + keep_len).
kernel void kv_compact_scatter(const device uint4* k_tmp [[buffer(0)]],
                               const device uint4* v_tmp [[buffer(1)]],
                               device uint4* k_cache [[buffer(2)]],
                               device uint4* v_cache [[buffer(3)]],
                               constant KvCompactParams& p [[buffer(4)]],
                               uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= p.row_u4 || tid.y >= p.keep_len) return;
  const size_t src_off = (size_t)tid.y * p.row_u4 + tid.x;
  const size_t dst_off = (size_t)(p.base_pos + tid.y) * p.row_u4 + tid.x;
  k_cache[dst_off] = k_tmp[src_off];
  v_cache[dst_off] = v_tmp[src_off];
}

struct ForkPrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint suffix_stream_stride; // stride in bfloat elements between streams' suffix KV buffers
  float eps;
};

// Forked Q/K RMSNorm + partial RoPE + store K/V into per-stream suffix KV buffer.
// Works for both per-stream delta prefill (all rows same stream) and lockstep B-stream decode.
// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_qk_rope_store_forked(const device bfloat* qkv_buf [[buffer(0)]],
                                      const device bfloat* q_norm [[buffer(1)]],
                                      const device bfloat* k_norm [[buffer(2)]],
                                      const device float* inv_freq [[buffer(3)]],
                                      const device uint* positions [[buffer(4)]],
                                      const device uint* suffix_slots [[buffer(5)]],
                                      const device uint* row_streams [[buffer(6)]],
                                      device bfloat* q_out [[buffer(7)]],
                                      device bfloat* k_suffix [[buffer(8)]],
                                      device bfloat* v_suffix [[buffer(9)]],
                                      constant ForkPrepParams& p [[buffer(10)]],
                                      uint2 grp [[threadgroup_position_in_grid]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;
  const float pos = float(positions[row]);
  const float angle = pos * inv_freq[lane];
  const float c = cos(angle);
  const float s = sin(angle);

  if (idx < p.num_heads) {
    const device bfloat* src = row_ptr + (size_t)idx * (2 * HD);
    float v[8];
    float sq = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      const float a = float(src[g * 64 + lane]);
      const float b = float(src[g * 64 + 32 + lane]);
      v[g * 2] = a;
      v[g * 2 + 1] = b;
      sq = fma(a, a, fma(b, b, sq));
    }
    const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
    UNROLL for (uint g = 0; g < 4; g++) {
      v[g * 2] = v[g * 2] * inv * float(q_norm[g * 64 + lane]);
      v[g * 2 + 1] = v[g * 2 + 1] * inv * float(q_norm[g * 64 + 32 + lane]);
    }
    const float r0 = v[0] * c - v[1] * s;
    const float r1 = v[1] * c + v[0] * s;
    v[0] = r0;
    v[1] = r1;

    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * HD;
    UNROLL for (uint g = 0; g < 4; g++) {
      dst[g * 64 + lane] = bfloat(v[g * 2]);
      dst[g * 64 + 32 + lane] = bfloat(v[g * 2 + 1]);
    }
    return;
  }

  const uint kh = idx - p.num_heads;
  const uint sid = row_streams[row];
  const uint slot = suffix_slots[row];
  const size_t k_off = (size_t)(2 * p.num_heads) * HD + (size_t)kh * HD;
  const size_t v_off = k_off + (size_t)p.num_kv_heads * HD;
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float kv[8];
  float sq = 0.0f;
  UNROLL for (uint g = 0; g < 4; g++) {
    const float a = float(k_src[g * 64 + lane]);
    const float b = float(k_src[g * 64 + 32 + lane]);
    kv[g * 2] = a;
    kv[g * 2 + 1] = b;
    sq = fma(a, a, fma(b, b, sq));
  }
  const float inv = rsqrt(simd_sum(sq) / float(HD) + p.eps);
  UNROLL for (uint g = 0; g < 4; g++) {
    kv[g * 2] = kv[g * 2] * inv * float(k_norm[g * 64 + lane]);
    kv[g * 2 + 1] = kv[g * 2 + 1] * inv * float(k_norm[g * 64 + 32 + lane]);
  }
  const float rk0 = kv[0] * c - kv[1] * s;
  const float rk1 = kv[1] * c + kv[0] * s;
  kv[0] = rk0;
  kv[1] = rk1;

  const size_t cache_base = (size_t)sid * p.suffix_stream_stride + (size_t)(slot * p.num_kv_heads + kh) * HD;
  device bfloat* k_dst = k_suffix + cache_base;
  device bfloat* v_dst = v_suffix + cache_base;
  UNROLL for (uint g = 0; g < 4; g++) {
    k_dst[g * 64 + lane] = bfloat(kv[g * 2]);
    k_dst[g * 64 + 32 + lane] = bfloat(kv[g * 2 + 1]);
    v_dst[g * 64 + lane] = v_src[g * 64 + lane];
    v_dst[g * 64 + 32 + lane] = v_src[g * 64 + 32 + lane];
  }
}

struct ForkFlashParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint sums_stride;
  uint prefix_len;
  uint suffix_stream_stride;
  float scale;
};

// Zero-copy forked Flash GQA: attends to read-only shared prefix KV [0..prefix_len) PLUS
// per-stream suffix KV [0..suffix_lens[row]) with fused sigmoid(gate) + o_proj group sums.
// Grid: threadgroups (num_heads, rows) of 128 threads.
kernel void attn_flash_forked(const device bfloat* q_in [[buffer(0)]],
                              const device bfloat* qkv_buf [[buffer(1)]],
                              const device bfloat* k_prefix [[buffer(2)]],
                              const device bfloat* v_prefix [[buffer(3)]],
                              const device bfloat* k_suffix [[buffer(4)]],
                              const device bfloat* v_suffix [[buffer(5)]],
                              const device uint* suffix_lens [[buffer(6)]],
                              const device uint* row_streams [[buffer(7)]],
                              device bfloat* out [[buffer(8)]],
                              device float* sums [[buffer(9)]],
                              constant ForkFlashParams& p [[buffer(10)]],
                              uint2 grp [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint sid = row_streams[row];
  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const uint s_len = suffix_lens[row];
  const uint total_len = p.prefix_len + s_len;
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * HD;

  float q[8];
  UNROLL for (uint g = 0; g < 4; g++) {
    q[g * 2] = float(q_ptr[g * 64 + lane]) * p.scale;
    q[g * 2 + 1] = float(q_ptr[g * 64 + 32 + lane]) * p.scale;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float acc[8];
  UNROLL for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * HD;
  const size_t kh_off = (size_t)kv_head * HD;
  const device bfloat* ks_base = k_suffix + (size_t)sid * p.suffix_stream_stride;
  const device bfloat* vs_base = v_suffix + (size_t)sid * p.suffix_stream_stride;

  for (uint t = sg; t < total_len; t += 4) {
    const device bfloat* kp;
    const device bfloat* vp;
    if (t < p.prefix_len) {
      kp = k_prefix + (size_t)t * kv_stride + kh_off;
      vp = v_prefix + (size_t)t * kv_stride + kh_off;
    } else {
      const uint st = t - p.prefix_len;
      kp = ks_base + (size_t)st * kv_stride + kh_off;
      vp = vs_base + (size_t)st * kv_stride + kh_off;
    }
    float dot_part = 0.0f;
    UNROLL for (uint g = 0; g < 4; g++) {
      dot_part = fma(q[g * 2], float(kp[g * 64 + lane]), dot_part);
      dot_part = fma(q[g * 2 + 1], float(kp[g * 64 + 32 + lane]), dot_part);
    }
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    UNROLL for (uint g = 0; g < 4; g++) {
      acc[g * 2] = fma(acc[g * 2], alpha, prob * float(vp[g * 64 + lane]));
      acc[g * 2 + 1] = fma(acc[g * 2 + 1], alpha, prob * float(vp[g * 64 + 32 + lane]));
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][HD];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  UNROLL for (uint g = 0; g < 4; g++) {
    tg_acc[sg][g * 64 + lane] = acc[g * 2];
    tg_acc[sg][g * 64 + 32 + lane] = acc[g * 2 + 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  const device bfloat* gate_ptr = qkv_buf + (size_t)row * p.qkv_stride + (size_t)head * (2 * HD) + HD;
  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * HD;

  UNROLL for (uint g = 0; g < 4; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    o0 *= inv_l;
    o1 *= inv_l;
    const float g0 = 1.0f / (1.0f + exp(-float(gate_ptr[i0])));
    const float g1 = 1.0f / (1.0f + exp(-float(gate_ptr[i1])));
    const bfloat y0 = bfloat(o0 * g0);
    const bfloat y1 = bfloat(o1 * g1);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 4 + g) * p.sums_stride + row] = gs;
    }
  }
}

// ============================================================================
// Standard Dense Transformer GQA kernels (head_dim = 128, full split-half RoPE,
// optional QKV bias, no Q/K norm, no output gate) — Qwen2 / Qwen2.5 / Llama.
// ============================================================================

struct Std128PrepParams {
  uint rows;
  uint num_heads;
  uint num_kv_heads;
  uint qkv_stride;
  uint has_bias;
};

// Grid: threadgroups (num_heads + num_kv_heads, rows) of 32 threads.
kernel void attn_std128_rope_store(const device bfloat* qkv_buf [[buffer(0)]],
                                   const device bfloat* qkv_bias [[buffer(1)]],
                                   const device float* inv_freq [[buffer(2)]],
                                   const device uint* positions [[buffer(3)]],
                                   const device uint* kv_slots [[buffer(4)]],
                                   device bfloat* q_out [[buffer(5)]],
                                   device bfloat* k_cache [[buffer(6)]],
                                   device bfloat* v_cache [[buffer(7)]],
                                   constant Std128PrepParams& p [[buffer(8)]],
                                   uint2 grp [[threadgroup_position_in_grid]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint H128 = 128;
  const uint idx = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* row_ptr = qkv_buf + (size_t)row * p.qkv_stride;
  const float pos = float(positions[row]);
  const float a0 = pos * inv_freq[lane];
  const float a1 = pos * inv_freq[lane + 32];
  const float c0 = cos(a0), s0 = sin(a0);
  const float c1 = cos(a1), s1 = sin(a1);

  if (idx < p.num_heads) {
    const size_t q_off = (size_t)idx * H128;
    const device bfloat* src = row_ptr + q_off;
    float x0 = float(src[lane]);
    float x1 = float(src[lane + 32]);
    float x2 = float(src[lane + 64]);
    float x3 = float(src[lane + 96]);
    if (p.has_bias != 0) {
      const device bfloat* bs = qkv_bias + q_off;
      x0 += float(bs[lane]);
      x1 += float(bs[lane + 32]);
      x2 += float(bs[lane + 64]);
      x3 += float(bs[lane + 96]);
    }
    device bfloat* dst = q_out + (size_t)(row * p.num_heads + idx) * H128;
    dst[lane] = bfloat(x0 * c0 - x2 * s0);
    dst[lane + 64] = bfloat(x2 * c0 + x0 * s0);
    dst[lane + 32] = bfloat(x1 * c1 - x3 * s1);
    dst[lane + 96] = bfloat(x3 * c1 + x1 * s1);
    return;
  }

  const uint kh = idx - p.num_heads;
  const uint slot = kv_slots[row];
  const size_t k_off = (size_t)p.num_heads * H128 + (size_t)kh * H128;
  const size_t v_off = k_off + (size_t)p.num_kv_heads * H128;
  const device bfloat* k_src = row_ptr + k_off;
  const device bfloat* v_src = row_ptr + v_off;

  float k0 = float(k_src[lane]);
  float k1 = float(k_src[lane + 32]);
  float k2 = float(k_src[lane + 64]);
  float k3 = float(k_src[lane + 96]);
  float v0 = float(v_src[lane]);
  float v1 = float(v_src[lane + 32]);
  float v2 = float(v_src[lane + 64]);
  float v3 = float(v_src[lane + 96]);
  if (p.has_bias != 0) {
    const device bfloat* kb = qkv_bias + k_off;
    const device bfloat* vb = qkv_bias + v_off;
    k0 += float(kb[lane]);
    k1 += float(kb[lane + 32]);
    k2 += float(kb[lane + 64]);
    k3 += float(kb[lane + 96]);
    v0 += float(vb[lane]);
    v1 += float(vb[lane + 32]);
    v2 += float(vb[lane + 64]);
    v3 += float(vb[lane + 96]);
  }

  const size_t cache_base = (size_t)(slot * p.num_kv_heads + kh) * H128;
  device bfloat* k_dst = k_cache + cache_base;
  device bfloat* v_dst = v_cache + cache_base;
  k_dst[lane] = bfloat(k0 * c0 - k2 * s0);
  k_dst[lane + 64] = bfloat(k2 * c0 + k0 * s0);
  k_dst[lane + 32] = bfloat(k1 * c1 - k3 * s1);
  k_dst[lane + 96] = bfloat(k3 * c1 + k1 * s1);
  v_dst[lane] = bfloat(v0);
  v_dst[lane + 32] = bfloat(v1);
  v_dst[lane + 64] = bfloat(v2);
  v_dst[lane + 96] = bfloat(v3);
}

// Fused HD=128 Flash GQA + o_proj group sums (supports causal chain + bit-exact tree).
// Grid: threadgroups (num_heads, rows) of 128 threads (4 simdgroups).
kernel void attn_std128_flash(const device bfloat* q_in [[buffer(0)]],
                              const device bfloat* k_cache [[buffer(1)]],
                              const device bfloat* v_cache [[buffer(2)]],
                              const device uint* seq_lens [[buffer(3)]],
                              const device uint4* tree_masks [[buffer(4)]],
                              device bfloat* out [[buffer(5)]],
                              device float* sums [[buffer(6)]],
                              constant AttnFlashParams& p [[buffer(7)]],
                              uint2 grp [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint H128 = 128;
  const uint head = grp.x;
  const uint row = grp.y;
  if (head >= p.num_heads || row >= p.rows) return;

  const uint kv_head = head / (p.num_heads / p.num_kv_heads);
  const device bfloat* q_ptr = q_in + (size_t)(row * p.num_heads + head) * H128;

  const float q0 = float(q_ptr[lane]) * p.scale;
  const float q1 = float(q_ptr[32 + lane]) * p.scale;
  const float q2 = float(q_ptr[64 + lane]) * p.scale;
  const float q3 = float(q_ptr[96 + lane]) * p.scale;

  float m_val = -INFINITY;
  float l_val = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  const size_t kv_stride = (size_t)p.num_kv_heads * H128;
  const size_t kh_off = (size_t)kv_head * H128;
  const uint prefix_end = (p.is_tree != 0) ? p.base_pos : seq_lens[row];

  for (uint t = sg; t < prefix_end; t += 4) {
    const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
    const float dot_part = fma(q0, float(kp[lane]), fma(q1, float(kp[32 + lane]), fma(q2, float(kp[64 + lane]), q3 * float(kp[96 + lane]))));
    const float score = simd_sum(dot_part);
    const float new_m = max(m_val, score);
    const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
    const float prob = exp(score - new_m);
    l_val = fma(l_val, alpha, prob);
    m_val = new_m;

    const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
    a0 = fma(a0, alpha, prob * float(vp[lane]));
    a1 = fma(a1, alpha, prob * float(vp[32 + lane]));
    a2 = fma(a2, alpha, prob * float(vp[64 + lane]));
    a3 = fma(a3, alpha, prob * float(vp[96 + lane]));
  }

  if (p.is_tree != 0) {
    const uint4 mask = tree_masks[row];
    uint anc_idx = 0;
    for (uint r = 0; r <= row; r++) {
      if ((mask[r >> 5] & (1u << (r & 31))) == 0u) continue;
      if (((p.base_pos + anc_idx) & 3u) == sg) {
        const uint t = p.base_pos + r;
        const device bfloat* kp = k_cache + (size_t)t * kv_stride + kh_off;
        const float dot_part = fma(q0, float(kp[lane]), fma(q1, float(kp[32 + lane]), fma(q2, float(kp[64 + lane]), q3 * float(kp[96 + lane]))));
        const float score = simd_sum(dot_part);
        const float new_m = max(m_val, score);
        const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - new_m);
        const float prob = exp(score - new_m);
        l_val = fma(l_val, alpha, prob);
        m_val = new_m;

        const device bfloat* vp = v_cache + (size_t)t * kv_stride + kh_off;
        a0 = fma(a0, alpha, prob * float(vp[lane]));
        a1 = fma(a1, alpha, prob * float(vp[32 + lane]));
        a2 = fma(a2, alpha, prob * float(vp[64 + lane]));
        a3 = fma(a3, alpha, prob * float(vp[96 + lane]));
      }
      anc_idx++;
    }
  }

  threadgroup float tg_m[4];
  threadgroup float tg_l[4];
  threadgroup float tg_acc[4][H128];

  if (lane == 0) {
    tg_m[sg] = m_val;
    tg_l[sg] = l_val;
  }
  tg_acc[sg][lane] = a0;
  tg_acc[sg][32 + lane] = a1;
  tg_acc[sg][64 + lane] = a2;
  tg_acc[sg][96 + lane] = a3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  const float gm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
  float w[4];
  float gl = 0.0f;
  UNROLL for (uint s = 0; s < 4; s++) {
    w[s] = (tg_l[s] > 0.0f) ? exp(tg_m[s] - gm) : 0.0f;
    gl += tg_l[s] * w[s];
  }
  const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

  device bfloat* out_ptr = out + (size_t)(row * p.num_heads + head) * H128;
  UNROLL for (uint g = 0; g < 2; g++) {
    const uint i0 = g * 64 + lane;
    const uint i1 = g * 64 + 32 + lane;
    float o0 = 0.0f, o1 = 0.0f;
    UNROLL for (uint s = 0; s < 4; s++) {
      o0 = fma(tg_acc[s][i0], w[s], o0);
      o1 = fma(tg_acc[s][i1], w[s], o1);
    }
    const bfloat y0 = bfloat(o0 * inv_l);
    const bfloat y1 = bfloat(o1 * inv_l);
    out_ptr[i0] = y0;
    out_ptr[i1] = y1;
    const float gs = simd_sum(float(y0) + float(y1));
    if (lane == 0) {
      sums[(size_t)(head * 2 + g) * p.sums_stride + row] = gs;
    }
  }
}
