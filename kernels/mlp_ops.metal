// Clean-room fused SwiGLU activation + down_proj per-64-group x-sums.
//
// Input: gate_up[rows, 2 * inter] where each row holds [gate(0..inter), up(0..inter)].
// Output: act[rows, inter] = silu(gate) * up, plus sums[inter/64, sums_stride].
// Grid: threadgroups (inter / 64, rows) of 32 threads (1 simdgroup per group/row, zero TG barriers).
#include <metal_stdlib>
using namespace metal;

struct MlpActParams {
  uint rows;
  uint inter;
  uint sums_stride;
};

kernel void gu_activate_sums(const device bfloat* gate_up [[buffer(0)]],
                             device bfloat* act [[buffer(1)]],
                             device float* sums [[buffer(2)]],
                             constant MlpActParams& p [[buffer(3)]],
                             uint2 grp [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const uint g = grp.x;
  const uint row = grp.y;
  if (row >= p.rows) return;

  const device bfloat* g_ptr = gate_up + (size_t)row * (2 * p.inter);
  const device bfloat* u_ptr = g_ptr + p.inter;
  const uint i0 = g * 64 + lane;
  const uint i1 = g * 64 + 32 + lane;

  const float gv0 = float(g_ptr[i0]);
  const float gv1 = float(g_ptr[i1]);
  const float uv0 = float(u_ptr[i0]);
  const float uv1 = float(u_ptr[i1]);

  const bfloat y0 = bfloat((gv0 / (1.0f + exp(-gv0))) * uv0);
  const bfloat y1 = bfloat((gv1 / (1.0f + exp(-gv1))) * uv1);

  device bfloat* out = act + (size_t)row * p.inter;
  out[i0] = y0;
  out[i1] = y1;

  const float gs = simd_sum(float(y0) + float(y1));
  if (lane == 0) {
    sums[(size_t)g * p.sums_stride + row] = gs;
  }
}
