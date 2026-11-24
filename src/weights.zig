//! Parallel weight loader & qk tile packer for Qwen3.5 / Qwen3.8 MLX 4-bit checkpoints.
//!
//! Fuses shared-input projections vertically at load time into single qk matrices:
//!   - GDN layers:  in_proj_all = [in_proj_qkv ; in_proj_z ; in_proj_b ; in_proj_a]
//!   - Attn layers: qkv_proj    = [q_proj ; k_proj ; v_proj]
//!   - All layers:  gate_up     = [gate_proj ; up_proj]
const std = @import("std");
const mtl = @import("metal4.zig");
const st = @import("safetensors.zig");
const cfg_mod = @import("config.zig");

const ModelConfig = cfg_mod.ModelConfig;
const LayerKind = cfg_mod.LayerKind;
const Buffer = mtl.Buffer;
const BufferRef = mtl.BufferRef;

extern "c" fn dispatch_apply_f(
    iterations: usize,
    queue: ?*anyopaque,
    context: ?*anyopaque,
    work: *const fn (?*anyopaque, usize) callconv(.c) void,
) void;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;

pub const qk_nt: usize = 32;
pub const qk_gs: usize = 64;
pub const qk_codes_per_block: usize = qk_nt * qk_gs / 2; // 1024 bytes
pub const qk_block_bytes: usize = qk_codes_per_block + qk_nt * 4; // 1152 bytes (codes + interleaved scale/bias)

pub const QkMatrix = struct {
    ref: BufferRef,
    n: u32,
    k: u32,

    pub fn groups(self: QkMatrix) u32 {
        return self.k / @as(u32, qk_gs);
    }

    pub fn tiles(self: QkMatrix) u32 {
        return self.n / @as(u32, qk_nt);
    }

    pub fn byteSize(n: usize, k: usize) usize {
        std.debug.assert(n % qk_nt == 0);
        std.debug.assert(k % qk_gs == 0);
        const tiles_n = n / qk_nt;
        const g = k / qk_gs;
        // +1 block slack for lookahead scale/bias prefetch in qk.metal
        return (tiles_n * g + 1) * qk_block_bytes;
    }
};

pub const EmbedWeights = struct {
    codes: BufferRef, // U32 [vocab, hidden / 8]
    scales: BufferRef, // BF16 [vocab, hidden / 64]
    biases: BufferRef, // BF16 [vocab, hidden / 64]
};

pub const LinearLayerWeights = struct {
    in_proj_all: QkMatrix, // [qkv + z + b + a, hidden]
    out_proj: QkMatrix, // [hidden, gdnValDim]
    conv_weight: BufferRef, // BF16 [gdnQkvDim, 4]
    a_log: BufferRef, // F32 [gdn_v_heads]
    dt_bias: BufferRef, // BF16 [gdn_v_heads]
    gdn_norm: BufferRef, // BF16 [gdn_v_dim]
};

pub const FullLayerWeights = struct {
    qkv_proj: QkMatrix, // [q_out + k_out + v_out, hidden]
    o_proj: QkMatrix, // [hidden, num_heads * head_dim]
    q_norm: BufferRef, // BF16 [head_dim] (unused when !has_qk_norm)
    k_norm: BufferRef, // BF16 [head_dim] (unused when !has_qk_norm)
    qkv_bias: ?BufferRef = null, // Optional BF16 [q_out + k_out + v_out] (Qwen2/Qwen2.5)
};

pub const LayerWeights = struct {
    kind: LayerKind,
    input_norm: BufferRef, // BF16 [hidden]
    post_attn_norm: BufferRef, // BF16 [hidden]
    gate_up: QkMatrix, // [2 * intermediate, hidden]
    down_proj: QkMatrix, // [hidden, intermediate]
    linear: LinearLayerWeights = undefined,
    full: FullLayerWeights = undefined,
};

pub const ModelWeights = struct {
    a: std.mem.Allocator,
    buffers: std.ArrayList(Buffer) = .empty,
    embed: EmbedWeights,
    final_norm: BufferRef,
    lm_head: QkMatrix,
    layers: []LayerWeights,
    total_bytes: usize = 0,

    pub fn deinit(self: *ModelWeights) void {
        self.a.free(self.layers);
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.a);
    }

    pub fn load(a: std.mem.Allocator, ctx: *mtl.Context, c: ModelConfig, model_dir: []const u8) !ModelWeights {
        var ckpt = try st.Checkpoint.openDir(a, model_dir);
        defer ckpt.close();

        var w = ModelWeights{
            .a = a,
            .embed = undefined,
            .final_norm = undefined,
            .lm_head = undefined,
            .layers = try a.alloc(LayerWeights, c.num_layers),
        };
        errdefer w.deinit();

        // Detect tensor prefix ("language_model.model." vs "model.")
        const prefix: []const u8 = if (ckpt.getOptional("language_model.model.norm.weight") != null)
            "language_model.model."
        else
            "model.";
        const head_prefix: []const u8 = if (ckpt.getOptional("language_model.lm_head.weight") != null)
            "language_model."
        else
            "";

        // 1. Embeddings (either pre-quantized 4-bit MLX or raw BF16 quantized on-the-fly via GCD!)
        {
            const ew = try getPrefixed(&ckpt, prefix, "embed_tokens.weight");
            const es_opt = getPrefixedOpt(&ckpt, prefix, "embed_tokens.scales");
            const eb_opt = getPrefixedOpt(&ckpt, prefix, "embed_tokens.biases");

            if (es_opt != null and eb_opt != null) {
                const es = es_opt.?;
                const eb = eb_opt.?;
                const total = align64(ew.bytes.len) + align64(es.bytes.len) + align64(eb.bytes.len);
                const buf = try allocTracked(&w, ctx, total);
                var off: usize = 0;
                w.embed.codes = copyInto(buf, &off, ew.bytes);
                w.embed.scales = copyInto(buf, &off, es.bytes);
                w.embed.biases = copyInto(buf, &off, eb.bytes);

                const hw = if (c.tie_word_embeddings) ew else try getPrefixed(&ckpt, head_prefix, "lm_head.weight");
                const hs = if (c.tie_word_embeddings) es else try getPrefixed(&ckpt, head_prefix, "lm_head.scales");
                const hb = if (c.tie_word_embeddings) eb else try getPrefixed(&ckpt, head_prefix, "lm_head.biases");
                const head_bytes = QkMatrix.byteSize(c.vocab_size, c.hidden_size);
                const head_buf = try allocTracked(&w, ctx, head_bytes);
                const srcs = [1]RawQuant{.{ .w = hw, .s = hs, .b = hb, .n = c.vocab_size }};
                packQkConcat(head_buf.ptr, &srcs, c.vocab_size, c.hidden_size);
                w.lm_head = .{ .ref = head_buf.view(0), .n = @intCast(c.vocab_size), .k = @intCast(c.hidden_size) };
            } else {
                // On-the-fly parallel BF16 -> 4-bit affine group-64 quantization for embeddings + lm_head!
                const codes_len = c.vocab_size * (c.hidden_size / 2);
                const sb_len = c.vocab_size * (c.hidden_size / qk_gs) * 2;
                const total = align64(codes_len) + align64(sb_len) * 2;
                const buf = try allocTracked(&w, ctx, total);
                const codes_off: usize = 0;
                const scales_off: usize = align64(codes_len);
                const biases_off: usize = scales_off + align64(sb_len);
                quantizeEmbedBf16(
                    ew,
                    c.vocab_size,
                    c.hidden_size,
                    buf.ptr + codes_off,
                    @ptrCast(@alignCast(buf.ptr + scales_off)),
                    @ptrCast(@alignCast(buf.ptr + biases_off)),
                );
                w.embed.codes = buf.view(codes_off);
                w.embed.scales = buf.view(scales_off);
                w.embed.biases = buf.view(biases_off);

                const hw = if (c.tie_word_embeddings) ew else try getPrefixed(&ckpt, head_prefix, "lm_head.weight");
                const hs_opt = if (c.tie_word_embeddings) null else getPrefixedOpt(&ckpt, head_prefix, "lm_head.scales");
                const hb_opt = if (c.tie_word_embeddings) null else getPrefixedOpt(&ckpt, head_prefix, "lm_head.biases");
                const head_bytes = QkMatrix.byteSize(c.vocab_size, c.hidden_size);
                const head_buf = try allocTracked(&w, ctx, head_bytes);
                const srcs = [1]RawQuant{.{ .w = hw, .s = hs_opt, .b = hb_opt, .n = c.vocab_size }};
                packQkConcat(head_buf.ptr, &srcs, c.vocab_size, c.hidden_size);
                w.lm_head = .{ .ref = head_buf.view(0), .n = @intCast(c.vocab_size), .k = @intCast(c.hidden_size) };
            }
        }

        // 2. Small non-quantized weights (norms, conv1d, A_log, dt_bias, optional QKV bias) in one contiguous buffer
        {
            const fn_t = try getPrefixed(&ckpt, prefix, "norm.weight");
            var small_bytes: usize = align64(fn_t.bytes.len);
            const attn_qkv_dim = c.qProjOut() + 2 * c.kvProjOut();
            for (0..c.num_layers) |li| {
                small_bytes += align64(c.hidden_size * 2) * 2; // input_norm + post_attn_norm
                if (c.layer_kinds[li] == .linear) {
                    small_bytes += align64(c.gdnQkvDim() * c.conv_kernel * 2); // conv_weight bf16
                    small_bytes += align64(c.gdn_v_heads * 4); // a_log f32
                    small_bytes += align64(c.gdn_v_heads * 2); // dt_bias bf16
                    small_bytes += align64(c.gdn_v_dim * 2); // gdn_norm bf16
                } else {
                    if (c.has_qk_norm) small_bytes += align64(c.head_dim * 2) * 2;
                    if (c.has_qkv_bias) small_bytes += align64(attn_qkv_dim * 2);
                }
            }
            const small_buf = try allocTracked(&w, ctx, small_bytes);
            var off: usize = 0;
            w.final_norm = copyInto(small_buf, &off, fn_t.bytes);

            var name_buf: [160]u8 = undefined;
            for (0..c.num_layers) |li| {
                w.layers[li].kind = c.layer_kinds[li];
                const in_n = try getLayer(&ckpt, &name_buf, prefix, li, "input_layernorm.weight");
                const pa_n = try getLayer(&ckpt, &name_buf, prefix, li, "post_attention_layernorm.weight");
                w.layers[li].input_norm = copyInto(small_buf, &off, in_n.bytes);
                w.layers[li].post_attn_norm = copyInto(small_buf, &off, pa_n.bytes);

                if (c.layer_kinds[li] == .linear) {
                    const cw = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.conv1d.weight");
                    const al = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.A_log");
                    const dtb = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.dt_bias");
                    const gn = try getLayer(&ckpt, &name_buf, prefix, li, "linear_attn.norm.weight");
                    w.layers[li].linear.conv_weight = copyInto(small_buf, &off, cw.bytes);
                    w.layers[li].linear.a_log = copyAsF32(small_buf, &off, al);
                    w.layers[li].linear.dt_bias = copyInto(small_buf, &off, dtb.bytes);
                    w.layers[li].linear.gdn_norm = copyInto(small_buf, &off, gn.bytes);
                } else {
                    if (c.has_qk_norm) {
                        const qn = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.q_norm.weight");
                        const kn = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.k_norm.weight");
                        w.layers[li].full.q_norm = copyInto(small_buf, &off, qn.bytes);
                        w.layers[li].full.k_norm = copyInto(small_buf, &off, kn.bytes);
                    } else {
                        w.layers[li].full.q_norm = small_buf.view(0);
                        w.layers[li].full.k_norm = small_buf.view(0);
                    }
                    if (c.has_qkv_bias) {
                        const qb = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.q_proj.bias");
                        const kb = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.k_proj.bias");
                        const vb = try getLayer(&ckpt, &name_buf, prefix, li, "self_attn.v_proj.bias");
                        const bstart = off;
                        @memcpy(small_buf.ptr[off..][0..qb.bytes.len], qb.bytes);
                        off += qb.bytes.len;
                        @memcpy(small_buf.ptr[off..][0..kb.bytes.len], kb.bytes);
                        off += kb.bytes.len;
                        @memcpy(small_buf.ptr[off..][0..vb.bytes.len], vb.bytes);
                        off = bstart + align64(qb.bytes.len + kb.bytes.len + vb.bytes.len);
                        w.layers[li].full.qkv_bias = small_buf.view(bstart);
                    } else {
                        w.layers[li].full.qkv_bias = null;
                    }
                }
            }
        }

        // 3. Per-layer packed qk matrices (one Metal buffer per layer to keep allocation count tiny)
        const gdn_all_n = c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads;
        const attn_qkv_n = c.qProjOut() + 2 * c.kvProjOut();
        const attn_o_k = c.num_heads * c.head_dim;

        var name_buf: [160]u8 = undefined;
        for (0..c.num_layers) |li| {
            const gu_bytes = QkMatrix.byteSize(2 * c.intermediate_size, c.hidden_size);
            const down_bytes = QkMatrix.byteSize(c.hidden_size, c.intermediate_size);

            if (c.layer_kinds[li] == .linear) {
                const in_bytes = QkMatrix.byteSize(gdn_all_n, c.hidden_size);
                const out_bytes = QkMatrix.byteSize(c.hidden_size, c.gdnValDim());
                const total = align64(in_bytes) + align64(out_bytes) + align64(gu_bytes) + align64(down_bytes);
                const lbuf = try allocTracked(&w, ctx, total);
                var off: usize = 0;

                const qkv_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "linear_attn.in_proj_qkv", c.gdnQkvDim());
                const z_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "linear_attn.in_proj_z", c.gdnValDim());
                const b_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "linear_attn.in_proj_b", c.gdn_v_heads);
                const a_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "linear_attn.in_proj_a", c.gdn_v_heads);
                const in_srcs = [4]RawQuant{ qkv_q, z_q, b_q, a_q };
                w.layers[li].linear.in_proj_all = packInto(lbuf, &off, &in_srcs, gdn_all_n, c.hidden_size);

                const out_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "linear_attn.out_proj", c.hidden_size);
                w.layers[li].linear.out_proj = packInto(lbuf, &off, &.{out_q}, c.hidden_size, c.gdnValDim());

                const g_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "mlp.gate_proj", c.intermediate_size);
                const u_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "mlp.up_proj", c.intermediate_size);
                w.layers[li].gate_up = packInto(lbuf, &off, &.{ g_q, u_q }, 2 * c.intermediate_size, c.hidden_size);

                const d_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "mlp.down_proj", c.hidden_size);
                w.layers[li].down_proj = packInto(lbuf, &off, &.{d_q}, c.hidden_size, c.intermediate_size);
            } else {
                const qkv_bytes = QkMatrix.byteSize(attn_qkv_n, c.hidden_size);
                const o_bytes = QkMatrix.byteSize(c.hidden_size, attn_o_k);
                const total = align64(qkv_bytes) + align64(o_bytes) + align64(gu_bytes) + align64(down_bytes);
                const lbuf = try allocTracked(&w, ctx, total);
                var off: usize = 0;

                const q_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "self_attn.q_proj", c.qProjOut());
                const k_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "self_attn.k_proj", c.kvProjOut());
                const v_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "self_attn.v_proj", c.kvProjOut());
                w.layers[li].full.qkv_proj = packInto(lbuf, &off, &.{ q_q, k_q, v_q }, attn_qkv_n, c.hidden_size);

                const o_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "self_attn.o_proj", c.hidden_size);
                w.layers[li].full.o_proj = packInto(lbuf, &off, &.{o_q}, c.hidden_size, attn_o_k);

                const g_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "mlp.gate_proj", c.intermediate_size);
                const u_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "mlp.up_proj", c.intermediate_size);
                w.layers[li].gate_up = packInto(lbuf, &off, &.{ g_q, u_q }, 2 * c.intermediate_size, c.hidden_size);

                const d_q = try getQuantLayer(&ckpt, &name_buf, prefix, li, "mlp.down_proj", c.hidden_size);
                w.layers[li].down_proj = packInto(lbuf, &off, &.{d_q}, c.hidden_size, c.intermediate_size);
            }
        }

        return w;
    }
};

const RawQuant = struct {
    w: st.Tensor,
    s: ?st.Tensor = null,
    b: ?st.Tensor = null,
    n: usize,
};

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7FFF +% lsb) >> 16);
}

inline fn quantGroup64Bf16(src: [*]align(1) const u16, dst_codes: [*]u8) struct { s: u16, b: u16 } {
    var vmin: f32 = std.math.inf(f32);
    var vmax: f32 = -std.math.inf(f32);
    for (0..qk_gs) |i| {
        const v = bf16ToF32(src[i]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    const span = vmax - vmin;
    const scale: f32 = if (span > 1e-12) span / 15.0 else 0.0;
    const s_bf = f32ToBf16(scale);
    const b_bf = f32ToBf16(vmin);
    const s_f = bf16ToF32(s_bf);
    const b_f = bf16ToF32(b_bf);
    const inv_s: f32 = if (s_f > 0.0) 1.0 / s_f else 0.0;

    for (0..32) |bi| {
        const v0 = bf16ToF32(src[bi * 2]);
        const v1 = bf16ToF32(src[bi * 2 + 1]);
        const q0: u8 = @intFromFloat(@round(std.math.clamp((v0 - b_f) * inv_s, 0.0, 15.0)));
        const q1: u8 = @intFromFloat(@round(std.math.clamp((v1 - b_f) * inv_s, 0.0, 15.0)));
        dst_codes[bi] = q0 | (q1 << 4);
    }
    return .{ .s = s_bf, .b = b_bf };
}

const EmbedQuantJob = struct {
    src: [*]align(1) const u16,
    dst_codes: [*]u8,
    dst_scales: [*]u16,
    dst_biases: [*]u16,
    hidden: usize,
    groups: usize,
};

fn embedQuantWorker(raw_ctx: ?*anyopaque, row_idx: usize) callconv(.c) void {
    const job: *const EmbedQuantJob = @ptrCast(@alignCast(raw_ctx.?));
    const src_row = job.src + row_idx * job.hidden;
    const dst_c = job.dst_codes + row_idx * (job.hidden / 2);
    const dst_s = job.dst_scales + row_idx * job.groups;
    const dst_b = job.dst_biases + row_idx * job.groups;
    for (0..job.groups) |g| {
        const sb = quantGroup64Bf16(src_row + g * qk_gs, dst_c + g * 32);
        dst_s[g] = sb.s;
        dst_b[g] = sb.b;
    }
}

fn quantizeEmbedBf16(ew: st.Tensor, vocab: usize, hidden: usize, dst_codes: [*]u8, dst_scales: [*]u16, dst_biases: [*]u16) void {
    var job = EmbedQuantJob{
        .src = ew.asSlice(u16).ptr,
        .dst_codes = dst_codes,
        .dst_scales = dst_scales,
        .dst_biases = dst_biases,
        .hidden = hidden,
        .groups = hidden / qk_gs,
    };
    const q = dispatch_get_global_queue(0, 0);
    dispatch_apply_f(vocab, q, &job, embedQuantWorker);
}

fn packInto(buf: Buffer, off: *usize, srcs: []const RawQuant, n: usize, k: usize) QkMatrix {
    const start = off.*;
    const size = QkMatrix.byteSize(n, k);
    packQkConcat(buf.ptr + start, srcs, n, k);
    off.* += align64(size);
    return .{ .ref = buf.view(start), .n = @intCast(n), .k = @intCast(k) };
}

const PackJob = struct {
    dst: [*]u8,
    srcs: []const RawQuant,
    groups: usize,
    row_code_bytes: usize,
};

fn packTileWorker(raw_ctx: ?*anyopaque, tile_idx: usize) callconv(.c) void {
    const job: *const PackJob = @ptrCast(@alignCast(raw_ctx.?));
    const g_count = job.groups;
    const k_cols = g_count * qk_gs;
    const tile_dst = job.dst + tile_idx * g_count * qk_block_bytes;
    const base_row = tile_idx * qk_nt;

    for (0..qk_nt) |c| {
        const global_row = base_row + c;
        // Locate which source matrix owns global_row
        var local_row = global_row;
        var src_idx: usize = 0;
        while (src_idx < job.srcs.len and local_row >= job.srcs[src_idx].n) : (src_idx += 1) {
            local_row -= job.srcs[src_idx].n;
        }
        const src = job.srcs[src_idx];
        if (src.s != null and src.b != null) {
            const w_row = src.w.bytes.ptr + local_row * job.row_code_bytes;
            const s_row = src.s.?.asSlice(u16).ptr + local_row * g_count;
            const b_row = src.b.?.asSlice(u16).ptr + local_row * g_count;

            for (0..g_count) |g| {
                const blk = tile_dst + g * qk_block_bytes;
                @memcpy(blk[c * 32 ..][0..32], w_row[g * 32 ..][0..32]);
                const sb: u32 = @as(u32, s_row[g]) | (@as(u32, b_row[g]) << 16);
                std.mem.writeInt(u32, blk[qk_codes_per_block + c * 4 ..][0..4], sb, .little);
            }
        } else {
            // Raw BF16 weight -> on-the-fly 4-bit affine group-64 quantization directly into qk tile!
            const bf_row = src.w.asSlice(u16).ptr + local_row * k_cols;
            for (0..g_count) |g| {
                const blk = tile_dst + g * qk_block_bytes;
                const sb_pair = quantGroup64Bf16(bf_row + g * qk_gs, blk + c * 32);
                const sb: u32 = @as(u32, sb_pair.s) | (@as(u32, sb_pair.b) << 16);
                std.mem.writeInt(u32, blk[qk_codes_per_block + c * 4 ..][0..4], sb, .little);
            }
        }
    }
}

/// Pack vertically concatenated MLX 4-bit matrices into qk tile layout in parallel via GCD.
fn packQkConcat(dst: [*]u8, srcs: []const RawQuant, total_n: usize, k: usize) void {
    const tiles = total_n / qk_nt;
    const groups = k / qk_gs;
    var job = PackJob{
        .dst = dst,
        .srcs = srcs,
        .groups = groups,
        .row_code_bytes = k / 2,
    };
    const q = dispatch_get_global_queue(0, 0);
    dispatch_apply_f(tiles, q, &job, packTileWorker);
    // Zero trailing slack block
    @memset(dst[tiles * groups * qk_block_bytes ..][0..qk_block_bytes], 0);
}

fn allocTracked(w: *ModelWeights, ctx: *mtl.Context, size: usize) !Buffer {
    const b = try ctx.allocBuffer(size);
    try w.buffers.append(w.a, b);
    w.total_bytes += b.len;
    return b;
}

fn align64(n: usize) usize {
    return (n + 63) & ~@as(usize, 63);
}

fn copyInto(buf: Buffer, off: *usize, src: []const u8) BufferRef {
    const start = off.*;
    @memcpy(buf.ptr[start..][0..src.len], src);
    off.* += align64(src.len);
    return buf.view(start);
}

fn copyAsF32(buf: Buffer, off: *usize, t: st.Tensor) BufferRef {
    const start = off.*;
    const n = t.elemCount();
    const dst = @as([*]f32, @ptrCast(@alignCast(buf.ptr + start)))[0..n];
    if (t.dtype == .f32) {
        const src = t.asSlice(f32);
        for (dst, src) |*d, s| d.* = s;
    } else if (t.dtype == .bf16) {
        const src = t.asSlice(u16);
        for (dst, src) |*d, s| {
            const bits: u32 = @as(u32, s) << 16;
            d.* = @bitCast(bits);
        }
    } else {
        @panic("Unsupported A_log dtype");
    }
    off.* += align64(n * 4);
    return buf.view(start);
}

fn getPrefixed(ckpt: *const st.Checkpoint, prefix: []const u8, suffix: []const u8) !st.Tensor {
    var buf: [160]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }) catch return error.NameTooLong;
    return ckpt.get(key);
}

fn getPrefixedOpt(ckpt: *const st.Checkpoint, prefix: []const u8, suffix: []const u8) ?st.Tensor {
    var buf: [160]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }) catch return null;
    return ckpt.getOptional(key);
}

fn getLayer(ckpt: *const st.Checkpoint, buf: *[160]u8, prefix: []const u8, layer: usize, suffix: []const u8) !st.Tensor {
    const key = std.fmt.bufPrint(buf, "{s}layers.{d}.{s}", .{ prefix, layer, suffix }) catch return error.NameTooLong;
    return ckpt.get(key);
}

fn getQuantLayer(ckpt: *const st.Checkpoint, buf: *[160]u8, prefix: []const u8, layer: usize, mod: []const u8, expected_n: usize) !RawQuant {
    var b1: [160]u8 = undefined;
    var b2: [160]u8 = undefined;
    const w = try ckpt.get(std.fmt.bufPrint(buf, "{s}layers.{d}.{s}.weight", .{ prefix, layer, mod }) catch return error.NameTooLong);
    const s = ckpt.getOptional(std.fmt.bufPrint(&b1, "{s}layers.{d}.{s}.scales", .{ prefix, layer, mod }) catch return error.NameTooLong);
    const b = ckpt.getOptional(std.fmt.bufPrint(&b2, "{s}layers.{d}.{s}.biases", .{ prefix, layer, mod }) catch return error.NameTooLong);
    std.debug.assert(w.shape[0] == expected_n);
    return .{ .w = w, .s = s, .b = b, .n = expected_n };
}
