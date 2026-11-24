//! Clean-room Qwen3.5 / Qwen3.8 Hybrid GDN + GQA Metal forward engine.
//!
//! Zero Apple MLX. Zero TensorFold code. Supports:
//!   - Fast chunked prefill (1..128 rows)
//!   - Dedicated coalesced 32-col 128-bit qmv_1 for single-token decode (M == 1)
//!   - M5 hardware tensor-unit qk_{16,32,48} for multi-row speculative chain & tree verification (M = 2..128)
//!   - Fast speculative state commit (commitKeep) for both GDN and KV cache
//!   - Optional 5-layer hidden tap capture for DFlash2 drafting
const std = @import("std");
const shaders = @import("shaders");
const mtl = @import("metal4.zig");
const cfg_mod = @import("config.zig");
const w_mod = @import("weights.zig");
const samp_mod = @import("sampler.zig");

const ModelConfig = cfg_mod.ModelConfig;
const ModelWeights = w_mod.ModelWeights;
const QkMatrix = w_mod.QkMatrix;
const Buffer = mtl.Buffer;
const BufferRef = mtl.BufferRef;
const Pipeline = mtl.Pipeline;
const Size = mtl.Size;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
pub fn nowMs() f64 {
    return @as(f64, @floatFromInt(clock_gettime_nsec_np(8))) / 1e6;
}

pub const tap_layers = [5]usize{ 5, 19, 33, 47, 61 };

const QkParams = extern struct {
    M: i32,
    K: i32,
    N: i32,
    G: i32,
    first_row: i32,
    sums_stride: i32,
    tile_first: u32,
};

const EmbedParams = extern struct {
    rows: u32,
    hidden: u32,
};

const NormParams = extern struct {
    rows: u32,
    hidden: u32,
    sums_stride: u32,
    eps: f32,
};

const GdnNormParams = extern struct {
    rows: u32,
    nv: u32,
    dv: u32,
    in_proj_stride: u32,
    z_offset: u32,
    sums_stride: u32,
    eps: f32,
};

const GdnConvParams = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    qkv_dim: u32,
    in_proj_stride: u32,
    b_offset: u32,
    a_offset: u32,
    commit_single: u32,
    save_qkv: u32,
};

const GdnFusedStepParams = extern struct {
    nk: u32,
    nv: u32,
    qkv_dim: u32,
    in_proj_stride: u32,
    z_offset: u32,
    b_offset: u32,
    a_offset: u32,
    sums_stride: u32,
    write_sums: u32,
    eps: f32,
};

const GdnHistCommitParams = extern struct {
    qkv_dim: u32,
    in_proj_stride: u32,
    src0: u32,
    src1: u32,
    src2: u32,
};

const GdnScanParams = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    commit_state: u32,
};

const AttnPrepParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    eps: f32,
};

const Std128PrepParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    has_bias: u32,
};

const AttnFlashParams = extern struct {
    rows: u32,
    num_heads: u32,
    num_kv_heads: u32,
    qkv_stride: u32,
    sums_stride: u32,
    scale: f32,
    splits: u32,
    base_pos: u32,
    is_tree: u32,
};

const KvCompactParams = extern struct {
    base_pos: u32,
    keep_len: u32,
    row_u4: u32,
};

const MlpActParams = extern struct {
    rows: u32,
    inter: u32,
    sums_stride: u32,
};

const ArgmaxParams = extern struct {
    vocab: u32,
    chunks: u32,
    rows: u32,
};

const CopyParams = extern struct {
    count: u32,
};

pub const Pipelines = struct {
    libs: [6]mtl.Library,
    qk_16: Pipeline,
    qk_32: Pipeline,
    qk_48: Pipeline,
    qmv_1: Pipeline,
    qmv_1_silu_mul: Pipeline,
    qmv_1_argmax: Pipeline,
    embed_lookup: Pipeline,
    rms_norm_sums: Pipeline,
    add_rms_norm_sums: Pipeline,
    gdn_norm_gate_sums: Pipeline,
    gdn_conv: Pipeline,
    gdn_commit_history: Pipeline,
    gdn_scan_1: Pipeline,
    gdn_scan_2: Pipeline,
    gdn_scan_4: Pipeline,
    gdn_scan_8: Pipeline,
    gdn_conv_batched: Pipeline,
    gdn_scan_batched: Pipeline,
    gdn_fused_step_1: Pipeline,
    gdn_scan_record_corr: Pipeline,
    gdn_rewind_dual: Pipeline,
    attn_qk_rope_store: Pipeline,
    attn_qk_rope_store_forked: Pipeline,
    attn_flash_fused: Pipeline,
    attn_flash_forked: Pipeline,
    attn_flash_split: Pipeline,
    attn_merge_gate_sums: Pipeline,
    attn_std128_rope_store: Pipeline,
    attn_std128_flash: Pipeline,
    kv_compact_gather: Pipeline,
    kv_compact_scatter: Pipeline,
    gu_activate_sums: Pipeline,
    argmax_stage1: Pipeline,
    argmax_stage2: Pipeline,
    copy_u4: Pipeline,

    pub fn compile(ctx: *const mtl.Context) !Pipelines {
        const l_qk = try ctx.compileSource(shaders.qk, .fast);
        const l_norm = try ctx.compileSource(shaders.norm_ops, .fast);
        const l_gdn = try ctx.compileSource(shaders.gdn_ops, .fast);
        const l_attn = try ctx.compileSource(shaders.attention, .fast);
        const l_mlp = try ctx.compileSource(shaders.mlp_ops, .fast);
        const l_sample = try ctx.compileSource(shaders.sample, .fast);

        return .{
            .libs = .{ l_qk, l_norm, l_gdn, l_attn, l_mlp, l_sample },
            .qk_16 = try ctx.makePipeline(l_qk, "qk_16"),
            .qk_32 = try ctx.makePipeline(l_qk, "qk_32"),
            .qk_48 = try ctx.makePipeline(l_qk, "qk_48"),
            .qmv_1 = try ctx.makePipeline(l_qk, "qmv_1"),
            .qmv_1_silu_mul = try ctx.makePipeline(l_qk, "qmv_1_silu_mul"),
            .qmv_1_argmax = try ctx.makePipeline(l_qk, "qmv_1_argmax"),
            .embed_lookup = try ctx.makePipeline(l_norm, "embed_lookup"),
            .rms_norm_sums = try ctx.makePipeline(l_norm, "rms_norm_sums"),
            .add_rms_norm_sums = try ctx.makePipeline(l_norm, "add_rms_norm_sums"),
            .gdn_norm_gate_sums = try ctx.makePipeline(l_norm, "gdn_norm_gate_sums"),
            .gdn_conv = try ctx.makePipeline(l_gdn, "gdn_conv"),
            .gdn_commit_history = try ctx.makePipeline(l_gdn, "gdn_commit_history"),
            .gdn_scan_1 = try ctx.makePipeline(l_gdn, "gdn_scan_1"),
            .gdn_scan_2 = try ctx.makePipeline(l_gdn, "gdn_scan_2"),
            .gdn_scan_4 = try ctx.makePipeline(l_gdn, "gdn_scan_4"),
            .gdn_scan_8 = try ctx.makePipeline(l_gdn, "gdn_scan_8"),
            .gdn_conv_batched = try ctx.makePipeline(l_gdn, "gdn_conv_batched"),
            .gdn_scan_batched = try ctx.makePipeline(l_gdn, "gdn_scan_batched"),
            .gdn_fused_step_1 = try ctx.makePipeline(l_gdn, "gdn_fused_step_1"),
            .gdn_scan_record_corr = try ctx.makePipeline(l_gdn, "gdn_scan_record_corr"),
            .gdn_rewind_dual = try ctx.makePipeline(l_gdn, "gdn_rewind_dual"),
            .attn_qk_rope_store = try ctx.makePipeline(l_attn, "attn_qk_rope_store"),
            .attn_qk_rope_store_forked = try ctx.makePipeline(l_attn, "attn_qk_rope_store_forked"),
            .attn_flash_fused = try ctx.makePipeline(l_attn, "attn_flash_fused"),
            .attn_flash_forked = try ctx.makePipeline(l_attn, "attn_flash_forked"),
            .attn_flash_split = try ctx.makePipeline(l_attn, "attn_flash_split"),
            .attn_merge_gate_sums = try ctx.makePipeline(l_attn, "attn_merge_gate_sums"),
            .attn_std128_rope_store = try ctx.makePipeline(l_attn, "attn_std128_rope_store"),
            .attn_std128_flash = try ctx.makePipeline(l_attn, "attn_std128_flash"),
            .kv_compact_gather = try ctx.makePipeline(l_attn, "kv_compact_gather"),
            .kv_compact_scatter = try ctx.makePipeline(l_attn, "kv_compact_scatter"),
            .gu_activate_sums = try ctx.makePipeline(l_mlp, "gu_activate_sums"),
            .argmax_stage1 = try ctx.makePipeline(l_sample, "argmax_stage1"),
            .argmax_stage2 = try ctx.makePipeline(l_sample, "argmax_stage2"),
            .copy_u4 = try ctx.makePipeline(l_sample, "copy_u4"),
        };
    }

    pub fn deinit(self: *Pipelines) void {
        self.qk_16.deinit();
        self.qk_32.deinit();
        self.qk_48.deinit();
        self.qmv_1.deinit();
        self.qmv_1_silu_mul.deinit();
        self.qmv_1_argmax.deinit();
        self.embed_lookup.deinit();
        self.rms_norm_sums.deinit();
        self.add_rms_norm_sums.deinit();
        self.gdn_norm_gate_sums.deinit();
        self.gdn_conv.deinit();
        self.gdn_commit_history.deinit();
        self.gdn_scan_1.deinit();
        self.gdn_scan_2.deinit();
        self.gdn_scan_4.deinit();
        self.gdn_scan_8.deinit();
        self.gdn_conv_batched.deinit();
        self.gdn_scan_batched.deinit();
        self.gdn_fused_step_1.deinit();
        self.gdn_scan_record_corr.deinit();
        self.gdn_rewind_dual.deinit();
        self.attn_qk_rope_store.deinit();
        self.attn_qk_rope_store_forked.deinit();
        self.attn_flash_fused.deinit();
        self.attn_flash_forked.deinit();
        self.attn_flash_split.deinit();
        self.attn_merge_gate_sums.deinit();
        self.attn_std128_rope_store.deinit();
        self.attn_std128_flash.deinit();
        self.kv_compact_gather.deinit();
        self.kv_compact_scatter.deinit();
        self.gu_activate_sums.deinit();
        self.argmax_stage1.deinit();
        self.argmax_stage2.deinit();
        self.copy_u4.deinit();
        for (self.libs) |l| l.deinit();
    }
};

pub const Options = struct {
    max_rows: u32 = 128,
    max_context: u32 = 8192,
    gu_parts: u32 = 0, // 0 = auto (4 on >=27B, 1 on smaller models)
    capture_taps: bool = false,
};

pub const PassTiming = struct {
    wall_ms: f64 = 0,
    encode_ms: f64 = 0,
};

const GdnVerifyLayout = struct {
    q_off: usize,
    k_off: usize,
    v_off: usize,
    decay_off: usize,
    mix_off: usize,
    qkv_off: usize,
    corr_off: usize,
    layer_stride: usize,
};

pub const Model = struct {
    a: std.mem.Allocator,
    ctx: *mtl.Context,
    c: ModelConfig,
    w: ModelWeights,
    p: Pipelines,
    fence: mtl.Fence,
    max_rows: u32,
    max_context: u32,
    sums_stride: u32,
    gu_parts: u32,
    commit_every: usize,
    capture_taps: bool,

    // Scratch buffers
    scratch_bufs: std.ArrayList(Buffer) = .empty,
    tokens_buf: Buffer,
    h_buf: Buffer,
    normed_buf: Buffer,
    sums_buf: Buffer,
    proj_big_buf: Buffer,
    delta_buf: Buffer,
    act_buf: Buffer,
    gdn_scan_out_buf: Buffer,
    conv_windows_buf: Buffer,
    scan_steps_buf: Buffer,
    keep_rows_buf: Buffer,
    attn_q_buf: Buffer,
    inv_freq_buf: Buffer,
    positions_buf: Buffer,
    kv_slots_buf: Buffer,
    seq_lens_buf: Buffer,
    tree_masks_buf: Buffer,
    partial_acc_buf: Buffer,
    partial_ml_buf: Buffer,
    kv_tmp_k_buf: Buffer,
    kv_tmp_v_buf: Buffer,
    logits_buf: Buffer,
    argmax_val_buf: Buffer,
    argmax_idx_buf: Buffer,
    out_tok_buf: Buffer,
    out_val_buf: Buffer,

    // Per-linear-layer verify cache (holds q, k, v, decay, mix, raw qkv for fast commitKeep)
    gdn_verify_buf: Buffer,
    gdn_vl: GdnVerifyLayout,

    // Optional 5-layer hidden taps for DFlash2 drafter ([5][max_rows][hidden] bf16)
    taps_buf: ?Buffer = null,
    taps_layer_stride: usize = 0,

    // Persistent stream state
    gdn_state_buf: Buffer,
    gdn_hist_buf: Buffer,
    gdn_hist_alt_buf: Buffer,
    k_cache_buf: Buffer,
    v_cache_buf: Buffer,
    gdn_state_stride: usize,
    gdn_hist_stride: usize,
    kv_layer_stride: usize,
    pos: u32 = 0,
    tape_capture: bool = false,
    last_timing: PassTiming = .{},

    pub const max_splits: u32 = 32;
    pub var use_qmv_1: bool = true;
    pub var use_fused_1: bool = true;
    pub var use_split_k: bool = true;

    pub fn init(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, opts: Options) !Model {
        const c = try ModelConfig.load(a, model_dir);
        var w = try ModelWeights.load(a, ctx, c, model_dir);
        errdefer w.deinit();
        var p = try Pipelines.compile(ctx);
        errdefer p.deinit();
        const fence = try ctx.createFence();
        errdefer fence.deinit();

        const max_rows = opts.max_rows;
        const sums_stride = ((max_rows + 15) / 16) * 16;
        const gdn_all_n = c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads;
        const attn_qkv_n = c.qProjOut() + 2 * c.kvProjOut();
        const max_proj_out = @max(2 * c.intermediate_size, @max(gdn_all_n, attn_qkv_n));
        const max_act_dim = @max(c.intermediate_size, @max(c.gdnValDim(), c.num_heads * c.head_dim));
        const max_groups = @max(c.hidden_size, max_act_dim) / 64;
        const chunks: usize = (c.vocab_size + 1023) / 1024;

        // Per-linear-layer verify cache layout
        var vl_off: usize = 0;
        const q_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnKeyDim() * 2);
        const k_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnKeyDim() * 2);
        const v_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnValDim() * 2);
        const decay_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdn_v_heads * 4);
        const mix_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdn_v_heads * 2);
        const qkv_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnQkvDim() * 2);
        const corr_off = vl_off;
        vl_off += align256(@as(usize, max_rows) * c.gdnValDim() * 4);
        const gdn_vl = GdnVerifyLayout{
            .q_off = q_off,
            .k_off = k_off,
            .v_off = v_off,
            .decay_off = decay_off,
            .mix_off = mix_off,
            .qkv_off = qkv_off,
            .corr_off = corr_off,
            .layer_stride = vl_off,
        };

        var m = Model{
            .a = a,
            .ctx = ctx,
            .c = c,
            .w = w,
            .p = p,
            .fence = fence,
            .max_rows = max_rows,
            .max_context = opts.max_context,
            .sums_stride = sums_stride,
            .gu_parts = if (opts.gu_parts != 0) opts.gu_parts else if (c.hidden_size >= 4096) 4 else 1,
            .commit_every = if (c.num_layers >= 48) 8 else c.num_layers,
            .capture_taps = opts.capture_taps,
            .tokens_buf = undefined,
            .h_buf = undefined,
            .normed_buf = undefined,
            .sums_buf = undefined,
            .proj_big_buf = undefined,
            .delta_buf = undefined,
            .act_buf = undefined,
            .gdn_scan_out_buf = undefined,
            .conv_windows_buf = undefined,
            .scan_steps_buf = undefined,
            .keep_rows_buf = undefined,
            .attn_q_buf = undefined,
            .inv_freq_buf = undefined,
            .positions_buf = undefined,
            .kv_slots_buf = undefined,
            .seq_lens_buf = undefined,
            .tree_masks_buf = undefined,
            .partial_acc_buf = undefined,
            .partial_ml_buf = undefined,
            .kv_tmp_k_buf = undefined,
            .kv_tmp_v_buf = undefined,
            .logits_buf = undefined,
            .argmax_val_buf = undefined,
            .argmax_idx_buf = undefined,
            .out_tok_buf = undefined,
            .out_val_buf = undefined,
            .gdn_verify_buf = undefined,
            .gdn_vl = gdn_vl,
            .gdn_state_buf = undefined,
            .gdn_hist_buf = undefined,
            .gdn_hist_alt_buf = undefined,
            .k_cache_buf = undefined,
            .v_cache_buf = undefined,
            .gdn_state_stride = align256(c.gdn_v_heads * c.gdn_v_dim * c.gdn_k_dim * 4),
            .gdn_hist_stride = align256(3 * c.gdnQkvDim() * 2),
            .kv_layer_stride = align256(@as(usize, opts.max_context) * c.num_kv_heads * c.head_dim * 2),
        };
        errdefer m.freeScratch();

        m.tokens_buf = try m.allocScratch(max_rows * 4);
        m.h_buf = try m.allocScratch(@as(usize, max_rows) * c.hidden_size * 2);
        m.normed_buf = try m.allocScratch(@as(usize, max_rows) * c.hidden_size * 2);
        m.sums_buf = try m.allocScratch(max_groups * sums_stride * 4);
        m.proj_big_buf = try m.allocScratch(@as(usize, max_rows) * max_proj_out * 2);
        m.delta_buf = try m.allocScratch(@as(usize, max_rows) * c.hidden_size * 2);
        m.act_buf = try m.allocScratch(@as(usize, max_rows) * max_act_dim * 2);

        m.gdn_scan_out_buf = try m.allocScratch(@max(@as(usize, 256), @as(usize, max_rows) * c.gdnValDim() * 2));
        m.conv_windows_buf = try m.allocScratch(@as(usize, max_rows) * 16);
        m.scan_steps_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.keep_rows_buf = try m.allocScratch(@as(usize, max_rows) * 4);

        const rope_pairs = @max(@as(usize, 32), c.rope_dims / 2);
        m.attn_q_buf = try m.allocScratch(@as(usize, max_rows) * c.num_heads * c.head_dim * 2);
        m.inv_freq_buf = try m.allocScratch(rope_pairs * 4);
        m.positions_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.kv_slots_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.seq_lens_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.tree_masks_buf = try m.allocScratch(@as(usize, max_rows) * 16);
        m.partial_acc_buf = try m.allocScratch(@as(usize, max_splits) * c.num_heads * c.head_dim * 4);
        m.partial_ml_buf = try m.allocScratch(@as(usize, max_splits) * c.num_heads * 8);
        m.kv_tmp_k_buf = try m.allocScratch(@as(usize, max_rows) * c.num_kv_heads * c.head_dim * 2);
        m.kv_tmp_v_buf = try m.allocScratch(@as(usize, max_rows) * c.num_kv_heads * c.head_dim * 2);

        // Batched logits & argmax buffers (large enough for fused qmv_1_argmax vocab tiles too)
        const argmax_slots = @max(@as(usize, max_rows) * chunks, w.lm_head.tiles());
        m.logits_buf = try m.allocScratch(@as(usize, max_rows) * c.vocab_size * 2);
        m.argmax_val_buf = try m.allocScratch(argmax_slots * 4);
        m.argmax_idx_buf = try m.allocScratch(argmax_slots * 4);
        m.out_tok_buf = try m.allocScratch(@as(usize, max_rows) * 4);
        m.out_val_buf = try m.allocScratch(@as(usize, max_rows) * 4);

        const n_lin = c.numLinearLayers();
        const n_full = c.numFullLayers();
        m.gdn_verify_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * gdn_vl.layer_stride));
        m.gdn_state_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * m.gdn_state_stride));
        m.gdn_hist_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * m.gdn_hist_stride));
        m.gdn_hist_alt_buf = try m.allocScratch(@max(@as(usize, 256), n_lin * m.gdn_hist_stride));
        m.k_cache_buf = try m.allocScratch(@max(@as(usize, 256), n_full * m.kv_layer_stride));
        m.v_cache_buf = try m.allocScratch(@max(@as(usize, 256), n_full * m.kv_layer_stride));

        if (opts.capture_taps) {
            m.taps_layer_stride = align256(@as(usize, max_rows) * c.hidden_size * 2);
            m.taps_buf = try m.allocScratch(tap_layers.len * m.taps_layer_stride);
        }

        // Precompute exact f64 RoPE inverse frequencies (32 pairs for Qwen3.5 partial RoPE, 64 pairs for Qwen2.5 full RoPE)
        const inv_f = m.inv_freq_buf.slice(f32, rope_pairs);
        const rdims: f64 = @floatFromInt(c.rope_dims);
        for (0..rope_pairs) |i| {
            const exp_arg = (2.0 * @as(f64, @floatFromInt(i))) / rdims;
            inv_f[i] = @floatCast(1.0 / std.math.pow(f64, c.rope_theta, exp_arg));
        }

        ctx.commitResidency();
        m.reset();
        return m;
    }

    pub fn deinit(self: *Model) void {
        self.freeScratch();
        self.fence.deinit();
        self.p.deinit();
        self.w.deinit();
    }

    fn allocScratch(self: *Model, len: usize) !Buffer {
        const b = try self.ctx.allocBuffer(len);
        try self.scratch_bufs.append(self.a, b);
        return b;
    }

    fn freeScratch(self: *Model) void {
        for (self.scratch_bufs.items) |b| b.deinit();
        self.scratch_bufs.deinit(self.a);
    }

    pub fn reset(self: *Model) void {
        @memset(self.gdn_state_buf.ptr[0..self.gdn_state_buf.len], 0);
        @memset(self.gdn_hist_buf.ptr[0..self.gdn_hist_buf.len], 0);
        @memset(self.gdn_hist_alt_buf.ptr[0..self.gdn_hist_alt_buf.len], 0);
        self.pos = 0;
    }

    /// Prefill prompt tokens in chunks up to `chunk_size` (<= max_rows).
    /// Returns the greedy next token ID predicted from the final prompt position (and populates `logits_buf`).
    pub fn prefill(self: *Model, prompt: []const u32, chunk_size: u32) !u32 {
        std.debug.assert(prompt.len > 0);
        const cap = @min(chunk_size, self.max_rows);
        var offset: usize = 0;
        var next_tok: u32 = 0;
        while (offset < prompt.len) {
            const n: u32 = @intCast(@min(prompt.len - offset, cap));
            const is_last = (offset + n == prompt.len);
            next_tok = try self.forwardCommitChain(prompt[offset..][0..n], is_last, is_last);
            offset += n;
        }
        return next_tok;
    }

    /// Decode one token (committed immediately) using fused LM-head argmax (zero logits_buf write).
    pub fn step(self: *Model, token: u32) !u32 {
        const one = [1]u32{token};
        return self.forwardCommitChain(&one, true, false);
    }

    /// Decode one token (committed immediately) and materialize full bf16 logits in `logits_buf` (for sampling).
    pub fn stepWithLogits(self: *Model, token: u32) !u32 {
        const one = [1]u32{token};
        return self.forwardCommitChain(&one, true, true);
    }

    /// Speculatively verify a linear chain `tokens[0..rows]` without committing GDN state or `pos`.
    /// Returns the slice of target greedy picks `picks[0..rows]` (in shared memory, valid until next call).
    pub fn verifyChain(self: *Model, tokens: []const u32) ![]const u32 {
        const rows: u32 = @intCast(tokens.len);
        std.debug.assert(rows > 0 and rows <= self.max_rows);
        var parents: [128]i32 = undefined;
        for (0..rows) |r| parents[r] = @as(i32, @intCast(r)) - 1;
        return self.verifyTree(tokens, parents[0..rows], false);
    }

    /// Speculatively verify an arbitrary tree (`parents[0] == -1`, `parents[r] < r`) without committing GDN state or `pos`.
    /// Returns greedy picks `picks[0..rows]`. Call `commitKeep` afterwards with the accepted row indices!
    pub fn verifyTree(self: *Model, tokens: []const u32, parents: []const i32, is_tree: bool) ![]const u32 {
        const rows: u32 = @intCast(tokens.len);
        std.debug.assert(rows > 0 and rows <= self.max_rows and parents.len == rows);
        std.debug.assert(self.pos + rows <= self.max_context);

        const t_start = nowMs();

        const tok_dst = self.tokens_buf.slice(u32, rows);
        const pos_dst = self.positions_buf.slice(u32, rows);
        const slot_dst = self.kv_slots_buf.slice(u32, rows);
        const seq_dst = self.seq_lens_buf.slice(u32, rows);
        const mask_dst = self.tree_masks_buf.slice([4]u32, rows);
        const win_dst = self.conv_windows_buf.slice([4]u32, rows);
        const step_dst = self.scan_steps_buf.slice(u32, rows);

        // Compute depth, positions, KV slots, 128-bit ancestor masks, and 4-tap conv windows
        var depths: [128]u32 = undefined;
        for (0..rows) |r| {
            const p = parents[r];
            depths[r] = if (p < 0) 0 else depths[@intCast(p)] + 1;
            tok_dst[r] = tokens[r];
            pos_dst[r] = self.pos + depths[r];
            slot_dst[r] = self.pos + @as(u32, @intCast(r));
            seq_dst[r] = self.pos + @as(u32, @intCast(r)) + 1;

            // Ancestor mask for FlashAttention
            var mask: [4]u32 = .{ 0, 0, 0, 0 };
            var cur: i32 = @intCast(r);
            while (cur >= 0) {
                const uc: u32 = @intCast(cur);
                mask[uc >> 5] |= (@as(u32, 1) << @intCast(uc & 31));
                cur = parents[uc];
            }
            mask_dst[r] = mask;

            // 4-tap causal conv window walking 3 steps back along tree parents
            var w: [4]u32 = undefined;
            var node: i32 = @intCast(r);
            var back_from_root: u32 = 0;
            var tap_i: usize = 4;
            while (tap_i > 0) {
                tap_i -= 1;
                if (node >= 0) {
                    w[tap_i] = 3 + @as(u32, @intCast(node));
                    node = parents[@intCast(node)];
                } else {
                    // 1 step before root -> history[2], 2 steps -> history[1], 3 steps -> history[0]
                    w[tap_i] = if (back_from_root < 3) 2 - back_from_root else 0;
                    back_from_root += 1;
                }
            }
            win_dst[r] = w;
        }

        const slots_needed = if (is_tree) compileTreeSteps(parents, step_dst) else blk: {
            for (0..rows) |r| step_dst[r] = @intCast(r);
            break :blk @as(u32, 0);
        };

        try self.executeForward(rows, true, false, true, true, is_tree, slots_needed, t_start);
        return self.out_tok_buf.slice(u32, rows);
    }

    /// Commit the accepted path `accepted_rows[0..keep_len]` after `verifyChain` or `verifyTree`.
    /// Updates all linear layers' GDN state + conv history, compacts KV cache if non-contiguous, and advances `pos`.
    pub fn commitKeep(self: *Model, accepted_rows: []const u32) !void {
        const keep_len: u32 = @intCast(accepted_rows.len);
        std.debug.assert(keep_len > 0 and keep_len <= self.max_rows);

        var contiguous = true;
        const step_dst = self.scan_steps_buf.slice(u32, keep_len);
        const keep_dst = self.keep_rows_buf.slice(u32, keep_len);
        for (accepted_rows, 0..) |r, i| {
            step_dst[i] = r;
            keep_dst[i] = r;
            if (r != i) contiguous = false;
        }

        // Compute new 3-tap history sources from [hist0, hist1, hist2, keep_qkv[accepted_rows...]]
        const hist_src0: u32 = if (keep_len >= 3) 3 + accepted_rows[keep_len - 3] else if (keep_len == 2) 2 else 1;
        const hist_src1: u32 = if (keep_len >= 2) 3 + accepted_rows[keep_len - 2] else 2;
        const hist_src2: u32 = 3 + accepted_rows[keep_len - 1];

        const c = self.c;
        const n_lin = c.numLinearLayers();
        if (n_lin == 0 and contiguous) {
            self.pos += keep_len;
            return;
        }
        const gdn_qkv_dim: u32 = @intCast(c.gdnQkvDim());

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // Advance all GDN layers (state + conv history) in one pass
        for (0..n_lin) |lin_idx| {
            const state_ref = self.gdn_state_buf.view(lin_idx * self.gdn_state_stride);
            const hist_ref = self.gdn_hist_buf.view(lin_idx * self.gdn_hist_stride);
            const vbase = lin_idx * self.gdn_vl.layer_stride;

            enc.setPipeline(self.p.gdn_scan_1);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.q_off), 0);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.k_off), 1);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.v_off), 2);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.decay_off), 3);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.mix_off), 4);
            enc.setRef(state_ref, 5);
            enc.setBuffer(self.scan_steps_buf, 0, 6);
            enc.setValue(GdnScanParams{
                .rows = keep_len,
                .nk = @intCast(c.gdn_k_heads),
                .nv = @intCast(c.gdn_v_heads),
                .commit_state = 1,
            }, 7);
            enc.setBuffer(self.gdn_scan_out_buf, 0, 8);
            enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });

            enc.setPipeline(self.p.gdn_commit_history);
            enc.setRef(self.gdn_verify_buf.view(vbase + self.gdn_vl.qkv_off), 0);
            enc.setRef(hist_ref, 1);
            enc.setValue(GdnHistCommitParams{
                .qkv_dim = gdn_qkv_dim,
                .in_proj_stride = gdn_qkv_dim,
                .src0 = hist_src0,
                .src1 = hist_src1,
                .src2 = hist_src2,
            }, 2);
            enc.dispatchThreads(.{ .width = gdn_qkv_dim / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        }

        // If tree accepted rows were non-contiguous, compact K/V cache slots into [pos .. pos + keep_len)
        if (!contiguous) {
            const n_full = c.numFullLayers();
            const row_u4: u32 = @intCast((c.num_kv_heads * c.head_dim * 2) / 16);
            const kp = KvCompactParams{
                .base_pos = self.pos,
                .keep_len = keep_len,
                .row_u4 = row_u4,
            };
            for (0..n_full) |full_idx| {
                const k_ref = self.k_cache_buf.view(full_idx * self.kv_layer_stride);
                const v_ref = self.v_cache_buf.view(full_idx * self.kv_layer_stride);

                enc.setPipeline(self.p.kv_compact_gather);
                enc.setRef(k_ref, 0);
                enc.setRef(v_ref, 1);
                enc.setBuffer(self.keep_rows_buf, 0, 2);
                enc.setBuffer(self.kv_tmp_k_buf, 0, 3);
                enc.setBuffer(self.kv_tmp_v_buf, 0, 4);
                enc.setValue(kp, 5);
                enc.dispatchThreads(.{ .width = row_u4, .height = keep_len, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(self.p.kv_compact_scatter);
                enc.setBuffer(self.kv_tmp_k_buf, 0, 0);
                enc.setBuffer(self.kv_tmp_v_buf, 0, 1);
                enc.setRef(k_ref, 2);
                enc.setRef(v_ref, 3);
                enc.setValue(kp, 4);
                enc.dispatchThreads(.{ .width = row_u4, .height = keep_len, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
                enc.barrier();
            }
        }

        enc.end();
        cb.commit();
        try self.ctx.waitIdle();
        self.pos += keep_len;
    }

    pub fn readLogit(self: *const Model, token_id: usize) f32 {
        const raw = self.logits_buf.slice(u16, self.c.vocab_size)[token_id];
        return @bitCast(@as(u32, raw) << 16);
    }

    /// Sample one token from row `row` of `logits_buf` using `params` (greedy fast-path when temp == 0).
    pub fn sampleRow(self: *const Model, row: usize, params: samp_mod.SamplingParams, rng: *samp_mod.Rng) !u32 {
        const greedy = self.out_tok_buf.slice(u32, row + 1)[row];
        if (params.temperature <= 1e-6) return greedy;
        const max_v = self.out_val_buf.slice(f32, row + 1)[row];
        const all_logits = self.logits_buf.slice(u16, (row + 1) * self.c.vocab_size);
        const row_logits = all_logits[row * self.c.vocab_size ..][0..self.c.vocab_size];
        return samp_mod.sampleLogits(self.a, row_logits, greedy, max_v, params, rng);
    }

    /// Public projection helper reused by multi-stream batching.
    pub fn dispatchQkPub(self: *const Model, enc: *mtl.Encoder, x: BufferRef, w: QkMatrix, sums: BufferRef, y: BufferRef, rows: u32, parts: u32) void {
        self.dispatchQk(enc, x, w, sums, y, rows, parts);
    }

    /// Return a host-readable bf16 slice for captured tap `tap_idx` (0..4) at row `row`.
    pub fn tapRowBf16(self: *const Model, tap_idx: usize, row: usize) []const u16 {
        const tb = self.taps_buf orelse unreachable;
        const off = tap_idx * self.taps_layer_stride + row * self.c.hidden_size * 2;
        const ptr: [*]const u16 = @ptrCast(@alignCast(tb.ptr + off));
        return ptr[0..self.c.hidden_size];
    }

    fn dispatchQk(self: *const Model, enc: *mtl.Encoder, x: BufferRef, w: QkMatrix, sums: BufferRef, y: BufferRef, rows: u32, parts: u32) void {
        if (rows == 1 and use_qmv_1) {
            enc.setPipeline(self.p.qmv_1);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(y, 2);
            const params = QkParams{
                .M = 1,
                .K = @intCast(w.k),
                .N = @intCast(w.n),
                .G = @intCast(w.groups()),
                .first_row = 0,
                .sums_stride = @intCast(self.sums_stride),
                .tile_first = 0,
            };
            enc.setValue(params, 3);
            enc.dispatchGroups(.{ .width = w.tiles(), .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            return;
        }
        const total_tiles = w.tiles();
        const n_parts = @max(1, parts);

        var first_row: u32 = 0;
        while (first_row < rows) {
            const rem = rows - first_row;
            const step_rows: u32 = if (rows <= 48) rem else @min(rem, 32);
            const pso = if (step_rows <= 16) self.p.qk_16 else if (step_rows <= 32) self.p.qk_32 else self.p.qk_48;
            enc.setPipeline(pso);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(sums, 2);
            enc.setRef(y, 3);

            var part: u32 = 0;
            while (part < n_parts) : (part += 1) {
                const t0 = total_tiles * part / n_parts;
                const t1 = total_tiles * (part + 1) / n_parts;
                if (t1 <= t0) continue;
                const params = QkParams{
                    .M = @intCast(step_rows),
                    .K = @intCast(w.k),
                    .N = @intCast(w.n),
                    .G = @intCast(w.groups()),
                    .first_row = @intCast(first_row),
                    .sums_stride = @intCast(self.sums_stride),
                    .tile_first = t0,
                };
                enc.setValue(params, 4);
                enc.dispatchGroups(.{ .width = t1 - t0, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            }
            first_row += step_rows;
        }
    }

    fn forwardCommitChain(self: *Model, tokens: []const u32, want_logits: bool, need_full_logits: bool) !u32 {
        const rows: u32 = @intCast(tokens.len);
        std.debug.assert(rows > 0 and rows <= self.max_rows);
        std.debug.assert(self.pos + rows <= self.max_context);

        const t_start = nowMs();

        const tok_dst = self.tokens_buf.slice(u32, rows);
        const pos_dst = self.positions_buf.slice(u32, rows);
        const slot_dst = self.kv_slots_buf.slice(u32, rows);
        const seq_dst = self.seq_lens_buf.slice(u32, rows);
        const win_dst = self.conv_windows_buf.slice([4]u32, rows);
        const step_dst = self.scan_steps_buf.slice(u32, rows);

        for (0..rows) |r| {
            const ur: u32 = @intCast(r);
            tok_dst[r] = tokens[r];
            pos_dst[r] = self.pos + ur;
            slot_dst[r] = self.pos + ur;
            seq_dst[r] = self.pos + ur + 1;
            win_dst[r] = .{ ur, ur + 1, ur + 2, ur + 3 };
            step_dst[r] = ur;
        }

        try self.executeForward(rows, want_logits, true, false, need_full_logits, false, 0, t_start);
        self.pos += rows;
        return if (want_logits) self.out_tok_buf.slice(u32, 1)[0] else 0;
    }

    fn executeForward(
        self: *Model,
        rows: u32,
        want_logits: bool,
        commit_immediately: bool,
        all_row_logits: bool,
        need_full_logits: bool,
        is_tree: bool,
        scan_slots: u32,
        t_start: f64,
    ) !void {
        const c = self.c;
        const gdn_qkv_dim: u32 = @intCast(c.gdnQkvDim());
        const gdn_val_dim: u32 = @intCast(c.gdnValDim());
        const gdn_all_stride: u32 = @intCast(c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads);
        const attn_qkv_stride: u32 = @intCast(c.qProjOut() + 2 * c.kvProjOut());

        const use_fused_gdn_1 = (rows == 1 and commit_immediately and use_fused_1 and c.numLinearLayers() > 0 and !self.tape_capture);
        const use_qmv_fused_1 = (rows == 1 and use_qmv_1 and use_fused_1);
        const norm_sums_stride: u32 = if (rows == 1 and use_qmv_1) 0 else self.sums_stride;

        const total_hist: u32 = 3 + rows;
        const hist_src0 = total_hist - 3;
        const hist_src1 = total_hist - 2;
        const hist_src2 = total_hist - 1;

        const scan_pso = if (scan_slots == 0)
            self.p.gdn_scan_1
        else if (scan_slots <= 2)
            self.p.gdn_scan_2
        else if (scan_slots <= 4)
            self.p.gdn_scan_4
        else
            self.p.gdn_scan_8;

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // 1. Embedding lookup -> h_buf
        enc.setPipeline(self.p.embed_lookup);
        enc.setBuffer(self.tokens_buf, 0, 0);
        enc.setRef(self.w.embed.codes, 1);
        enc.setRef(self.w.embed.scales, 2);
        enc.setRef(self.w.embed.biases, 3);
        enc.setBuffer(self.h_buf, 0, 4);
        enc.setValue(EmbedParams{ .rows = rows, .hidden = @intCast(c.hidden_size) }, 5);
        enc.dispatchThreads(.{ .width = c.hidden_size / 8, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        enc.barrier();

        // 2. Layer 0 input_layernorm + group sums -> normed_buf, sums_buf
        enc.setPipeline(self.p.rms_norm_sums);
        enc.setBuffer(self.h_buf, 0, 0);
        enc.setRef(self.w.layers[0].input_norm, 1);
        enc.setBuffer(self.normed_buf, 0, 2);
        enc.setBuffer(self.sums_buf, 0, 3);
        enc.setValue(NormParams{
            .rows = rows,
            .hidden = @intCast(c.hidden_size),
            .sums_stride = self.sums_stride,
            .eps = c.rms_eps,
        }, 4);
        enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        var lin_idx: usize = 0;
        var full_idx: usize = 0;
        var tap_cursor: usize = 0;
        const cur_max_seq: u32 = self.pos + rows;

        for (0..c.num_layers) |li| {
            const lw = self.w.layers[li];

            if (lw.kind == .linear) {
                const state_ref = self.gdn_state_buf.view(lin_idx * self.gdn_state_stride);
                const hist_ref = self.gdn_hist_buf.view(lin_idx * self.gdn_hist_stride);
                const hist_alt_ref = self.gdn_hist_alt_buf.view(lin_idx * self.gdn_hist_stride);
                // When not committing immediately (speculative verify) or capturing tape, each linear layer gets its own slice;
                // when committing immediately without tape capture, reuse slice 0 to stay hot in SLC!
                const vbase = (if (commit_immediately and !self.tape_capture) 0 else lin_idx) * self.gdn_vl.layer_stride;
                const q_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.q_off);
                const k_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.k_off);
                const v_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.v_off);
                const dec_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.decay_off);
                const mix_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.mix_off);
                const qkv_save_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.qkv_off);
                lin_idx += 1;

                self.dispatchQk(&enc, self.normed_buf.view(0), lw.linear.in_proj_all, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                if (use_fused_gdn_1) {
                    // Super-fused SRAM-resident GDN step: conv + scan + norm_gate in 1 dispatch!
                    enc.setPipeline(self.p.gdn_fused_step_1);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(hist_ref, 1);
                    enc.setRef(hist_alt_ref, 2);
                    enc.setRef(lw.linear.conv_weight, 3);
                    enc.setRef(lw.linear.a_log, 4);
                    enc.setRef(lw.linear.dt_bias, 5);
                    enc.setRef(lw.linear.gdn_norm, 6);
                    enc.setRef(state_ref, 7);
                    enc.setBuffer(self.act_buf, 0, 8);
                    enc.setBuffer(self.sums_buf, 0, 9);
                    enc.setValue(GdnFusedStepParams{
                        .nk = @intCast(c.gdn_k_heads),
                        .nv = @intCast(c.gdn_v_heads),
                        .qkv_dim = gdn_qkv_dim,
                        .in_proj_stride = gdn_all_stride,
                        .z_offset = gdn_qkv_dim,
                        .b_offset = gdn_qkv_dim + gdn_val_dim,
                        .a_offset = gdn_qkv_dim + gdn_val_dim + @as(u32, @intCast(c.gdn_v_heads)),
                        .sums_stride = self.sums_stride,
                        .write_sums = if (use_qmv_1) 0 else 1,
                        .eps = c.rms_eps,
                    }, 10);
                    enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                    enc.barrier();
                } else {
                    enc.setPipeline(self.p.gdn_conv);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(hist_ref, 1);
                    enc.setRef(lw.linear.conv_weight, 2);
                    enc.setRef(lw.linear.a_log, 3);
                    enc.setRef(lw.linear.dt_bias, 4);
                    enc.setBuffer(self.conv_windows_buf, 0, 5);
                    enc.setValue(GdnConvParams{
                        .rows = rows,
                        .nk = @intCast(c.gdn_k_heads),
                        .nv = @intCast(c.gdn_v_heads),
                        .qkv_dim = gdn_qkv_dim,
                        .in_proj_stride = gdn_all_stride,
                        .b_offset = gdn_qkv_dim + gdn_val_dim,
                        .a_offset = gdn_qkv_dim + gdn_val_dim + @as(u32, @intCast(c.gdn_v_heads)),
                        .commit_single = if (commit_immediately and rows == 1) 1 else 0,
                        .save_qkv = if (!commit_immediately or self.tape_capture) 1 else 0,
                    }, 6);
                    enc.setRef(q_ref, 7);
                    enc.setRef(k_ref, 8);
                    enc.setRef(v_ref, 9);
                    enc.setRef(dec_ref, 10);
                    enc.setRef(mix_ref, 11);
                    enc.setRef(qkv_save_ref, 12);
                    enc.dispatchGroups(.{ .width = 1, .height = 2 * c.gdn_k_heads + c.gdn_v_heads, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();

                    if (self.tape_capture and scan_slots == 0) {
                        const corr_ref = self.gdn_verify_buf.view(vbase + self.gdn_vl.corr_off);
                        enc.setPipeline(self.p.gdn_scan_record_corr);
                        enc.setRef(q_ref, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setRef(dec_ref, 3);
                        enc.setRef(mix_ref, 4);
                        enc.setRef(state_ref, 5);
                        enc.setBuffer(self.scan_steps_buf, 0, 6);
                        enc.setValue(GdnScanParams{
                            .rows = rows,
                            .nk = @intCast(c.gdn_k_heads),
                            .nv = @intCast(c.gdn_v_heads),
                            .commit_state = if (commit_immediately) 1 else 0,
                        }, 7);
                        enc.setBuffer(self.gdn_scan_out_buf, 0, 8);
                        enc.setRef(corr_ref, 9);
                        enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });
                    } else {
                        enc.setPipeline(scan_pso);
                        enc.setRef(q_ref, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setRef(dec_ref, 3);
                        enc.setRef(mix_ref, 4);
                        enc.setRef(state_ref, 5);
                        enc.setBuffer(self.scan_steps_buf, 0, 6);
                        enc.setValue(GdnScanParams{
                            .rows = rows,
                            .nk = @intCast(c.gdn_k_heads),
                            .nv = @intCast(c.gdn_v_heads),
                            .commit_state = if (commit_immediately) 1 else 0,
                        }, 7);
                        enc.setBuffer(self.gdn_scan_out_buf, 0, 8);
                        enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });
                    }

                    if (commit_immediately and rows > 1) {
                        enc.setPipeline(self.p.gdn_commit_history);
                        enc.setBuffer(self.proj_big_buf, 0, 0);
                        enc.setRef(hist_ref, 1);
                        enc.setValue(GdnHistCommitParams{
                            .qkv_dim = gdn_qkv_dim,
                            .in_proj_stride = gdn_all_stride,
                            .src0 = hist_src0,
                            .src1 = hist_src1,
                            .src2 = hist_src2,
                        }, 2);
                        enc.dispatchThreads(.{ .width = gdn_qkv_dim / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    }
                    enc.barrier();

                    enc.setPipeline(self.p.gdn_norm_gate_sums);
                    enc.setBuffer(self.gdn_scan_out_buf, 0, 0);
                    enc.setBuffer(self.proj_big_buf, 0, 1);
                    enc.setRef(lw.linear.gdn_norm, 2);
                    enc.setBuffer(self.act_buf, 0, 3);
                    enc.setBuffer(self.sums_buf, 0, 4);
                    enc.setValue(GdnNormParams{
                        .rows = rows,
                        .nv = @intCast(c.gdn_v_heads),
                        .dv = @intCast(c.gdn_v_dim),
                        .in_proj_stride = gdn_all_stride,
                        .z_offset = gdn_qkv_dim,
                        .sums_stride = self.sums_stride,
                        .eps = c.rms_eps,
                    }, 5);
                    enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();
                }

                self.dispatchQk(&enc, self.act_buf.view(0), lw.linear.out_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                enc.barrier();
            } else {
                const k_ref = self.k_cache_buf.view(full_idx * self.kv_layer_stride);
                const v_ref = self.v_cache_buf.view(full_idx * self.kv_layer_stride);
                full_idx += 1;

                self.dispatchQk(&enc, self.normed_buf.view(0), lw.full.qkv_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
                if (c.arch == .qwen2_dense and c.head_dim == 128) {
                    const bias_ref = lw.full.qkv_bias orelse self.w.final_norm;
                    enc.setPipeline(self.p.attn_std128_rope_store);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(bias_ref, 1);
                    enc.setBuffer(self.inv_freq_buf, 0, 2);
                    enc.setBuffer(self.positions_buf, 0, 3);
                    enc.setBuffer(self.kv_slots_buf, 0, 4);
                    enc.setBuffer(self.attn_q_buf, 0, 5);
                    enc.setRef(k_ref, 6);
                    enc.setRef(v_ref, 7);
                    enc.setValue(Std128PrepParams{
                        .rows = rows,
                        .num_heads = @intCast(c.num_heads),
                        .num_kv_heads = @intCast(c.num_kv_heads),
                        .qkv_stride = attn_qkv_stride,
                        .has_bias = if (lw.full.qkv_bias != null) 1 else 0,
                    }, 8);
                    enc.dispatchGroups(.{ .width = c.num_heads + c.num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();

                    enc.setPipeline(self.p.attn_std128_flash);
                    enc.setBuffer(self.attn_q_buf, 0, 0);
                    enc.setRef(k_ref, 1);
                    enc.setRef(v_ref, 2);
                    enc.setBuffer(self.seq_lens_buf, 0, 3);
                    enc.setBuffer(self.tree_masks_buf, 0, 4);
                    enc.setBuffer(self.act_buf, 0, 5);
                    enc.setBuffer(self.sums_buf, 0, 6);
                    enc.setValue(AttnFlashParams{
                        .rows = rows,
                        .num_heads = @intCast(c.num_heads),
                        .num_kv_heads = @intCast(c.num_kv_heads),
                        .qkv_stride = attn_qkv_stride,
                        .sums_stride = self.sums_stride,
                        .scale = scale,
                        .splits = 1,
                        .base_pos = self.pos,
                        .is_tree = if (is_tree) 1 else 0,
                    }, 7);
                    enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                    enc.barrier();
                } else {
                    enc.setPipeline(self.p.attn_qk_rope_store);
                    enc.setBuffer(self.proj_big_buf, 0, 0);
                    enc.setRef(lw.full.q_norm, 1);
                    enc.setRef(lw.full.k_norm, 2);
                    enc.setBuffer(self.inv_freq_buf, 0, 3);
                    enc.setBuffer(self.positions_buf, 0, 4);
                    enc.setBuffer(self.kv_slots_buf, 0, 5);
                    enc.setBuffer(self.attn_q_buf, 0, 6);
                    enc.setRef(k_ref, 7);
                    enc.setRef(v_ref, 8);
                    enc.setValue(AttnPrepParams{
                        .rows = rows,
                        .num_heads = @intCast(c.num_heads),
                        .num_kv_heads = @intCast(c.num_kv_heads),
                        .qkv_stride = attn_qkv_stride,
                        .eps = c.rms_eps,
                    }, 9);
                    enc.dispatchGroups(.{ .width = c.num_heads + c.num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                    enc.barrier();

                    if (rows == 1 and use_split_k and cur_max_seq > 256 and !is_tree) {
                        const splits: u32 = @min(max_splits, @max(4, (cur_max_seq + 63) / 64));
                        const fp = AttnFlashParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .sums_stride = self.sums_stride,
                            .scale = scale,
                            .splits = splits,
                            .base_pos = self.pos,
                            .is_tree = 0,
                        };
                        enc.setPipeline(self.p.attn_flash_split);
                        enc.setBuffer(self.attn_q_buf, 0, 0);
                        enc.setRef(k_ref, 1);
                        enc.setRef(v_ref, 2);
                        enc.setBuffer(self.seq_lens_buf, 0, 3);
                        enc.setBuffer(self.partial_acc_buf, 0, 4);
                        enc.setBuffer(self.partial_ml_buf, 0, 5);
                        enc.setValue(fp, 6);
                        enc.dispatchGroups(.{ .width = splits, .height = c.num_heads, .depth = rows }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();

                        enc.setPipeline(self.p.attn_merge_gate_sums);
                        enc.setBuffer(self.partial_acc_buf, 0, 0);
                        enc.setBuffer(self.partial_ml_buf, 0, 1);
                        enc.setBuffer(self.proj_big_buf, 0, 2);
                        enc.setBuffer(self.act_buf, 0, 3);
                        enc.setBuffer(self.sums_buf, 0, 4);
                        enc.setValue(fp, 5);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                        enc.barrier();
                    } else {
                        enc.setPipeline(self.p.attn_flash_fused);
                        enc.setBuffer(self.attn_q_buf, 0, 0);
                        enc.setBuffer(self.proj_big_buf, 0, 1);
                        enc.setRef(k_ref, 2);
                        enc.setRef(v_ref, 3);
                        enc.setBuffer(self.seq_lens_buf, 0, 4);
                        enc.setBuffer(self.tree_masks_buf, 0, 5);
                        enc.setBuffer(self.act_buf, 0, 6);
                        enc.setBuffer(self.sums_buf, 0, 7);
                        enc.setValue(AttnFlashParams{
                            .rows = rows,
                            .num_heads = @intCast(c.num_heads),
                            .num_kv_heads = @intCast(c.num_kv_heads),
                            .qkv_stride = attn_qkv_stride,
                            .sums_stride = self.sums_stride,
                            .scale = scale,
                            .splits = 1,
                            .base_pos = self.pos,
                            .is_tree = if (is_tree) 1 else 0,
                        }, 8);
                        enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                        enc.barrier();
                    }
                }

                self.dispatchQk(&enc, self.act_buf.view(0), lw.full.o_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
                enc.barrier();
            }

            // Post-attention residual + norm + sums
            enc.setPipeline(self.p.add_rms_norm_sums);
            enc.setBuffer(self.h_buf, 0, 0);
            enc.setBuffer(self.delta_buf, 0, 1);
            enc.setRef(lw.post_attn_norm, 2);
            enc.setBuffer(self.normed_buf, 0, 3);
            enc.setBuffer(self.sums_buf, 0, 4);
            enc.setValue(NormParams{
                .rows = rows,
                .hidden = @intCast(c.hidden_size),
                .sums_stride = norm_sums_stride,
                .eps = c.rms_eps,
            }, 5);
            enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            // MLP gate+up -> SwiGLU -> down
            if (use_qmv_fused_1) {
                // Super-fused Gate+Up+SiLU*Mul: 4x X reuse, zero proj_big_buf traffic, zero gu_activate_sums dispatch!
                enc.setPipeline(self.p.qmv_1_silu_mul);
                enc.setBuffer(self.normed_buf, 0, 0);
                enc.setRef(lw.gate_up.ref, 1);
                enc.setBuffer(self.act_buf, 0, 2);
                enc.setValue(QkParams{
                    .M = 1,
                    .K = @intCast(lw.gate_up.k),
                    .N = @intCast(c.intermediate_size),
                    .G = @intCast(lw.gate_up.groups()),
                    .first_row = 0,
                    .sums_stride = 0,
                    .tile_first = 0,
                }, 3);
                enc.dispatchGroups(.{ .width = c.intermediate_size / 32, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();
            } else {
                self.dispatchQk(&enc, self.normed_buf.view(0), lw.gate_up, self.sums_buf.view(0), self.proj_big_buf.view(0), rows, self.gu_parts);
                enc.barrier();

                enc.setPipeline(self.p.gu_activate_sums);
                enc.setBuffer(self.proj_big_buf, 0, 0);
                enc.setBuffer(self.act_buf, 0, 1);
                enc.setBuffer(self.sums_buf, 0, 2);
                enc.setValue(MlpActParams{
                    .rows = rows,
                    .inter = @intCast(c.intermediate_size),
                    .sums_stride = self.sums_stride,
                }, 3);
                enc.dispatchGroups(.{ .width = c.intermediate_size / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();
            }

            self.dispatchQk(&enc, self.act_buf.view(0), lw.down_proj, self.sums_buf.view(0), self.delta_buf.view(0), rows, 1);
            enc.barrier();

            // Post-MLP residual + next layer norm (or final_norm)
            if (li + 1 < c.num_layers) {
                enc.setPipeline(self.p.add_rms_norm_sums);
                enc.setBuffer(self.h_buf, 0, 0);
                enc.setBuffer(self.delta_buf, 0, 1);
                enc.setRef(self.w.layers[li + 1].input_norm, 2);
                enc.setBuffer(self.normed_buf, 0, 3);
                enc.setBuffer(self.sums_buf, 0, 4);
                enc.setValue(NormParams{
                    .rows = rows,
                    .hidden = @intCast(c.hidden_size),
                    .sums_stride = norm_sums_stride,
                    .eps = c.rms_eps,
                }, 5);
                enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();

                // Capture hidden-state tap after specified layers for DFlash2 drafter
                if (self.taps_buf) |tb| {
                    if (tap_cursor < tap_layers.len and tap_layers[tap_cursor] == li) {
                        const u4_count: u32 = @intCast((@as(usize, rows) * c.hidden_size * 2) / 16);
                        enc.setPipeline(self.p.copy_u4);
                        enc.setBuffer(self.h_buf, 0, 0);
                        enc.setBuffer(tb, tap_cursor * self.taps_layer_stride, 1);
                        enc.setValue(CopyParams{ .count = u4_count }, 2);
                        enc.dispatchThreads(.{ .width = u4_count, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                        enc.barrier();
                        tap_cursor += 1;
                    }
                }
            } else if (want_logits) {
                const norm_rows: u32 = if (all_row_logits) rows else 1;
                const h_off: usize = if (all_row_logits) 0 else @as(usize, rows - 1) * c.hidden_size * 2;
                enc.setPipeline(self.p.add_rms_norm_sums);
                enc.setBuffer(self.h_buf, h_off, 0);
                enc.setBuffer(self.delta_buf, h_off, 1);
                enc.setRef(self.w.final_norm, 2);
                enc.setBuffer(self.normed_buf, 0, 3);
                enc.setBuffer(self.sums_buf, 0, 4);
                enc.setValue(NormParams{
                    .rows = norm_rows,
                    .hidden = @intCast(c.hidden_size),
                    .sums_stride = if (norm_rows == 1 and use_qmv_1) 0 else self.sums_stride,
                    .eps = c.rms_eps,
                }, 5);
                enc.dispatchGroups(.{ .width = norm_rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();
            }

            if ((li + 1) % self.commit_every == 0 and li + 1 < c.num_layers) {
                enc.updateFence(self.fence);
                enc.end();
                cb.commit();
                cb = self.ctx.beginCommandBuffer();
                enc = cb.beginCompute();
                enc.waitFence(self.fence);
            }
        }

        // 3. LM Head + Batched Multi-Core Argmax
        if (want_logits) {
            const logit_rows: u32 = if (all_row_logits) rows else 1;
            if (logit_rows == 1 and use_qmv_fused_1 and !need_full_logits) {
                // Super-fused LM Head + Stage-1 Argmax: eliminates 496 KB logits write/read + argmax_stage1!
                const lm_tiles: u32 = @intCast(self.w.lm_head.tiles());
                enc.setPipeline(self.p.qmv_1_argmax);
                enc.setBuffer(self.normed_buf, 0, 0);
                enc.setRef(self.w.lm_head.ref, 1);
                enc.setBuffer(self.argmax_val_buf, 0, 2);
                enc.setBuffer(self.argmax_idx_buf, 0, 3);
                enc.setValue(QkParams{
                    .M = 1,
                    .K = @intCast(self.w.lm_head.k),
                    .N = @intCast(c.vocab_size),
                    .G = @intCast(self.w.lm_head.groups()),
                    .first_row = 0,
                    .sums_stride = 0,
                    .tile_first = 0,
                }, 4);
                enc.dispatchGroups(.{ .width = lm_tiles, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();

                const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = lm_tiles, .rows = 1 };
                enc.setPipeline(self.p.argmax_stage2);
                enc.setBuffer(self.argmax_val_buf, 0, 0);
                enc.setBuffer(self.argmax_idx_buf, 0, 1);
                enc.setBuffer(self.out_tok_buf, 0, 2);
                enc.setBuffer(self.out_val_buf, 0, 3);
                enc.setValue(ap, 4);
                enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            } else {
                self.dispatchQk(&enc, self.normed_buf.view(0), self.w.lm_head, self.sums_buf.view(0), self.logits_buf.view(0), logit_rows, 1);
                enc.barrier();

                const chunks: u32 = @intCast((c.vocab_size + 1023) / 1024);
                const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = chunks, .rows = logit_rows };
                enc.setPipeline(self.p.argmax_stage1);
                enc.setBuffer(self.logits_buf, 0, 0);
                enc.setBuffer(self.argmax_val_buf, 0, 1);
                enc.setBuffer(self.argmax_idx_buf, 0, 2);
                enc.setValue(ap, 3);
                enc.dispatchGroups(.{ .width = chunks, .height = logit_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(self.p.argmax_stage2);
                enc.setBuffer(self.argmax_val_buf, 0, 0);
                enc.setBuffer(self.argmax_idx_buf, 0, 1);
                enc.setBuffer(self.out_tok_buf, 0, 2);
                enc.setBuffer(self.out_val_buf, 0, 3);
                enc.setValue(ap, 4);
                enc.dispatchGroups(.{ .width = 1, .height = logit_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            }
        }

        enc.end();
        cb.commit();
        const encode_ms = nowMs() - t_start;

        try self.ctx.waitIdle();
        if (use_fused_gdn_1) {
            std.mem.swap(Buffer, &self.gdn_hist_buf, &self.gdn_hist_alt_buf);
        }
        const wall_ms = nowMs() - t_start;
        self.last_timing = .{ .wall_ms = wall_ms, .encode_ms = encode_ms };
    }
};

/// Compile a topological parent array (`parents[0] = -1`, `parents[r] < r`) into a depth-first
/// GDN scan schedule (`steps[i] = row | (restore_slot << 16) | (save_slot << 24)`).
/// Returns the maximum number of concurrent threadgroup state slots needed (1..8).
fn compileTreeSteps(parents: []const i32, steps_out: []u32) u32 {
    const n = parents.len;
    var first_child: [128]i32 = @splat(-1);
    var next_sibling: [128]i32 = @splat(-1);
    var child_count: [128]u32 = @splat(0);

    // Build adjacency lists (reverse iteration preserves ascending child order)
    var i: usize = n;
    while (i > 1) {
        i -= 1;
        const p: usize = @intCast(parents[i]);
        next_sibling[i] = first_child[p];
        first_child[p] = @intCast(i);
        child_count[p] += 1;
    }

    const Frame = struct { node: u32, restore_slot: u32, free_after_restore: bool };
    var stack: [128]Frame = undefined;
    var sp: usize = 0;
    stack[sp] = .{ .node = 0, .restore_slot = 0, .free_after_restore = false };
    sp += 1;

    var slot_used: [8]bool = @splat(false);
    var max_slots: u32 = 0;
    var out_i: usize = 0;

    while (sp > 0) {
        sp -= 1;
        const f = stack[sp];
        if (f.free_after_restore and f.restore_slot > 0) {
            slot_used[f.restore_slot - 1] = false;
        }

        var save_slot: u32 = 0;
        if (child_count[f.node] > 1) {
            // Allocate a free TG slot to save state for non-first children
            for (0..8) |s| {
                if (!slot_used[s]) {
                    slot_used[s] = true;
                    save_slot = @intCast(s + 1);
                    if (save_slot > max_slots) max_slots = save_slot;
                    break;
                }
            }
        }

        steps_out[out_i] = f.node | (f.restore_slot << 16) | (save_slot << 24);
        out_i += 1;

        // Gather children in order, then push onto DFS stack in reverse so child 0 runs immediately
        var kids: [128]u32 = undefined;
        var k_cnt: usize = 0;
        var ch = first_child[f.node];
        while (ch >= 0) {
            kids[k_cnt] = @intCast(ch);
            k_cnt += 1;
            ch = next_sibling[@intCast(ch)];
        }
        var ki: usize = k_cnt;
        while (ki > 0) {
            ki -= 1;
            const is_first = (ki == 0);
            const is_last = (ki + 1 == k_cnt);
            stack[sp] = .{
                .node = kids[ki],
                .restore_slot = if (is_first) 0 else save_slot,
                .free_after_restore = (!is_first and is_last),
            };
            sp += 1;
        }
    }

    return max_slots;
}

fn align256(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}
