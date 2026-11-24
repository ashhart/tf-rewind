//! HuggingFace config.json parser supporting both:
//!   - Qwen3.5 / Qwen3.8 Hybrid GDN + GQA (4-bit MLX checkpoints)
//!   - Qwen2 / Qwen2.5 / Llama Standard Dense Transformers (BF16 or 4-bit MLX checkpoints)
const std = @import("std");
const safetensors = @import("safetensors.zig");

pub const Arch = enum { qwen35_hybrid, qwen2_dense };
pub const LayerKind = enum { linear, full };
pub const max_layers = 128;
pub const max_eos = 8;

pub const ModelConfig = struct {
    arch: Arch = .qwen35_hybrid,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    vocab_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // Gated DeltaNet (linear_attention) geometry
    gdn_k_heads: usize,
    gdn_v_heads: usize,
    gdn_k_dim: usize,
    gdn_v_dim: usize,
    conv_kernel: usize,
    // Normalization & RoPE
    rms_eps: f32,
    rope_dims: usize,
    rope_theta: f64,
    // Quantization & embeddings
    bits: usize = 4,
    group_size: usize = 64,
    tie_word_embeddings: bool = false,
    attn_output_gate: bool = true,
    has_qk_norm: bool = true,
    has_qkv_bias: bool = false,
    // Layer schedule & EOS
    layer_kinds: [max_layers]LayerKind = @splat(.linear),
    eos_ids: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,
    max_position: usize = 262144,

    pub fn qProjOut(self: ModelConfig) usize {
        return if (self.attn_output_gate) 2 * self.num_heads * self.head_dim else self.num_heads * self.head_dim;
    }

    pub fn kvProjOut(self: ModelConfig) usize {
        return self.num_kv_heads * self.head_dim;
    }

    pub fn gdnKeyDim(self: ModelConfig) usize {
        return self.gdn_k_heads * self.gdn_k_dim;
    }

    pub fn gdnValDim(self: ModelConfig) usize {
        return self.gdn_v_heads * self.gdn_v_dim;
    }

    pub fn gdnQkvDim(self: ModelConfig) usize {
        return 2 * self.gdnKeyDim() + self.gdnValDim();
    }

    pub fn numFullLayers(self: ModelConfig) usize {
        var n: usize = 0;
        for (self.layer_kinds[0..self.num_layers]) |k| if (k == .full) {
            n += 1;
        };
        return n;
    }

    pub fn numLinearLayers(self: ModelConfig) usize {
        return self.num_layers - self.numFullLayers();
    }

    pub fn isEos(self: ModelConfig, tok: u32) bool {
        for (self.eos_ids[0..self.eos_count]) |e| {
            if (e == tok) return true;
        }
        return false;
    }

    pub fn load(a: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
        const path = try std.fs.path.join(a, &.{ model_dir, "config.json" });
        defer a.free(path);
        const bytes = try safetensors.mapFile(a, path);
        defer safetensors.unmapFile(bytes);

        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        const tc = if (root.get("text_config")) |v| v.object else root;

        const num_layers = getUsize(tc, "num_hidden_layers");
        if (num_layers > max_layers) return error.TooManyLayers;

        const mtype = if (tc.get("model_type")) |mv| mv.string else "qwen3_5";
        const is_qwen2_dense = std.mem.eql(u8, mtype, "qwen2") or std.mem.eql(u8, mtype, "llama") or std.mem.eql(u8, mtype, "mistral");
        const arch: Arch = if (is_qwen2_dense) .qwen2_dense else .qwen35_hybrid;

        const hidden_size = getUsize(tc, "hidden_size");
        const num_heads = getUsize(tc, "num_attention_heads");
        const head_dim = getUsizeOr(tc, "head_dim", hidden_size / num_heads);
        var rope_theta: f64 = if (is_qwen2_dense) 1_000_000.0 else 10_000_000.0;
        var partial_rotary: f64 = if (is_qwen2_dense) 1.0 else 0.25;
        if (tc.get("rope_parameters")) |rp| {
            const ro = rp.object;
            if (ro.get("rope_theta")) |t| rope_theta = asF64(t);
            if (ro.get("partial_rotary_factor")) |p| partial_rotary = asF64(p);
        } else {
            if (tc.get("rope_theta")) |t| rope_theta = asF64(t);
            if (tc.get("partial_rotary_factor")) |p| partial_rotary = asF64(p);
        }

        var cfg = ModelConfig{
            .arch = arch,
            .hidden_size = hidden_size,
            .intermediate_size = getUsize(tc, "intermediate_size"),
            .num_layers = num_layers,
            .vocab_size = getUsize(tc, "vocab_size"),
            .num_heads = num_heads,
            .num_kv_heads = getUsize(tc, "num_key_value_heads"),
            .head_dim = head_dim,
            .gdn_k_heads = if (is_qwen2_dense) 0 else getUsizeOr(tc, "linear_num_key_heads", 16),
            .gdn_v_heads = if (is_qwen2_dense) 0 else getUsizeOr(tc, "linear_num_value_heads", 16),
            .gdn_k_dim = getUsizeOr(tc, "linear_key_head_dim", 128),
            .gdn_v_dim = getUsizeOr(tc, "linear_value_head_dim", 128),
            .conv_kernel = getUsizeOr(tc, "linear_conv_kernel_dim", 4),
            .rms_eps = @floatCast(if (tc.get("rms_norm_eps")) |e| asF64(e) else 1e-6),
            .rope_dims = @intFromFloat(@round(@as(f64, @floatFromInt(head_dim)) * partial_rotary)),
            .rope_theta = rope_theta,
            .tie_word_embeddings = if (tc.get("tie_word_embeddings")) |v| v.bool else false,
            .attn_output_gate = if (is_qwen2_dense) false else (if (tc.get("attn_output_gate")) |v| v.bool else true),
            .has_qk_norm = !is_qwen2_dense,
            .has_qkv_bias = std.mem.eql(u8, mtype, "qwen2"),
            .max_position = getUsizeOr(tc, "max_position_embeddings", 262144),
        };

        if (root.get("quantization")) |qv| {
            const qo = qv.object;
            cfg.bits = getUsizeOr(qo, "bits", 4);
            cfg.group_size = getUsizeOr(qo, "group_size", 64);
        }

        // Layer schedule
        if (is_qwen2_dense) {
            for (0..num_layers) |i| cfg.layer_kinds[i] = .full;
        } else if (tc.get("layer_types")) |lt| {
            for (lt.array.items, 0..) |item, i| {
                if (i >= num_layers) break;
                cfg.layer_kinds[i] = if (std.mem.eql(u8, item.string, "full_attention")) .full else .linear;
            }
        } else {
            const interval = getUsizeOr(tc, "full_attention_interval", 4);
            for (0..num_layers) |i| {
                cfg.layer_kinds[i] = if ((i + 1) % interval == 0) .full else .linear;
            }
        }

        // EOS token IDs
        if (tc.get("eos_token_id")) |ev| {
            switch (ev) {
                .integer => |id| addEos(&cfg, @intCast(id)),
                .array => |arr| {
                    for (arr.items) |item| addEos(&cfg, @intCast(item.integer));
                },
                else => {},
            }
        }
        if (is_qwen2_dense) {
            addEos(&cfg, 151645); // Qwen2.5 <|im_end|>
        } else {
            addEos(&cfg, 248046); // Qwen3.5/3.8 <|im_end|>
            addEos(&cfg, 248044);
        }

        return cfg;
    }
};

fn addEos(cfg: *ModelConfig, id: u32) void {
    if (cfg.isEos(id) or cfg.eos_count >= max_eos) return;
    cfg.eos_ids[cfg.eos_count] = id;
    cfg.eos_count += 1;
}

fn getUsize(obj: std.json.ObjectMap, key: []const u8) usize {
    return @intCast(obj.get(key).?.integer);
}

fn getUsizeOr(obj: std.json.ObjectMap, key: []const u8, fallback: usize) usize {
    const v = obj.get(key) orelse return fallback;
    return @intCast(v.integer);
}

fn asF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}
