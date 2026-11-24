//! C ABI exported by `libm4lm.dylib` so Python (or Swift/Rust/C) calls directly into
//! our clean-room Zig + Metal engine with zero IPC and zero Apple MLX dependency.
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const spec_mod = @import("speculate.zig");
const draft_mod = @import("drafter.zig");
const cross_mod = @import("cross_draft.zig");
const samp_mod = @import("sampler.zig");
const batch_mod = @import("batch.zig");
const srv_mod = @import("server.zig");

const Model = model_mod.Model;
const Tokenizer = tok_mod.Tokenizer;
const nowMs = model_mod.nowMs;

pub const M4GenOptions = extern struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: f32 = 0.0,
    seed: u64 = 42,
    enable_thinking: u8 = 0,
    use_chat_template: u8 = 0,
    stop_on_eos: u8 = 1,
    _pad: u8 = 0,
};

pub const M4Stats = extern struct {
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    rounds: u32 = 0,
    drafts_accepted: u32 = 0,
    drafts_proposed: u32 = 0,
    cached_tokens: u32 = 0,
    decode_ms: f64 = 0,
    total_ms: f64 = 0,
    tok_per_sec: f64 = 0,
};

pub const M4TokenCb = ?*const fn (user_data: ?*anyopaque, token_id: u32, utf8: [*]const u8, utf8_len: usize) callconv(.c) u8;
pub const M4WaveTokenCb = ?*const fn (user_data: ?*anyopaque, stream_idx: u32, token_id: u32, utf8: [*]const u8, utf8_len: usize) callconv(.c) u8;

pub const M4Engine = struct {
    a: std.mem.Allocator,
    ctx: mtl.Context,
    tok: Tokenizer,
    model: Model,
    dflash: ?draft_mod.Drafter = null,
    dflash_dec: ?draft_mod.HybridDFlashDecoder = null,
    cross_model: ?Model = null,
    cross_dec: ?cross_mod.CrossModelDecoder = null,
    copy_dec: spec_mod.AdaptiveSpecDecoder,
    fork_engine: ?batch_mod.ForkBatchEngine = null,
    max_streams: usize,
    device_cstr: [128]u8 = @splat(0),
};

fn cSliceOpt(ptr: ?[*:0]const u8) ?[]const u8 {
    const p = ptr orelse return null;
    const s = std.mem.span(p);
    return if (s.len == 0) null else s;
}

pub export fn m4lm_load(
    model_dir_c: [*:0]const u8,
    dflash_dir_c: ?[*:0]const u8,
    cross_draft_dir_c: ?[*:0]const u8,
    max_context: u32,
    max_streams: u32,
) callconv(.c) ?*M4Engine {
    const a = std.heap.smp_allocator;
    const model_dir = std.mem.span(model_dir_c);
    const dflash_dir = cSliceOpt(dflash_dir_c);
    const cross_dir = cSliceOpt(cross_draft_dir_c);

    const eng = a.create(M4Engine) catch return null;
    errdefer a.destroy(eng);

    const ctx = mtl.Context.init(1024) catch return null;
    eng.* = .{
        .a = a,
        .ctx = ctx,
        .tok = undefined,
        .model = undefined,
        .copy_dec = undefined,
        .max_streams = @max(1, @min(@as(usize, if (max_streams == 0) 16 else max_streams), batch_mod.max_streams)),
    };
    errdefer eng.ctx.deinit();

    const dname = eng.ctx.deviceName();
    const dn = @min(dname.len, eng.device_cstr.len - 1);
    @memcpy(eng.device_cstr[0..dn], dname[0..dn]);

    eng.tok = Tokenizer.load(a, model_dir) catch return null;
    errdefer eng.tok.deinit();

    const ctx_len = if (max_context == 0) 8192 else max_context;
    eng.model = Model.init(a, &eng.ctx, model_dir, .{
        .max_context = ctx_len,
        .capture_taps = (dflash_dir != null),
    }) catch return null;
    errdefer eng.model.deinit();

    eng.copy_dec = spec_mod.AdaptiveSpecDecoder.init(a, 127) catch return null;
    errdefer eng.copy_dec.deinit();

    if (dflash_dir) |dd| {
        eng.dflash = draft_mod.Drafter.init(a, &eng.ctx, &eng.model, dd) catch return null;
        eng.dflash_dec = draft_mod.HybridDFlashDecoder.init(a, &eng.dflash.?, 15) catch return null;
    } else if (cross_dir) |cd| {
        eng.cross_model = Model.init(a, &eng.ctx, cd, .{ .max_context = ctx_len }) catch return null;
        eng.cross_dec = cross_mod.CrossModelDecoder.init(a, &eng.model, &eng.cross_model.?, 5) catch return null;
    }

    return eng;
}

pub export fn m4lm_free(eng: ?*M4Engine) callconv(.c) void {
    const e = eng orelse return;
    if (e.fork_engine) |*fe| fe.deinit();
    if (e.dflash_dec) |*dd| dd.deinit();
    if (e.dflash) |*d| d.deinit();
    if (e.cross_dec) |*cd| cd.deinit();
    if (e.cross_model) |*cm| cm.deinit();
    e.copy_dec.deinit();
    e.model.deinit();
    e.tok.deinit();
    e.ctx.deinit();
    e.a.destroy(e);
}

pub export fn m4lm_device_name(eng: ?*M4Engine) callconv(.c) [*:0]const u8 {
    const e = eng orelse return "Apple GPU";
    return @ptrCast(&e.device_cstr);
}

pub export fn m4lm_generate(
    eng: ?*M4Engine,
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    opts_ptr: ?*const M4GenOptions,
    cb: M4TokenCb,
    user_data: ?*anyopaque,
    out_stats: ?*M4Stats,
) callconv(.c) i32 {
    const e = eng orelse return -1;
    const opts = if (opts_ptr) |p| p.* else M4GenOptions{};
    const raw_prompt = prompt_ptr[0..prompt_len];
    const t_all0 = nowMs();

    const prompt_text = if (opts.use_chat_template != 0)
        (Tokenizer.formatChatPrompt(e.a, raw_prompt, opts.enable_thinking != 0) catch return -2)
    else
        (e.a.dupe(u8, raw_prompt) catch return -2);
    defer e.a.free(prompt_text);

    const prompt_ids = e.tok.encode(e.a, prompt_text) catch return -3;
    defer e.a.free(prompt_ids);

    const Bridge = struct {
        tok: *Tokenizer,
        cb: M4TokenCb,
        ud: ?*anyopaque,
        fn onTok(raw: ?*anyopaque, tok_id: u32) void {
            const b: *@This() = @ptrCast(@alignCast(raw.?));
            if (b.cb) |f| {
                const piece = b.tok.decodeToken(tok_id);
                _ = f(b.ud, tok_id, piece.ptr, piece.len);
            }
        }
    };
    var bridge = Bridge{ .tok = &e.tok, .cb = cb, .ud = user_data };
    const zig_cb: ?*const fn (?*anyopaque, u32) void = if (cb != null) Bridge.onTok else null;

    var stats = spec_mod.SpecStats{};
    const stop_eos = (opts.stop_on_eos != 0);

    if (opts.temperature <= 1e-6) {
        if (e.dflash_dec) |*dd| {
            stats = dd.generate(prompt_ids, opts.max_tokens, stop_eos, null, e.a, zig_cb, &bridge) catch return -4;
        } else if (e.cross_dec) |*cd| {
            stats = cd.generate(prompt_ids, opts.max_tokens, stop_eos, null, e.a, zig_cb, &bridge) catch return -4;
        } else {
            e.model.reset();
            stats = e.copy_dec.generate(&e.model, prompt_ids, opts.max_tokens, stop_eos, null, e.a, zig_cb, &bridge) catch return -4;
        }
    } else {
        const sp = samp_mod.SamplingParams{
            .temperature = opts.temperature,
            .top_p = opts.top_p,
            .top_k = opts.top_k,
            .min_p = opts.min_p,
            .seed = opts.seed,
        };
        if (e.cross_dec) |*cd| {
            stats = cd.generateSampled(prompt_ids, opts.max_tokens, stop_eos, sp, null, e.a, zig_cb, &bridge) catch return -4;
        } else {
            var rng = samp_mod.Rng.init(opts.seed);
            e.model.reset();
            _ = e.model.prefill(prompt_ids, 128) catch return -4;
            var tok_id = e.model.sampleRow(0, sp, &rng) catch return -4;
            const td0 = nowMs();
            while (stats.tokens_generated < opts.max_tokens) {
                if (stop_eos and e.model.c.isEos(tok_id)) break;
                if (cb) |f| {
                    const piece = e.tok.decodeToken(tok_id);
                    if (f(user_data, tok_id, piece.ptr, piece.len) == 0) break;
                }
                stats.tokens_generated += 1;
                stats.rounds += 1;
                if (stats.tokens_generated >= opts.max_tokens) break;
                _ = e.model.stepWithLogits(tok_id) catch return -4;
                tok_id = e.model.sampleRow(0, sp, &rng) catch return -4;
            }
            stats.decode_ms = nowMs() - td0;
        }
    }

    if (out_stats) |os| {
        os.* = .{
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = @intCast(stats.tokens_generated),
            .rounds = @intCast(stats.rounds),
            .drafts_accepted = @intCast(stats.drafts_accepted),
            .drafts_proposed = @intCast(stats.drafts_proposed),
            .cached_tokens = 0,
            .decode_ms = stats.decode_ms,
            .total_ms = nowMs() - t_all0,
            .tok_per_sec = stats.tokPerSec(),
        };
    }
    return 0;
}

pub export fn m4lm_stream_parallel(
    eng: ?*M4Engine,
    shared_ptr: [*]const u8,
    shared_len: usize,
    n_streams: u32,
    deltas_ptrs: [*]const [*]const u8,
    delta_lens: [*]const usize,
    max_tokens_arr: [*]const u32,
    temps_arr: [*]const f32,
    cb: M4WaveTokenCb,
    user_data: ?*anyopaque,
    out_stats: ?*M4Stats,
) callconv(.c) i32 {
    const e = eng orelse return -1;
    if (n_streams == 0 or n_streams > e.max_streams) return -2;
    const t_all0 = nowMs();

    if (e.fork_engine == null) {
        e.fork_engine = batch_mod.ForkBatchEngine.init(e.a, &e.model, e.max_streams, 2048) catch return -3;
    }

    const shared_ids = e.tok.encode(e.a, shared_ptr[0..shared_len]) catch return -4;
    defer e.a.free(shared_ids);

    const n: usize = n_streams;
    const specs = e.a.alloc(batch_mod.StreamSpec, n) catch return -5;
    defer e.a.free(specs);
    const delta_bufs = e.a.alloc([]u32, n) catch return -5;
    for (delta_bufs) |*d| d.* = &.{};
    defer {
        for (delta_bufs) |d| if (d.len > 0) e.a.free(d);
        e.a.free(delta_bufs);
    }

    for (0..n) |i| {
        const d_txt = deltas_ptrs[i][0..delta_lens[i]];
        delta_bufs[i] = e.tok.encode(e.a, d_txt) catch return -4;
        specs[i] = .{
            .delta_tokens = delta_bufs[i],
            .max_tokens = max_tokens_arr[i],
            .sampling = .{ .temperature = temps_arr[i], .seed = 42 + i },
        };
    }

    const WaveBridge = struct {
        tok: *Tokenizer,
        cb: M4WaveTokenCb,
        ud: ?*anyopaque,
        fn onTok(raw: ?*anyopaque, sidx: usize, tok_id: u32) void {
            const b: *@This() = @ptrCast(@alignCast(raw.?));
            if (b.cb) |f| {
                const piece = b.tok.decodeToken(tok_id);
                _ = f(b.ud, @intCast(sidx), tok_id, piece.ptr, piece.len);
            }
        }
    };
    var wb = WaveBridge{ .tok = &e.tok, .cb = cb, .ud = user_data };
    const zig_cb: ?batch_mod.StreamTokenCallback = if (cb != null) WaveBridge.onTok else null;

    const ws = e.fork_engine.?.runWave(shared_ids, specs, null, e.a, zig_cb, &wb) catch return -6;

    if (out_stats) |os| {
        os.* = .{
            .prompt_tokens = @intCast(ws.prefix_tokens),
            .completion_tokens = @intCast(ws.total_generated),
            .rounds = @intCast(ws.decode_steps),
            .drafts_accepted = 0,
            .drafts_proposed = 0,
            .cached_tokens = @intCast(ws.cached_tokens),
            .decode_ms = ws.decode_ms,
            .total_ms = nowMs() - t_all0,
            .tok_per_sec = ws.decodeTokPerSec(),
        };
    }
    return 0;
}

pub export fn m4lm_serve(eng: ?*M4Engine, port: u16, model_name_c: ?[*:0]const u8) callconv(.c) i32 {
    const e = eng orelse return -1;
    const mname = cSliceOpt(model_name_c) orelse "m4lm-model";
    const dd_ptr: ?*draft_mod.HybridDFlashDecoder = if (e.dflash_dec) |*dd| dd else null;
    const cd_ptr: ?*cross_mod.CrossModelDecoder = if (e.cross_dec) |*cd| cd else null;

    var srv = srv_mod.ServerEngine.init(e.a, &e.ctx, &e.model, &e.tok, mname, dd_ptr, cd_ptr, e.max_streams) catch return -2;
    defer srv.deinit();
    srv.serve(port) catch return -3;
    return 0;
}
