//! Native zero-dependency Zig OpenAI HTTP/SSE & `/v1/stream-parallel` server for m4lm.
//!
//! Endpoints:
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/chat/completions  (streaming SSE + non-streaming JSON)
//!   POST /v1/completions       (streaming SSE + non-streaming JSON)
//!   POST /v1/stream-parallel   (zero-copy prefix-forked lockstep multi-stream wave, SSE + JSON)
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const spec_mod = @import("speculate.zig");
const draft_mod = @import("drafter.zig");
const cross_mod = @import("cross_draft.zig");
const samp_mod = @import("sampler.zig");
const batch_mod = @import("batch.zig");
const pcache_mod = @import("prefix_cache.zig");

const Model = model_mod.Model;
const Tokenizer = tok_mod.Tokenizer;
const nowMs = model_mod.nowMs;

// Minimal POSIX socket declarations (macOS arm64 ABI)
const SockAddrIn = extern struct {
    sin_len: u8 = @sizeOf(SockAddrIn),
    sin_family: u8 = 2, // AF_INET
    sin_port: u16,
    sin_addr: u32,
    sin_zero: [8]u8 = @splat(0),
};

extern "c" fn socket(domain: c_int, sock_type: c_int, protocol: c_int) c_int;
extern "c" fn setsockopt(fd: c_int, level: c_int, optname: c_int, optval: *const anyopaque, optlen: u32) c_int;
extern "c" fn bind(fd: c_int, addr: *const SockAddrIn, len: u32) c_int;
extern "c" fn listen(fd: c_int, backlog: c_int) c_int;
extern "c" fn accept(fd: c_int, addr: ?*anyopaque, len: ?*u32) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;

const AF_INET: c_int = 2;
const SOCK_STREAM: c_int = 1;
const SOL_SOCKET: c_int = 0xffff;
const SO_REUSEADDR: c_int = 0x0004;
const IPPROTO_TCP: c_int = 6;
const TCP_NODELAY: c_int = 1;

pub const DraftKind = enum { none, dflash2, cross_model };

pub const ServerEngine = struct {
    a: std.mem.Allocator,
    ctx: *mtl.Context,
    model: *Model,
    tok: *Tokenizer,
    model_name: []const u8,
    draft_kind: DraftKind = .none,
    dflash_dec: ?*draft_mod.HybridDFlashDecoder = null,
    cross_dec: ?*cross_mod.CrossModelDecoder = null,
    copy_dec: spec_mod.AdaptiveSpecDecoder,
    prefix_cache: ?pcache_mod.HybridPrefixCache = null,
    fork_engine: ?batch_mod.ForkBatchEngine = null,
    max_streams: usize = 16,
    max_suffix: u32 = 2048,

    pub fn init(
        a: std.mem.Allocator,
        ctx: *mtl.Context,
        model: *Model,
        tok: *Tokenizer,
        model_name: []const u8,
        dflash_dec: ?*draft_mod.HybridDFlashDecoder,
        cross_dec: ?*cross_mod.CrossModelDecoder,
        max_streams: usize,
    ) !ServerEngine {
        // Resolve `<|im_end|>` token ID for automatic system/turn boundary checkpointing
        var im_end_id: u32 = 0;
        if (tok.encode(a, "<|im_end|>")) |ids| {
            defer a.free(ids);
            if (ids.len == 1) im_end_id = ids[0];
        } else |_| {}

        const cache_slots: usize = if (model.w.total_bytes > 8 * 1024 * 1024 * 1024) 4 else 8;
        const pcache = pcache_mod.HybridPrefixCache.init(a, model, cache_slots, im_end_id) catch null;

        return .{
            .a = a,
            .ctx = ctx,
            .model = model,
            .tok = tok,
            .model_name = model_name,
            .draft_kind = if (dflash_dec != null) .dflash2 else if (cross_dec != null) .cross_model else .none,
            .dflash_dec = dflash_dec,
            .cross_dec = cross_dec,
            .copy_dec = try spec_mod.AdaptiveSpecDecoder.init(a, 127),
            .prefix_cache = pcache,
            .max_streams = @max(1, @min(max_streams, batch_mod.max_streams)),
        };
    }

    pub fn deinit(self: *ServerEngine) void {
        if (self.prefix_cache) |*pc| pc.deinit();
        if (self.fork_engine) |*fe| fe.deinit();
        self.copy_dec.deinit();
    }

    fn ensureForkEngine(self: *ServerEngine) !*batch_mod.ForkBatchEngine {
        if (self.fork_engine == null) {
            self.fork_engine = try batch_mod.ForkBatchEngine.init(self.a, self.model, self.max_streams, self.max_suffix);
        }
        return &self.fork_engine.?;
    }

    pub fn serve(self: *ServerEngine, port: u16) !void {
        const srv_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (srv_fd < 0) return error.SocketFailed;
        defer _ = close(srv_fd);

        const one: c_int = 1;
        _ = setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &one, @sizeOf(c_int));

        const addr = SockAddrIn{
            .sin_port = std.mem.nativeToBig(u16, port),
            .sin_addr = std.mem.nativeToBig(u32, 0x7F000001), // 127.0.0.1
        };
        if (bind(srv_fd, &addr, @sizeOf(SockAddrIn)) < 0) return error.BindFailed;
        if (listen(srv_fd, 64) < 0) return error.ListenFailed;

        std.debug.print("m4lm HTTP server listening on http://127.0.0.1:{d} (model={s}, draft={s}, max_streams={d})\n", .{
            port, self.model_name, @tagName(self.draft_kind), self.max_streams,
        });

        while (true) {
            const client_fd = accept(srv_fd, null, null);
            if (client_fd < 0) continue;
            _ = setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, @sizeOf(c_int));
            self.handleClient(client_fd) catch |err| {
                std.debug.print("HTTP request error: {}\n", .{err});
                sendJson(client_fd, 500, "{\"error\":\"internal server error\"}");
            };
            _ = close(client_fd);
        }
    }

    fn handleClient(self: *ServerEngine, fd: c_int) !void {
        var req_buf: std.ArrayList(u8) = .empty;
        defer req_buf.deinit(self.a);

        var chunk: [8192]u8 = undefined;
        var header_end: ?usize = null;
        var content_len: usize = 0;

        while (true) {
            const n = read(fd, &chunk, chunk.len);
            if (n <= 0) break;
            try req_buf.appendSlice(self.a, chunk[0..@intCast(n)]);
            if (header_end == null) {
                if (std.mem.indexOf(u8, req_buf.items, "\r\n\r\n")) |pos| {
                    header_end = pos + 4;
                    content_len = parseContentLength(req_buf.items[0..pos]);
                }
            }
            if (header_end) |he| {
                if (req_buf.items.len >= he + content_len) break;
            }
            if (req_buf.items.len > 16 * 1024 * 1024) return error.RequestTooLarge;
        }

        const he = header_end orelse return;
        const header_text = req_buf.items[0..he];
        const body = req_buf.items[he..@min(req_buf.items.len, he + content_len)];

        // Parse "METHOD PATH HTTP/1.1"
        const line_end = std.mem.indexOf(u8, header_text, "\r\n") orelse return;
        const first_line = header_text[0..line_end];
        var parts = std.mem.splitScalar(u8, first_line, ' ');
        const method = parts.next() orelse return;
        const raw_path = parts.next() orelse return;
        const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;

        if (std.mem.eql(u8, method, "OPTIONS")) {
            writeAll(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n");
            return;
        }

        if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(self.a);
            const txt = try std.fmt.allocPrint(self.a, "{{\"status\":\"ok\",\"engine\":\"m4lm\",\"device\":\"{s}\",\"model\":\"{s}\",\"drafter\":\"{s}\",\"max_streams\":{d}}}", .{
                self.ctx.deviceName(), self.model_name, @tagName(self.draft_kind), self.max_streams,
            });
            defer self.a.free(txt);
            sendJson(fd, 200, txt);
            return;
        }

        if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
            const txt = try std.fmt.allocPrint(self.a, "{{\"object\":\"list\",\"data\":[{{\"id\":\"{s}\",\"object\":\"model\",\"owned_by\":\"m4lm\"}}]}}", .{self.model_name});
            defer self.a.free(txt);
            sendJson(fd, 200, txt);
            return;
        }

        if (std.mem.eql(u8, method, "POST") and (std.mem.eql(u8, path, "/v1/chat/completions") or std.mem.eql(u8, path, "/v1/completions"))) {
            const is_chat = std.mem.eql(u8, path, "/v1/chat/completions");
            try self.handleCompletion(fd, body, is_chat);
            return;
        }

        if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/stream-parallel")) {
            try self.handleStreamParallel(fd, body);
            return;
        }

        sendJson(fd, 404, "{\"error\":\"not found\"}");
    }

    fn handleCompletion(self: *ServerEngine, fd: c_int, body: []const u8, is_chat: bool) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.a, body, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;

        const max_tokens: usize = getJsonUsize(obj, "max_tokens", getJsonUsize(obj, "max_completion_tokens", 256));
        const stream = getJsonBool(obj, "stream", false);
        var enable_thinking = false;
        if (obj.get("chat_template_kwargs")) |ctk| {
            if (ctk == .object) enable_thinking = getJsonBool(ctk.object, "enable_thinking", false);
        }
        const sampling = samp_mod.SamplingParams{
            .temperature = getJsonF32(obj, "temperature", 0.0),
            .top_p = getJsonF32(obj, "top_p", 1.0),
            .top_k = @intCast(getJsonUsize(obj, "top_k", 0)),
            .min_p = getJsonF32(obj, "min_p", 0.0),
            .seed = @intCast(getJsonUsize(obj, "seed", 42)),
        };

        const rendered = if (is_chat) blk: {
            const msgs_val = obj.get("messages") orelse return sendJson(fd, 400, "{\"error\":\"missing messages\"}");
            var msgs = try self.a.alloc(Tokenizer.ChatMessage, msgs_val.array.items.len);
            defer self.a.free(msgs);
            for (msgs_val.array.items, 0..) |item, i| {
                const mo = item.object;
                msgs[i] = .{
                    .role = mo.get("role").?.string,
                    .content = if (mo.get("content")) |cv| (if (cv == .string) cv.string else "") else "",
                };
            }
            break :blk try Tokenizer.formatChatMessages(self.a, msgs, true, enable_thinking);
        } else blk: {
            const p = obj.get("prompt") orelse return sendJson(fd, 400, "{\"error\":\"missing prompt\"}");
            break :blk try self.a.dupe(u8, p.string);
        };
        defer self.a.free(rendered);

        const prompt_ids = try self.tok.encode(self.a, rendered);
        defer self.a.free(prompt_ids);

        const SseCtx = struct {
            a: std.mem.Allocator,
            fd: c_int,
            tok: *Tokenizer,
            model_name: []const u8,
            is_chat: bool,
            fn emit(raw: ?*anyopaque, tok_id: u32) void {
                const s: *@This() = @ptrCast(@alignCast(raw.?));
                const piece = s.tok.decodeToken(tok_id);
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(s.a);
                if (s.is_chat) {
                    buf.appendSlice(s.a, "data: {\"id\":\"m4lm\",\"object\":\"chat.completion.chunk\",\"model\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, s.model_name) catch return;
                    buf.appendSlice(s.a, "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, piece) catch return;
                    buf.appendSlice(s.a, "\"},\"finish_reason\":null}]}\n\n") catch return;
                } else {
                    buf.appendSlice(s.a, "data: {\"id\":\"m4lm\",\"object\":\"text_completion\",\"model\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, s.model_name) catch return;
                    buf.appendSlice(s.a, "\",\"choices\":[{\"index\":0,\"text\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, piece) catch return;
                    buf.appendSlice(s.a, "\",\"finish_reason\":null}]}\n\n") catch return;
                }
                writeAll(s.fd, buf.items);
            }
        };

        var sse_ctx = SseCtx{ .a = self.a, .fd = fd, .tok = self.tok, .model_name = self.model_name, .is_chat = is_chat };
        if (stream) {
            writeAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        }

        var out_ids: std.ArrayList(u32) = .empty;
        defer out_ids.deinit(self.a);
        const cb: ?*const fn (?*anyopaque, u32) void = if (stream) SseCtx.emit else null;

        var stats = spec_mod.SpecStats{};
        var cached_tokens: usize = 0;
        if (sampling.temperature <= 1e-6) {
            if (self.dflash_dec) |dd| {
                if (self.prefix_cache) |*pc| pc.invalidate();
                stats = try dd.generate(prompt_ids, max_tokens, true, &out_ids, self.a, cb, &sse_ctx);
            } else if (self.cross_dec) |cd| {
                if (self.prefix_cache) |*pc| pc.invalidate();
                stats = try cd.generate(prompt_ids, max_tokens, true, &out_ids, self.a, cb, &sse_ctx);
            } else if (self.prefix_cache) |*pc| {
                const pr = try pc.prefillCached(prompt_ids, 128, false);
                cached_tokens = pr.cached_tokens;
                stats = try self.copy_dec.generateFromPrefilled(self.model, prompt_ids, pr.next_tok, max_tokens, true, &out_ids, self.a, cb, &sse_ctx);
                try pc.commitGenerated(out_ids.items, 0);
            } else {
                self.model.reset();
                stats = try self.copy_dec.generate(self.model, prompt_ids, max_tokens, true, &out_ids, self.a, cb, &sse_ctx);
            }
        } else if (self.cross_dec) |cd| {
            // Exact Leviathan/Chen sampled speculative decoding at temp > 0!
            if (self.prefix_cache) |*pc| pc.invalidate();
            stats = try cd.generateSampled(prompt_ids, max_tokens, true, sampling, &out_ids, self.a, cb, &sse_ctx);
        } else {
            // Non-greedy sampled generation (with automatic Hybrid KV+GDN prefix cache!)
            var rng = samp_mod.Rng.init(sampling.seed);
            if (self.prefix_cache) |*pc| {
                const pr = try pc.prefillCached(prompt_ids, 128, true);
                cached_tokens = pr.cached_tokens;
            } else {
                self.model.reset();
                _ = try self.model.prefill(prompt_ids, 128);
            }
            var next_tok = try self.model.sampleRow(0, sampling, &rng);
            const td0 = nowMs();
            while (stats.tokens_generated < max_tokens) {
                if (self.model.c.isEos(next_tok)) break;
                try out_ids.append(self.a, next_tok);
                if (cb) |f| f(&sse_ctx, next_tok);
                stats.tokens_generated += 1;
                stats.rounds += 1;
                if (stats.tokens_generated >= max_tokens) break;
                _ = try self.model.stepWithLogits(next_tok);
                next_tok = try self.model.sampleRow(0, sampling, &rng);
            }
            stats.decode_ms = nowMs() - td0;
            if (self.prefix_cache) |*pc| try pc.commitGenerated(out_ids.items, 0);
        }

        if (stream) {
            var end_buf: std.ArrayList(u8) = .empty;
            defer end_buf.deinit(self.a);
            const done_chunk = try std.fmt.allocPrint(self.a, "data: {{\"id\":\"m4lm\",\"object\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"tok_per_sec\":{d:.1}}}}}\n\ndata: [DONE]\n\n", .{
                if (is_chat) "chat.completion.chunk" else "text_completion",
                prompt_ids.len,
                stats.tokens_generated,
                prompt_ids.len + stats.tokens_generated,
                cached_tokens,
                stats.tokPerSec(),
            });
            defer self.a.free(done_chunk);
            writeAll(fd, done_chunk);
            return;
        }

        // Assemble full response text
        var text_buf: std.ArrayList(u8) = .empty;
        defer text_buf.deinit(self.a);
        for (out_ids.items) |tid| try text_buf.appendSlice(self.a, self.tok.decodeToken(tid));

        var json_buf: std.ArrayList(u8) = .empty;
        defer json_buf.deinit(self.a);
        if (is_chat) {
            try json_buf.appendSlice(self.a, "{\"id\":\"m4lm\",\"object\":\"chat.completion\",\"model\":\"");
            try appendJsonEscaped(self.a, &json_buf, self.model_name);
            try json_buf.appendSlice(self.a, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"");
            try appendJsonEscaped(self.a, &json_buf, text_buf.items);
            try json_buf.appendSlice(self.a, "\"},\"finish_reason\":\"stop\"}]");
        } else {
            try json_buf.appendSlice(self.a, "{\"id\":\"m4lm\",\"object\":\"text_completion\",\"model\":\"");
            try appendJsonEscaped(self.a, &json_buf, self.model_name);
            try json_buf.appendSlice(self.a, "\",\"choices\":[{\"index\":0,\"text\":\"");
            try appendJsonEscaped(self.a, &json_buf, text_buf.items);
            try json_buf.appendSlice(self.a, "\",\"finish_reason\":\"stop\"}]");
        }
        const usage = try std.fmt.allocPrint(self.a, ",\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"tok_per_sec\":{d:.1},\"tok_per_round\":{d:.2}}}}}", .{
            prompt_ids.len, stats.tokens_generated, prompt_ids.len + stats.tokens_generated, cached_tokens, stats.tokPerSec(), stats.avgTokensPerRound(),
        });
        defer self.a.free(usage);
        try json_buf.appendSlice(self.a, usage);
        sendJson(fd, 200, json_buf.items);
    }

    fn handleStreamParallel(self: *ServerEngine, fd: c_int, body: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.a, body, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;

        const stream_sse = getJsonBool(obj, "stream", false);
        const enable_thinking = getJsonBool(obj, "enable_thinking", false);
        const default_max = getJsonUsize(obj, "max_tokens", 256);
        const default_temp = getJsonF32(obj, "temperature", 0.0);

        const streams_val = obj.get("streams") orelse return sendJson(fd, 400, "{\"error\":\"missing streams array\"}");
        const n = streams_val.array.items.len;
        if (n == 0 or n > self.max_streams) {
            return sendJson(fd, 400, "{\"error\":\"streams count must be between 1 and max_streams\"}");
        }

        // Format shared prefix
        const shared_text = if (obj.get("shared_messages")) |sm| blk: {
            var msgs = try self.a.alloc(Tokenizer.ChatMessage, sm.array.items.len);
            defer self.a.free(msgs);
            for (sm.array.items, 0..) |item, i| {
                const mo = item.object;
                msgs[i] = .{
                    .role = mo.get("role").?.string,
                    .content = if (mo.get("content")) |cv| (if (cv == .string) cv.string else "") else "",
                };
            }
            break :blk try Tokenizer.formatChatMessages(self.a, msgs, false, enable_thinking);
        } else if (obj.get("shared_prompt")) |sp|
            try self.a.dupe(u8, sp.string)
        else
            try self.a.dupe(u8, "");
        defer self.a.free(shared_text);

        const shared_ids = try self.tok.encode(self.a, shared_text);
        defer self.a.free(shared_ids);

        // Format and tokenize each stream's delta
        var specs = try self.a.alloc(batch_mod.StreamSpec, n);
        defer self.a.free(specs);
        var stream_labels = try self.a.alloc([]const u8, n);
        defer self.a.free(stream_labels);
        var delta_allocs = try self.a.alloc([]u32, n);
        defer {
            for (delta_allocs) |d| self.a.free(d);
            self.a.free(delta_allocs);
        }

        for (streams_val.array.items, 0..) |item, i| {
            const so = item.object;
            stream_labels[i] = if (so.get("id")) |idv| idv.string else "stream";
            const d_text = if (so.get("delta_messages")) |dm| blk: {
                var msgs = try self.a.alloc(Tokenizer.ChatMessage, dm.array.items.len);
                defer self.a.free(msgs);
                for (dm.array.items, 0..) |mi, k| {
                    const mo = mi.object;
                    msgs[k] = .{
                        .role = mo.get("role").?.string,
                        .content = if (mo.get("content")) |cv| (if (cv == .string) cv.string else "") else "",
                    };
                }
                break :blk try Tokenizer.formatChatMessages(self.a, msgs, true, enable_thinking);
            } else if (so.get("delta_prompt")) |dp|
                try self.a.dupe(u8, dp.string)
            else
                try self.a.dupe(u8, "");
            defer self.a.free(d_text);

            delta_allocs[i] = try self.tok.encode(self.a, d_text);
            specs[i] = .{
                .delta_tokens = delta_allocs[i],
                .max_tokens = getJsonUsize(so, "max_tokens", default_max),
                .sampling = .{
                    .temperature = getJsonF32(so, "temperature", default_temp),
                    .top_p = getJsonF32(so, "top_p", 1.0),
                    .top_k = @intCast(getJsonUsize(so, "top_k", 0)),
                    .min_p = getJsonF32(so, "min_p", 0.0),
                    .seed = @intCast(getJsonUsize(so, "seed", 42 + i)),
                },
            };
        }

        if (self.prefix_cache) |*pc| pc.invalidate();
        const fe = try self.ensureForkEngine();

        const WaveSseCtx = struct {
            a: std.mem.Allocator,
            fd: c_int,
            tok: *Tokenizer,
            labels: []const []const u8,
            fn emit(raw: ?*anyopaque, sidx: usize, tok_id: u32) void {
                const s: *@This() = @ptrCast(@alignCast(raw.?));
                const piece = s.tok.decodeToken(tok_id);
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(s.a);
                const hdr = std.fmt.allocPrint(s.a, "data: {{\"stream_index\":{d},\"id\":\"", .{sidx}) catch return;
                defer s.a.free(hdr);
                buf.appendSlice(s.a, hdr) catch return;
                appendJsonEscaped(s.a, &buf, s.labels[sidx]) catch return;
                buf.appendSlice(s.a, "\",\"delta\":\"") catch return;
                appendJsonEscaped(s.a, &buf, piece) catch return;
                buf.appendSlice(s.a, "\"}\n\n") catch return;
                writeAll(s.fd, buf.items);
            }
        };

        var ws_ctx = WaveSseCtx{ .a = self.a, .fd = fd, .tok = self.tok, .labels = stream_labels };
        if (stream_sse) {
            writeAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        }

        const out_lists = try self.a.alloc(std.ArrayList(u32), n);
        for (out_lists) |*l| l.* = .empty;
        defer {
            for (out_lists) |*l| l.deinit(self.a);
            self.a.free(out_lists);
        }

        const cb: ?batch_mod.StreamTokenCallback = if (stream_sse) WaveSseCtx.emit else null;
        const stats = try fe.runWave(shared_ids, specs, out_lists, self.a, cb, &ws_ctx);

        if (stream_sse) {
            const summary = try std.fmt.allocPrint(self.a, "data: {{\"done\":true,\"num_streams\":{d},\"prefix_tokens\":{d},\"cached_tokens\":{d},\"total_generated\":{d},\"decode_tok_per_sec\":{d:.1},\"e2e_tok_per_sec\":{d:.1}}}\n\ndata: [DONE]\n\n", .{
                stats.num_streams, stats.prefix_tokens, stats.cached_tokens, stats.total_generated, stats.decodeTokPerSec(), stats.e2eTokPerSec(),
            });
            defer self.a.free(summary);
            writeAll(fd, summary);
            return;
        }

        var json_buf: std.ArrayList(u8) = .empty;
        defer json_buf.deinit(self.a);
        try json_buf.appendSlice(self.a, "{\"object\":\"stream_parallel.wave\",\"streams\":[");
        for (0..n) |i| {
            if (i > 0) try json_buf.append(self.a, ',');
            var tb: std.ArrayList(u8) = .empty;
            defer tb.deinit(self.a);
            for (out_lists[i].items) |tid| try tb.appendSlice(self.a, self.tok.decodeToken(tid));

            const pre = try std.fmt.allocPrint(self.a, "{{\"index\":{d},\"id\":\"", .{i});
            defer self.a.free(pre);
            try json_buf.appendSlice(self.a, pre);
            try appendJsonEscaped(self.a, &json_buf, stream_labels[i]);
            try json_buf.appendSlice(self.a, "\",\"text\":\"");
            try appendJsonEscaped(self.a, &json_buf, tb.items);
            const post = try std.fmt.allocPrint(self.a, "\",\"completion_tokens\":{d}}}", .{stats.per_stream_tokens[i]});
            defer self.a.free(post);
            try json_buf.appendSlice(self.a, post);
        }
        const foot = try std.fmt.allocPrint(self.a, "],\"usage\":{{\"num_streams\":{d},\"prefix_tokens\":{d},\"cached_tokens\":{d},\"total_generated\":{d},\"prefill_ms\":{d:.2},\"fork_ms\":{d:.2},\"delta_prefill_ms\":{d:.2},\"decode_ms\":{d:.2},\"decode_tok_per_sec\":{d:.1},\"e2e_tok_per_sec\":{d:.1}}}}}", .{
            stats.num_streams,
            stats.prefix_tokens,
            stats.cached_tokens,
            stats.total_generated,
            stats.prefill_ms,
            stats.fork_ms,
            stats.delta_prefill_ms,
            stats.decode_ms,
            stats.decodeTokPerSec(),
            stats.e2eTokPerSec(),
        });
        defer self.a.free(foot);
        try json_buf.appendSlice(self.a, foot);
        sendJson(fd, 200, json_buf.items);
    }
};

fn parseContentLength(headers: []const u8) usize {
    var it = std.mem.splitSequence(u8, headers, "\r\n");
    while (it.next()) |line| {
        if (line.len >= 15 and std.ascii.eqlIgnoreCase(line[0..15], "content-length:")) {
            const val = std.mem.trim(u8, line[15..], " \t");
            return std.fmt.parseInt(usize, val, 10) catch 0;
        }
    }
    return 0;
}

fn writeAll(fd: c_int, data: []const u8) void {
    var off: usize = 0;
    while (off < data.len) {
        const n = write(fd, data.ptr + off, data.len - off);
        if (n <= 0) break;
        off += @intCast(n);
    }
}

fn sendJson(fd: c_int, status: u16, body: []const u8) void {
    var hdr_buf: [256]u8 = undefined;
    const status_txt = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        else => "Internal Server Error",
    };
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{
        status, status_txt, body.len,
    }) catch return;
    writeAll(fd, hdr);
    writeAll(fd, body);
}

fn appendJsonEscaped(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |ch| {
        switch (ch) {
            '"' => try out.appendSlice(a, "\\\""),
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const e = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{ch}) catch unreachable;
                try out.appendSlice(a, e);
            },
            else => try out.append(a, ch),
        }
    }
}

fn getJsonUsize(obj: std.json.ObjectMap, key: []const u8, fallback: usize) usize {
    const v = obj.get(key) orelse return fallback;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else fallback,
        else => fallback,
    };
}

fn getJsonF32(obj: std.json.ObjectMap, key: []const u8, fallback: f32) f32 {
    const v = obj.get(key) orelse return fallback;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => fallback,
    };
}

fn getJsonBool(obj: std.json.ObjectMap, key: []const u8, fallback: bool) bool {
    const v = obj.get(key) orelse return fallback;
    return switch (v) {
        .bool => |b| b,
        else => fallback,
    };
}
