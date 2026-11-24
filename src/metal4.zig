//! Clean-room Metal 4 runtime backend for Apple Silicon (M5 Max+).
//!
//! Owns the MTLDevice, MTL4CommandQueue, MTL4CommandAllocator, recycled MTL4CommandBuffers,
//! MTL4ArgumentTable, constant-data bump ring, MTLResidencySet, and optional MTL4CounterHeap.
const std = @import("std");
const objc = @import("objc.zig");

const Id = objc.Id;

extern "c" fn MTLCreateSystemDefaultDevice() ?Id;

pub const Error = error{
    NoMetalDevice,
    NoMetal4Support,
    BufferAllocFailed,
    ResidencyFailed,
    ShaderCompileFailed,
    FunctionNotFound,
    PipelineFailed,
    GpuTimeout,
};

pub const Size = extern struct {
    width: usize = 1,
    height: usize = 1,
    depth: usize = 1,
};

pub const Range = extern struct {
    location: usize,
    length: usize,
};

// Metal resource & stage constants
const resource_storage_shared: usize = 0;
const resource_hazard_untracked: usize = 1 << 8;
const stage_dispatch: usize = 1 << 27; // MTLStageDispatch
const visibility_device: usize = 1; // MTL4VisibilityOptionDevice

pub const MathPrecision = enum(isize) {
    fast = 0,
    precise = 1,
};

pub const Buffer = struct {
    id: Id,
    gpu_addr: u64,
    ptr: [*]u8,
    len: usize,

    pub fn slice(self: Buffer, comptime T: type, count: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(self.ptr)))[0..count];
    }

    pub fn view(self: Buffer, offset: usize) BufferRef {
        return .{ .buf_id = self.id, .offset = offset, .gpu_addr = self.gpu_addr + offset, .ptr = self.ptr + offset };
    }

    pub fn deinit(self: Buffer) void {
        objc.release(self.id);
    }
};

pub const BufferRef = struct {
    buf_id: Id,
    offset: usize,
    gpu_addr: u64,
    ptr: [*]u8,

    pub fn at(self: BufferRef, byte_offset: usize) BufferRef {
        return .{ .buf_id = self.buf_id, .offset = self.offset + byte_offset, .gpu_addr = self.gpu_addr + byte_offset, .ptr = self.ptr + byte_offset };
    }
};

pub const Library = struct {
    id: Id,

    pub fn deinit(self: Library) void {
        objc.release(self.id);
    }
};

pub const Pipeline = struct {
    id: Id,
    name: [32]u8 = @splat(0),

    pub fn deinit(self: Pipeline) void {
        objc.release(self.id);
    }
};

pub const Fence = struct {
    id: Id,

    pub fn deinit(self: Fence) void {
        objc.release(self.id);
    }
};

pub const Timeline = struct {
    heap: Id,
    device: Id,
    names: [][32]u8,
    count: usize = 0,

    pub fn init(a: std.mem.Allocator, ctx: *const Context, capacity: usize) !Timeline {
        const desc = objc.allocInit("MTL4CounterHeapDescriptor");
        defer objc.release(desc);
        objc.send(void, desc, "setType:", .{@as(isize, 1)}); // timestamp
        objc.send(void, desc, "setCount:", .{capacity});
        var err: ?Id = null;
        const heap = objc.send(?Id, ctx.device, "newCounterHeapWithDescriptor:error:", .{ desc, &err }) orelse return error.NoMetal4Support;
        return .{
            .heap = heap,
            .device = ctx.device,
            .names = try a.alloc([32]u8, capacity),
        };
    }

    pub fn reset(self: *Timeline) void {
        self.count = 0;
    }

    pub fn readTicks(self: *const Timeline, out: []u64) void {
        const range = Range{ .location = 0, .length = self.count };
        const data = objc.send(Id, self.heap, "resolveCounterRange:", .{range});
        const words: [*]align(1) const u64 = @ptrCast(objc.send([*]const u8, data, "bytes", .{}));
        for (out[0..self.count], 0..) |*dst, i| dst.* = words[i];
    }

    pub fn frequencyHz(self: *const Timeline) f64 {
        return @floatFromInt(objc.send(u64, self.device, "queryTimestampFrequency", .{}));
    }

    pub fn deinit(self: *Timeline, a: std.mem.Allocator) void {
        a.free(self.names);
        objc.release(self.heap);
    }
};

pub const Stats = struct {
    command_buffers: u32 = 0,
    dispatches: u32 = 0,
    barriers: u32 = 0,
    fences: u32 = 0,
};

pub const BackendApi = enum {
    metal4_device,
    metal4_none,
    metal3_serial,
    metal3_concurrent,
};

pub const Context = struct {
    pub const max_cmd_buffers = 32;
    pub const constants_ring_bytes = 2 * 1024 * 1024; // 2 MiB constant bump ring

    api: BackendApi = .metal3_serial,
    device: Id,
    queue: Id,
    m3_queue: Id,
    allocator: Id,
    cmd_buffers: [max_cmd_buffers]Id,
    cmd_used: usize = 0,
    last_m3_cb: ?Id = null,
    arg_table: Id,
    shared_event: Id,
    residency: Id,
    constants_buf: Buffer,
    constants_bump: usize = 0,
    tick: u64 = 0,
    waited_tick: u64 = 0,
    timeline: ?*Timeline = null,
    stats: Stats = .{},

    pub fn init(residency_capacity: usize) Error!Context {
        const pool = objc.Pool.push();
        defer pool.pop();

        const device = MTLCreateSystemDefaultDevice() orelse return error.NoMetalDevice;
        const queue = objc.send(?Id, device, "newMTL4CommandQueue", .{}) orelse return error.NoMetal4Support;
        const m3_queue = objc.send(?Id, device, "newCommandQueue", .{}) orelse return error.NoMetalDevice;
        const cmd_alloc = objc.send(?Id, device, "newCommandAllocator", .{}) orelse return error.NoMetal4Support;
        const shared_event = objc.send(?Id, device, "newSharedEvent", .{}) orelse return error.NoMetal4Support;

        const tbl_desc = objc.allocInit("MTL4ArgumentTableDescriptor");
        defer objc.release(tbl_desc);
        objc.send(void, tbl_desc, "setMaxBufferBindCount:", .{@as(usize, 31)});
        var err: ?Id = null;
        const arg_table = objc.send(?Id, device, "newArgumentTableWithDescriptor:error:", .{ tbl_desc, &err }) orelse return error.NoMetal4Support;

        const res_desc = objc.allocInit("MTLResidencySetDescriptor");
        defer objc.release(res_desc);
        objc.send(void, res_desc, "setInitialCapacity:", .{residency_capacity});
        const residency = objc.send(?Id, device, "newResidencySetWithDescriptor:error:", .{ res_desc, &err }) orelse return error.ResidencyFailed;
        objc.send(void, queue, "addResidencySet:", .{residency});
        objc.send(void, m3_queue, "addResidencySet:", .{residency});

        var cbs: [max_cmd_buffers]Id = undefined;
        for (&cbs) |*cb| {
            cb.* = objc.send(?Id, device, "newCommandBuffer", .{}) orelse return error.NoMetal4Support;
        }

        var ctx = Context{
            .device = device,
            .queue = queue,
            .m3_queue = m3_queue,
            .allocator = cmd_alloc,
            .cmd_buffers = cbs,
            .arg_table = arg_table,
            .shared_event = shared_event,
            .residency = residency,
            .constants_buf = undefined,
        };
        ctx.constants_buf = try ctx.allocBuffer(constants_ring_bytes);
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.last_m3_cb) |cb| objc.release(cb);
        self.constants_buf.deinit();
        for (self.cmd_buffers) |cb| objc.release(cb);
        objc.release(self.residency);
        objc.release(self.shared_event);
        objc.release(self.arg_table);
        objc.release(self.allocator);
        objc.release(self.m3_queue);
        objc.release(self.queue);
        objc.release(self.device);
    }

    pub fn isMetal4(self: *const Context) bool {
        return self.api == .metal4_device or self.api == .metal4_none;
    }

    pub fn deviceName(self: *const Context) []const u8 {
        const s = objc.send(?Id, self.device, "name", .{}) orelse return "Apple GPU";
        const cstr = objc.send(?[*:0]const u8, s, "UTF8String", .{}) orelse return "Apple GPU";
        return std.mem.span(cstr);
    }

    /// Allocate a shared, hazard-untracked Metal buffer and register it in the residency set.
    pub fn allocBuffer(self: *Context, byte_len: usize) Error!Buffer {
        const aligned_len = @max((byte_len + 15) & ~@as(usize, 15), 16);
        const opts = resource_storage_shared | resource_hazard_untracked;
        const id = objc.send(?Id, self.device, "newBufferWithLength:options:", .{ aligned_len, opts }) orelse return error.BufferAllocFailed;
        const gpu_addr = objc.send(u64, id, "gpuAddress", .{});
        const ptr = objc.send([*]u8, id, "contents", .{});
        objc.send(void, self.residency, "addAllocation:", .{id});
        return .{ .id = id, .gpu_addr = gpu_addr, .ptr = ptr, .len = aligned_len };
    }

    /// Commit residency set so all allocated buffers are permanently resident on GPU.
    pub fn commitResidency(self: *Context) void {
        objc.send(void, self.residency, "commit", .{});
        objc.send(void, self.residency, "requestResidency", .{});
    }

    pub fn createFence(self: *Context) Error!Fence {
        const id = objc.send(?Id, self.device, "newFence", .{}) orelse return error.NoMetal4Support;
        return .{ .id = id };
    }

    pub fn compileSource(self: *const Context, source: []const u8, precision: MathPrecision) Error!Library {
        const pool = objc.Pool.push();
        defer pool.pop();

        const src = objc.nsString(source);
        defer objc.release(src);
        const opts = objc.allocInit("MTLCompileOptions");
        defer objc.release(opts);

        // Metal 4.1 language version = (4 << 16) | 1
        objc.send(void, opts, "setLanguageVersion:", .{@as(usize, (4 << 16) | 1)});
        objc.send(void, opts, "setMathMode:", .{@as(isize, 0)}); // safe IEEE math
        objc.send(void, opts, "setMathFloatingPointFunctions:", .{@intFromEnum(precision)});

        var err: ?Id = null;
        const lib = objc.send(?Id, self.device, "newLibraryWithSource:options:error:", .{ src, opts, &err }) orelse {
            std.debug.print("Metal compile error:\n{s}\n", .{objc.errorMessage(err)});
            return error.ShaderCompileFailed;
        };
        return .{ .id = lib };
    }

    pub fn makePipeline(self: *const Context, lib: Library, name: []const u8) Error!Pipeline {
        const pool = objc.Pool.push();
        defer pool.pop();

        const fn_name = objc.nsString(name);
        defer objc.release(fn_name);
        const func = objc.send(?Id, lib.id, "newFunctionWithName:", .{fn_name}) orelse {
            std.debug.print("Metal kernel not found: {s}\n", .{name});
            return error.FunctionNotFound;
        };
        defer objc.release(func);

        const desc = objc.allocInit("MTLComputePipelineDescriptor");
        defer objc.release(desc);
        objc.send(void, desc, "setComputeFunction:", .{func});

        var err: ?Id = null;
        const pso = objc.send(?Id, self.device, "newComputePipelineStateWithDescriptor:options:reflection:error:", .{
            desc,
            @as(usize, 0),
            @as(?*anyopaque, null),
            &err,
        }) orelse {
            std.debug.print("Metal pipeline build failed ({s}): {s}\n", .{ name, objc.errorMessage(err) });
            return error.PipelineFailed;
        };
        var out = Pipeline{ .id = pso };
        const n = @min(name.len, out.name.len);
        @memcpy(out.name[0..n], name[0..n]);
        return out;
    }

    pub fn beginCommandBuffer(self: *Context) CommandBuffer {
        self.stats.command_buffers += 1;
        if (!self.isMetal4()) {
            const cb = objc.send(Id, self.m3_queue, "commandBufferWithUnretainedReferences", .{});
            _ = objc.retain(cb);
            if (self.last_m3_cb) |old| objc.release(old);
            self.last_m3_cb = cb;
            return .{ .ctx = self, .id = cb, .is_m4 = false };
        }
        if (self.waited_tick == self.tick) {
            objc.send(void, self.allocator, "reset", .{});
            self.cmd_used = 0;
        }
        std.debug.assert(self.cmd_used < max_cmd_buffers);
        const cb = self.cmd_buffers[self.cmd_used];
        self.cmd_used += 1;
        objc.send(void, cb, "beginCommandBufferWithAllocator:", .{self.allocator});
        return .{ .ctx = self, .id = cb, .is_m4 = true };
    }

    pub fn waitIdle(self: *Context) Error!void {
        if (!self.isMetal4()) {
            if (self.last_m3_cb) |cb| {
                objc.send(void, cb, "waitUntilCompleted", .{});
            }
            return;
        }
        if (self.waited_tick == self.tick) return;
        const ok = objc.send(bool, self.shared_event, "waitUntilSignaledValue:timeoutMS:", .{ self.tick, @as(u64, 30_000) });
        if (!ok) return error.GpuTimeout;
        self.waited_tick = self.tick;
    }
};

pub const CommandBuffer = struct {
    ctx: *Context,
    id: Id,
    is_m4: bool = true,

    pub fn beginCompute(self: CommandBuffer) Encoder {
        if (!self.is_m4) {
            const dtype: usize = if (self.ctx.api == .metal3_concurrent) 1 else 0;
            const enc = objc.send(Id, self.id, "computeCommandEncoderWithDispatchType:", .{dtype});
            return .{ .ctx = self.ctx, .id = enc, .is_m4 = false };
        }
        const enc = objc.send(Id, self.id, "computeCommandEncoder", .{});
        objc.send(void, enc, "setArgumentTable:", .{self.ctx.arg_table});
        var e = Encoder{ .ctx = self.ctx, .id = enc, .is_m4 = true };
        if (self.ctx.timeline) |tl| {
            if (tl.count == 0) e.writeTimestamp("start");
        }
        return e;
    }

    pub fn commit(self: CommandBuffer) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "commit", .{});
            return;
        }
        objc.send(void, self.id, "endCommandBuffer", .{});
        const list = [1]Id{self.id};
        objc.send(void, self.ctx.queue, "commit:count:", .{ @as([*]const Id, &list), @as(usize, 1) });
        self.ctx.tick += 1;
        objc.send(void, self.ctx.queue, "signalEvent:value:", .{ self.ctx.shared_event, self.ctx.tick });
    }
};

pub const Encoder = struct {
    ctx: *Context,
    id: Id,
    is_m4: bool = true,
    dirty: bool = false,
    last_label: [32]u8 = @splat(0),

    pub fn setPipeline(self: *Encoder, p: Pipeline) void {
        self.last_label = p.name;
        objc.send(void, self.id, "setComputePipelineState:", .{p.id});
    }

    pub fn setBuffer(self: *Encoder, buf: Buffer, offset: usize, index: usize) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "setBuffer:offset:atIndex:", .{ buf.id, offset, index });
            return;
        }
        objc.send(void, self.ctx.arg_table, "setAddress:atIndex:", .{ buf.gpu_addr + offset, index });
    }

    pub fn setRef(self: *Encoder, ref: BufferRef, index: usize) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "setBuffer:offset:atIndex:", .{ ref.buf_id, ref.offset, index });
            return;
        }
        objc.send(void, self.ctx.arg_table, "setAddress:atIndex:", .{ ref.gpu_addr, index });
    }

    /// Copy small POD constants into the Metal 4 ring buffer and bind its GPU address.
    pub fn setValue(self: *Encoder, value: anytype, index: usize) void {
        const bytes = std.mem.asBytes(&value);
        if (!self.is_m4) {
            objc.send(void, self.id, "setBytes:length:atIndex:", .{ @as([*]const u8, bytes.ptr), bytes.len, index });
            return;
        }
        if (self.ctx.constants_bump + bytes.len > self.ctx.constants_buf.len) {
            self.ctx.constants_bump = 0;
        }
        const off = self.ctx.constants_bump;
        @memcpy(self.ctx.constants_buf.ptr[off..][0..bytes.len], bytes);
        objc.send(void, self.ctx.arg_table, "setAddress:atIndex:", .{ self.ctx.constants_buf.gpu_addr + off, index });
        self.ctx.constants_bump += (bytes.len + 63) & ~@as(usize, 63);
    }

    pub fn dispatchGroups(self: *Encoder, groups: Size, threads: Size) void {
        objc.send(void, self.id, "dispatchThreadgroups:threadsPerThreadgroup:", .{ groups, threads });
        self.ctx.stats.dispatches += 1;
        self.dirty = true;
    }

    pub fn dispatchThreads(self: *Encoder, grid: Size, threads: Size) void {
        objc.send(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
        self.ctx.stats.dispatches += 1;
        self.dirty = true;
    }

    /// Device-coherent dispatch-to-dispatch barrier (elided if nothing dispatched since previous barrier).
    pub fn barrier(self: *Encoder) void {
        if (!self.dirty) return;
        self.dirty = false;
        if (!self.is_m4) {
            if (self.ctx.api == .metal3_concurrent) {
                self.ctx.stats.barriers += 1;
                objc.send(void, self.id, "memoryBarrierWithScope:", .{@as(usize, 1)}); // MTLBarrierScopeBuffers
            }
            return;
        }
        self.ctx.stats.barriers += 1;
        const vis: usize = if (self.ctx.api == .metal4_device) visibility_device else 0;
        objc.send(void, self.id, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", .{
            stage_dispatch,
            stage_dispatch,
            vis,
        });
        if (self.ctx.timeline != null) self.writeTimestamp(std.mem.sliceTo(&self.last_label, 0));
    }

    pub fn updateFence(self: *Encoder, fence: Fence) void {
        self.ctx.stats.fences += 1;
        if (!self.is_m4) {
            objc.send(void, self.id, "updateFence:", .{fence.id});
            return;
        }
        objc.send(void, self.id, "updateFence:afterEncoderStages:", .{ fence.id, stage_dispatch });
    }

    pub fn waitFence(self: *Encoder, fence: Fence) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "waitForFence:", .{fence.id});
            return;
        }
        objc.send(void, self.id, "waitForFence:beforeEncoderStages:", .{ fence.id, stage_dispatch });
    }

    pub fn writeTimestamp(self: *Encoder, label: []const u8) void {
        if (!self.is_m4) return;
        const tl = self.ctx.timeline orelse return;
        if (tl.count >= tl.names.len) return;
        tl.names[tl.count] = @splat(0);
        const n = @min(label.len, 32);
        @memcpy(tl.names[tl.count][0..n], label[0..n]);
        objc.send(void, self.id, "writeTimestampWithGranularity:intoHeap:atIndex:", .{
            @as(isize, 1),
            tl.heap,
            tl.count,
        });
        tl.count += 1;
    }

    pub fn end(self: *Encoder) void {
        if (self.is_m4 and self.dirty and self.ctx.timeline != null) {
            self.writeTimestamp(std.mem.sliceTo(&self.last_label, 0));
        }
        objc.send(void, self.id, "endEncoding", .{});
    }
};
