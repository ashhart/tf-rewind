//! Embedded Metal 4 shader sources so m4lm is a completely self-contained binary / dylib.
pub const qk = @embedFile("qk.metal");
pub const norm_ops = @embedFile("norm_ops.metal");
pub const gdn_ops = @embedFile("gdn_ops.metal");
pub const attention = @embedFile("attention.metal");
pub const mlp_ops = @embedFile("mlp_ops.metal");
pub const sample = @embedFile("sample.metal");
pub const dflash = @embedFile("dflash.metal");
