"""m4lm — Drop-in mlx-lm replacement backed directly by libm4lm.dylib (zero Apple MLX)."""
from __future__ import annotations

import ctypes
import glob
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence


class _M4GenOptions(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_uint32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("min_p", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("enable_thinking", ctypes.c_uint8),
        ("use_chat_template", ctypes.c_uint8),
        ("stop_on_eos", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
    ]


class _M4Stats(ctypes.Structure):
    _fields_ = [
        ("prompt_tokens", ctypes.c_uint32),
        ("completion_tokens", ctypes.c_uint32),
        ("rounds", ctypes.c_uint32),
        ("drafts_accepted", ctypes.c_uint32),
        ("drafts_proposed", ctypes.c_uint32),
        ("cached_tokens", ctypes.c_uint32),
        ("decode_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("tok_per_sec", ctypes.c_double),
    ]


_TOKEN_CB = ctypes.CFUNCTYPE(
    ctypes.c_uint8,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)

_WAVE_CB = ctypes.CFUNCTYPE(
    ctypes.c_uint8,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)


def _load_dylib() -> ctypes.CDLL:
    candidates = [
        os.environ.get("M4LM_LIB", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "zig-out", "lib", "libm4lm.dylib")),
        os.path.abspath("zig-out/lib/libm4lm.dylib"),
        os.path.abspath("m4lm/zig-out/lib/libm4lm.dylib"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            lib = ctypes.CDLL(path)
            break
    else:
        raise RuntimeError(
            "libm4lm.dylib not found. Build it with `zig build --release=fast` "
            "or set M4LM_LIB=/path/to/libm4lm.dylib."
        )

    lib.m4lm_load.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.m4lm_load.restype = ctypes.c_void_p
    lib.m4lm_free.argtypes = [ctypes.c_void_p]
    lib.m4lm_free.restype = None
    lib.m4lm_device_name.argtypes = [ctypes.c_void_p]
    lib.m4lm_device_name.restype = ctypes.c_char_p
    lib.m4lm_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(_M4GenOptions),
        _TOKEN_CB,
        ctypes.c_void_p,
        ctypes.POINTER(_M4Stats),
    ]
    lib.m4lm_generate.restype = ctypes.c_int32
    lib.m4lm_stream_parallel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        _WAVE_CB,
        ctypes.c_void_p,
        ctypes.POINTER(_M4Stats),
    ]
    lib.m4lm_stream_parallel.restype = ctypes.c_int32
    lib.m4lm_serve.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_char_p]
    lib.m4lm_serve.restype = ctypes.c_int32
    return lib


_LIB: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_dylib()
    return _LIB


def resolve_model_path(path_or_repo: str) -> str:
    """Accept either a local directory or a HuggingFace repo ID cached under ~/.cache/huggingface/hub."""
    expanded = os.path.expanduser(path_or_repo)
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub")
    repo_key = "models--" + path_or_repo.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(hub_dir, repo_key, "snapshots", "*")))
    if snaps:
        return snaps[-1]
    raise FileNotFoundError(f"Model directory or cached HF repo not found: {path_or_repo}")


@dataclass
class GenerationResponse:
    text: str
    token: int
    prompt_tokens: int = 0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class WaveResponse:
    texts: list[str]
    prefix_tokens: int
    cached_tokens: int
    total_generated: int
    decode_ms: float
    total_ms: float
    decode_tok_per_sec: float


class M4Tokenizer:
    """Lightweight mlx_lm-compatible tokenizer facade (formatting runs in native Zig)."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def apply_chat_template(
        self,
        messages: Sequence[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        if tokenize:
            raise ValueError("Pass tokenize=False; m4lm tokenizes natively in Zig.")
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n")
        if add_generation_prompt:
            if enable_thinking:
                parts.append("<|im_start|>assistant\n<think>\n")
            else:
                parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)


class M4Model:
    def __init__(
        self,
        model_dir: str,
        draft_model: Optional[str] = None,
        cross_draft_model: Optional[str] = None,
        max_context: int = 8192,
        max_streams: int = 16,
    ):
        self.model_dir = resolve_model_path(model_dir)
        self.draft_dir = resolve_model_path(draft_model) if draft_model else None
        self.cross_dir = resolve_model_path(cross_draft_model) if cross_draft_model else None
        self._handle = _lib().m4lm_load(
            self.model_dir.encode("utf-8"),
            self.draft_dir.encode("utf-8") if self.draft_dir else None,
            self.cross_dir.encode("utf-8") if self.cross_dir else None,
            int(max_context),
            int(max_streams),
        )
        if not self._handle:
            raise RuntimeError(f"m4lm_load failed for {self.model_dir}")
        self.last_stats: Optional[_M4Stats] = None

    @property
    def device_name(self) -> str:
        return _lib().m4lm_device_name(self._handle).decode("utf-8")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib().m4lm_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def serve(self, port: int = 8080, model_name: Optional[str] = None) -> None:
        name = (model_name or os.path.basename(self.model_dir)).encode("utf-8")
        rc = _lib().m4lm_serve(self._handle, int(port), name)
        if rc != 0:
            raise RuntimeError(f"m4lm_serve exited with status {rc}")


def load(
    path_or_hf_repo: str,
    draft_model: Optional[str] = None,
    cross_draft_model: Optional[str] = None,
    max_context: int = 8192,
    max_streams: int = 16,
) -> tuple[M4Model, M4Tokenizer]:
    """Drop-in replacement for `mlx_lm.load()` — runs directly on Metal with zero Apple MLX."""
    model = M4Model(
        path_or_hf_repo,
        draft_model=draft_model,
        cross_draft_model=cross_draft_model,
        max_context=max_context,
        max_streams=max_streams,
    )
    return model, M4Tokenizer(model.model_dir)


def stream_generate(
    model: M4Model,
    tokenizer: M4Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int = 42,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> Iterator[GenerationResponse]:
    """Drop-in streaming generator compatible with `mlx_lm.stream_generate()`."""
    q: queue.Queue = queue.Queue()
    prompt_bytes = prompt.encode("utf-8")
    opts = _M4GenOptions(
        max_tokens=int(max_tokens),
        temperature=float(temp),
        top_p=float(top_p),
        top_k=int(top_k),
        min_p=float(min_p),
        seed=int(seed),
        enable_thinking=1 if enable_thinking else 0,
        use_chat_template=1 if use_chat_template else 0,
        stop_on_eos=1,
        _pad=0,
    )
    stats = _M4Stats()
    err_holder: list[int] = [0]

    @_TOKEN_CB
    def on_token(_ud, tok_id, utf8_ptr, utf8_len):
        raw = ctypes.string_at(utf8_ptr, utf8_len)
        q.put((int(tok_id), raw.decode("utf-8", errors="replace")))
        return 1

    def worker():
        rc = _lib().m4lm_generate(
            model._handle,
            prompt_bytes,
            len(prompt_bytes),
            ctypes.byref(opts),
            on_token,
            None,
            ctypes.byref(stats),
        )
        err_holder[0] = rc
        q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    count = 0
    t_first: Optional[float] = None
    while True:
        item = q.get()
        if item is None:
            break
        tok_id, piece = item
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        count += 1
        dt = now - t_first
        tps = (count - 1) / dt if (count > 1 and dt > 1e-6) else 0.0
        yield GenerationResponse(text=piece, token=tok_id, generation_tokens=count, generation_tps=tps)

    t.join()
    if err_holder[0] != 0:
        raise RuntimeError(f"m4lm_generate failed with code {err_holder[0]}")
    model.last_stats = stats


def generate(
    model: M4Model,
    tokenizer: M4Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
) -> str:
    """Drop-in replacement for `mlx_lm.generate()`."""
    chunks: list[str] = []
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
    ):
        chunks.append(resp.text)
        if verbose:
            print(resp.text, end="", flush=True)
    text = "".join(chunks)
    if verbose and model.last_stats is not None:
        s = model.last_stats
        tpr = s.completion_tokens / max(1, s.rounds)
        print(
            f"\n==========\nPrompt: {s.prompt_tokens} tok | "
            f"Generation: {s.completion_tokens} tok ({s.tok_per_sec:.1f} tok/s, {tpr:.2f} tok/rnd)"
        )
    return text


def stream_parallel(
    model: M4Model,
    shared_prompt: str,
    delta_prompts: Sequence[str],
    max_tokens: int | Sequence[int] = 256,
    temp: float | Sequence[float] = 0.0,
) -> WaveResponse:
    """Execute a zero-copy prefix-forked lockstep multi-stream wave (1..16 streams) on Metal."""
    n = len(delta_prompts)
    shared_b = shared_prompt.encode("utf-8")
    delta_bs = [d.encode("utf-8") for d in delta_prompts]

    ptrs = (ctypes.c_char_p * n)(*delta_bs)
    lens = (ctypes.c_size_t * n)(*(len(b) for b in delta_bs))
    max_list = [max_tokens] * n if isinstance(max_tokens, int) else list(max_tokens)
    temp_list = [temp] * n if isinstance(temp, (int, float)) else list(temp)
    max_arr = (ctypes.c_uint32 * n)(*(int(x) for x in max_list))
    temp_arr = (ctypes.c_float * n)(*(float(x) for x in temp_list))

    buffers: list[list[str]] = [[] for _ in range(n)]
    stats = _M4Stats()

    @_WAVE_CB
    def on_wave_tok(_ud, sidx, _tok_id, utf8_ptr, utf8_len):
        buffers[int(sidx)].append(ctypes.string_at(utf8_ptr, utf8_len).decode("utf-8", errors="replace"))
        return 1

    rc = _lib().m4lm_stream_parallel(
        model._handle,
        shared_b,
        len(shared_b),
        n,
        ptrs,
        lens,
        max_arr,
        temp_arr,
        on_wave_tok,
        None,
        ctypes.byref(stats),
    )
    if rc != 0:
        raise RuntimeError(f"m4lm_stream_parallel failed with code {rc}")
    model.last_stats = stats
    return WaveResponse(
        texts=["".join(b) for b in buffers],
        prefix_tokens=int(stats.prompt_tokens),
        cached_tokens=int(stats.cached_tokens),
        total_generated=int(stats.completion_tokens),
        decode_ms=float(stats.decode_ms),
        total_ms=float(stats.total_ms),
        decode_tok_per_sec=float(stats.tok_per_sec),
    )
