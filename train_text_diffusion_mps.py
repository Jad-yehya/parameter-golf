from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import time
import uuid
import zlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    mask_id: int = 1024
    padded_vocab: int = 1088
    seq_len: int = 512
    num_layers: int = 4
    model_dim: int = 256
    num_heads: int = 4
    mlp_mult: float = 2.0
    self_condition: bool = False
    tie_output_head: bool = False
    rope_base: float = 10000.0

    @property
    def total_vocab(self) -> int:
        return self.mask_id + 1


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", "1337"))

    device = os.environ.get("DEVICE", "cuda")
    compute_dtype = os.environ.get("COMPUTE_DTYPE", "auto")
    iterations = int(os.environ.get("ITERATIONS", "400"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "50"))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", "0"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "16384"))
    max_train_tokens = int(os.environ.get("MAX_TRAIN_TOKENS", "0"))
    max_val_tokens = int(os.environ.get("MAX_VAL_TOKENS", "1048576"))
    val_seqs = int(os.environ.get("VAL_SEQS", "64"))
    eval_steps = int(os.environ.get("DIFFUSION_EVAL_STEPS", "32"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "20"))
    warmdown_steps = int(os.environ.get("WARMDOWN_STEPS", "100"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "0"))

    vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))
    mask_id = int(os.environ.get("MASK_ID", str(vocab_size)))
    padded_vocab = int(os.environ.get("PADDED_VOCAB", "1088"))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "512"))
    num_layers = int(os.environ.get("NUM_LAYERS", "4"))
    model_dim = int(os.environ.get("MODEL_DIM", "256"))
    num_heads = int(os.environ.get("NUM_HEADS", "4"))
    mlp_mult = float(os.environ.get("MLP_MULT", "2.0"))
    self_condition = bool(int(os.environ.get("SELF_CONDITION", "0")))
    tie_output_head = bool(int(os.environ.get("TIE_OUTPUT_HEAD", "0")))
    mask_pattern = os.environ.get("MASK_PATTERN", "independent")
    span_len = int(os.environ.get("SPAN_LEN", "4"))
    noise_eps = float(os.environ.get("NOISE_EPS", "0.1"))

    lr = float(os.environ.get("LR", "6e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.1"))
    beta1 = float(os.environ.get("BETA1", "0.9"))
    beta2 = float(os.environ.get("BETA2", "0.95"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
    out_dir = os.environ.get("OUT_DIR", "logs")


def resolve_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported DEVICE={name!r}")


def resolve_compute_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if normalized == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return dtypes[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported COMPUTE_DTYPE={name!r}") from exc


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return nullcontext()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def log_linear_noise(t: Tensor, eps: float = 0.1) -> tuple[Tensor, Tensor]:
    alpha = 1.0 - (1.0 - eps) * t
    sigma = -torch.log(alpha.clamp_min(1e-8))
    return sigma, alpha


def make_corrupted_batch(
    x0: Tensor,
    t: Tensor,
    mask_id: int,
    eps: float,
    pattern: str = "independent",
    span_len: int = 4,
) -> tuple[Tensor, Tensor]:
    _, alpha = log_linear_noise(t, eps=eps)
    move_chance = 1.0 - alpha
    if pattern == "independent":
        mask = torch.rand_like(x0.float()) < move_chance[:, None]
    elif pattern == "span":
        mask = torch.zeros_like(x0, dtype=torch.bool)
        seq_len = x0.shape[1]
        for row, chance in enumerate(move_chance.detach().cpu()):
            width = max(1, int(round(float(chance) * seq_len)))
            width = min(seq_len, max(width, span_len))
            start = int(torch.randint(0, seq_len - width + 1, (1,)).item())
            mask[row, start : start + width] = True
    else:
        raise ValueError(f"Unsupported MASK_PATTERN={pattern!r}")
    xt = torch.where(mask, torch.full_like(x0, mask_id), x0)
    return xt, mask


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def count_sequence_bytes(
    tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    prev = torch.empty_like(tokens)
    prev[:, 0] = 0
    prev[:, 1:] = tokens[:, :-1]
    token_bytes = base_bytes_lut[tokens].to(dtype=torch.int32)
    token_bytes += (has_leading_space_lut[tokens] & ~is_boundary_token_lut[prev]).to(dtype=torch.int32)
    return token_bytes.sum(dim=-1)


def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return self.cos[:, :, :seq_len].to(dtype=dtype), self.sin[:, :, :seq_len].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half),
            persistent=False,
        )
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, sigma: Tensor) -> Tensor:
        emb = sigma[:, None].float() * self.freqs[None, :].to(sigma.device)
        return self.mlp(torch.cat((emb.sin(), emb.cos()), dim=-1))


class AdaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.proj(cond).to(dtype=x.dtype).unsqueeze(1).chunk(2, dim=-1)
        return rms_norm(x) * (1.0 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, seq_len: int, rope_base: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.rotary = Rotary(self.head_dim, seq_len, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, dim = x.shape
        q = self.c_q(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(seq_len, q.dtype)
        q = apply_rotary_emb(rms_norm(q), cos, sin)
        k = apply_rotary_emb(rms_norm(k), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seq_len, dim))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.model_dim * cfg.mlp_mult)
        self.adaln_attn = AdaLN(cfg.model_dim)
        self.adaln_mlp = AdaLN(cfg.model_dim)
        self.attn = Attention(cfg.model_dim, cfg.num_heads, cfg.seq_len, cfg.rope_base)
        self.fc = nn.Linear(cfg.model_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = x + self.attn(self.adaln_attn(x, cond))
        h = torch.relu(self.fc(self.adaln_mlp(x, cond))).square()
        return x + self.proj(h)


class DiffusionLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.model_dim)
        self.sigma_map = TimestepEmbedder(cfg.model_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.model_dim, elementwise_affine=False)
        self.head = None if cfg.tie_output_head else nn.Linear(cfg.model_dim, cfg.padded_vocab, bias=False)
        self.output_head_scale = cfg.model_dim**-0.5 if cfg.tie_output_head else 1.0
        if self.head is not None:
            nn.init.zeros_(self.head.weight)
        self.self_cond_proj = nn.Linear(cfg.total_vocab, cfg.model_dim, bias=False) if cfg.self_condition else None
        if self.self_cond_proj is not None:
            nn.init.zeros_(self.self_cond_proj.weight)

    def forward_logits(self, xt: Tensor, sigma: Tensor, self_condition_logits: Tensor | None = None) -> Tensor:
        x = self.tok_emb(xt)
        if self.self_cond_proj is not None and self_condition_logits is not None:
            probs = torch.softmax(self_condition_logits.detach().float(), dim=-1).to(dtype=x.dtype)
            x = x + self.self_cond_proj(probs)
        cond = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cond)
        h = self.final_norm(x)
        if self.cfg.tie_output_head:
            real_logits = F.linear(h, self.tok_emb.weight[: self.cfg.vocab_size]) * self.output_head_scale
            mask_logits = real_logits.new_zeros(*real_logits.shape[:-1], self.cfg.total_vocab - self.cfg.vocab_size)
            logits = torch.cat((real_logits, mask_logits), dim=-1).float()
        else:
            logits = self.head(h)[..., : self.cfg.total_vocab].float()
        return logits

    def subs_log_probs(self, xt: Tensor, sigma: Tensor, self_condition_logits: Tensor | None = None) -> Tensor:
        logits = self.forward_logits(xt, sigma, self_condition_logits=self_condition_logits)
        logits[:, :, self.cfg.mask_id] = -1e6
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        frozen = torch.full_like(logits, -1e6)
        frozen.scatter_(-1, xt.clamp_max(self.cfg.mask_id)[..., None], 0.0)
        visible = (xt != self.cfg.mask_id)[..., None]
        return torch.where(visible, frozen, logits)


def mdlm_loss(
    model: DiffusionLM,
    x0: Tensor,
    mask_pattern: str = "independent",
    eps: float = 0.1,
    span_len: int = 4,
) -> Tensor:
    bsz = x0.shape[0]
    t = torch.rand(bsz // 2 + 1, device=x0.device)
    t = torch.cat((t, 1.0 - t))[:bsz].clamp(1e-5, 1.0 - 1e-5)
    sigma, alpha = log_linear_noise(t, eps=eps)
    xt, mask = make_corrupted_batch(
        x0, t, mask_id=model.cfg.mask_id, eps=eps, pattern=mask_pattern, span_len=span_len
    )
    self_condition_logits = None
    if model.cfg.self_condition:
        with torch.no_grad():
            self_condition_logits = model.forward_logits(xt, sigma)
    log_probs = model.subs_log_probs(xt, sigma, self_condition_logits=self_condition_logits)
    log_p_x0 = torch.gather(log_probs, -1, x0[..., None]).squeeze(-1)
    dsigma = (1.0 - eps) / alpha
    loss = (dsigma[:, None] * (-log_p_x0) * mask.float()).sum() / (x0.numel())
    return loss


@torch.no_grad()
def variational_elbo_bits(
    model: DiffusionLM,
    x0: Tensor,
    n_steps: int,
    eps: float,
    mask_pattern: str,
    span_len: int,
    compute_dtype: torch.dtype,
) -> Tensor:
    bsz, seq_len = x0.shape
    total_bits = torch.zeros(bsz, device=x0.device)
    t_grid = torch.arange(1, n_steps + 1, device=x0.device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid, eps=eps)
    total_bits += seq_len * float(alpha_grid[-1]) * math.log(model.cfg.vocab_size) / math.log(2.0)
    alpha_prev = 1.0
    for step in range(n_steps):
        alpha_curr = float(alpha_grid[step])
        t = torch.full((bsz,), float(t_grid[step]), device=x0.device)
        sigma = sigma_grid[step].expand(bsz)
        xt, mask = make_corrupted_batch(
            x0, t, mask_id=model.cfg.mask_id, eps=eps, pattern=mask_pattern, span_len=span_len
        )
        self_condition_logits = None
        if model.cfg.self_condition:
            with autocast_context(x0.device, compute_dtype):
                self_condition_logits = model.forward_logits(xt, sigma)
        with autocast_context(x0.device, compute_dtype):
            log_probs = model.subs_log_probs(xt, sigma, self_condition_logits=self_condition_logits)
        log_p_x0 = torch.gather(log_probs.float(), -1, x0[..., None]).squeeze(-1)
        reveal_prob = (alpha_prev - alpha_curr) / max(1.0 - alpha_curr, 1e-12)
        step_bits = reveal_prob * (-log_p_x0) * mask.float() / math.log(2.0)
        total_bits += step_bits.sum(dim=-1)
        alpha_prev = alpha_curr
    return total_bits


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.int64, copy=False))


def load_tokens(pattern: str, max_tokens: int = 0) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    chunks: list[Tensor] = []
    remaining = max_tokens
    for file in files:
        tokens = load_data_shard(file)
        if max_tokens > 0:
            take = min(tokens.numel(), remaining)
            chunks.append(tokens[:take])
            remaining -= take
            if remaining <= 0:
                break
        else:
            chunks.append(tokens)
    return torch.cat(chunks).contiguous()


def sample_batch(tokens: Tensor, batch_seqs: int, seq_len: int, device: torch.device) -> Tensor:
    max_start = tokens.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError("training tokens too short")
    starts = torch.randint(0, max_start, (batch_seqs,))
    batch = torch.stack([tokens[int(start) : int(start) + seq_len] for start in starts])
    return batch.to(device=device, dtype=torch.long)


def lr_for_step(args: Hyperparameters, step: int) -> float:
    if step < args.warmup_steps:
        return args.lr * float(step + 1) / max(args.warmup_steps, 1)
    warmdown_start = max(args.iterations - args.warmdown_steps, args.warmup_steps)
    if step < warmdown_start:
        return args.lr
    progress = (step - warmdown_start) / max(args.iterations - warmdown_start, 1)
    return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def compressed_state_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(zlib.compress(buf.getvalue(), level=9))


@torch.no_grad()
def evaluate_bpb(
    args: Hyperparameters,
    model: DiffusionLM,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()
    total_bits = 0.0
    total_bytes = 0.0
    max_seqs = min(args.val_seqs, val_tokens.numel() // args.train_seq_len)
    for seq_idx in range(max_seqs):
        start = seq_idx * args.train_seq_len
        x0 = val_tokens[start : start + args.train_seq_len].unsqueeze(0).to(device)
        bits = variational_elbo_bits(
            model,
            x0,
            n_steps=args.eval_steps,
            eps=args.noise_eps,
            mask_pattern=args.mask_pattern,
            span_len=args.span_len,
            compute_dtype=compute_dtype,
        )
        bytes_ = count_sequence_bytes(x0, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        total_bits += float(bits.sum().item())
        total_bytes += float(bytes_.sum().item())
    model.train()
    return total_bits / max(total_bytes, 1.0), total_bits / max(max_seqs * args.train_seq_len, 1)


def main() -> None:
    args = Hyperparameters()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    compute_dtype = resolve_compute_dtype(args.compute_dtype, device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.out_dir, exist_ok=True)
    logfile = Path(args.out_dir) / f"{args.run_id}.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        mask_id=args.mask_id,
        padded_vocab=args.padded_vocab,
        seq_len=args.train_seq_len,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        self_condition=args.self_condition,
        tie_output_head=args.tie_output_head,
    )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab={sp.vocab_size()}")
    train_tokens = load_tokens(os.path.join(args.data_path, "fineweb_train_*.bin"), args.max_train_tokens)
    val_tokens = load_tokens(os.path.join(args.data_path, "fineweb_val_*.bin"), args.max_val_tokens)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model = DiffusionLM(cfg).to(device=device, dtype=compute_dtype)
    if device.type != "cuda":
        model.float()
        compute_dtype = torch.float32
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    batch_seqs = max(1, args.train_batch_tokens // args.train_seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"run_id:{args.run_id}")
    log(f"device:{device} compute_dtype:{compute_dtype}")
    log(f"model_params:{n_params}")
    log(
        f"diffusion:self_condition:{args.self_condition} mask_pattern:{args.mask_pattern} "
        f"tie_output_head:{args.tie_output_head} noise_eps:{args.noise_eps} eval_steps:{args.eval_steps}"
    )
    log(
        f"shape:layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} "
        f"seq_len:{args.train_seq_len} batch_seqs:{batch_seqs}"
    )
    log(f"data:train_tokens:{train_tokens.numel()} val_tokens:{val_tokens.numel()}")

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(args.iterations):
        lr = lr_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x0 = sample_batch(train_tokens, batch_seqs, args.train_seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, compute_dtype):
            loss = mdlm_loss(
                model,
                x0,
                mask_pattern=args.mask_pattern,
                eps=args.noise_eps,
                span_len=args.span_len,
            )
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if args.train_log_every > 0 and (step + 1 <= 5 or (step + 1) % args.train_log_every == 0):
            synchronize_device(device)
            elapsed = time.perf_counter() - t0
            log(
                f"step:{step + 1}/{args.iterations} train_loss:{np.mean(losses[-args.train_log_every:]):.4f} "
                f"lr:{lr:.3e} elapsed:{elapsed:.1f}s"
            )
        if args.val_loss_every > 0 and (step + 1) % args.val_loss_every == 0:
            bpb, bits_per_token = evaluate_bpb(
                args,
                model,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                device,
                compute_dtype,
            )
            log(f"step:{step + 1}/{args.iterations} val_var_bpb:{bpb:.6f} bits_per_token:{bits_per_token:.6f}")
        if args.max_wallclock_seconds > 0 and time.perf_counter() - t0 >= args.max_wallclock_seconds:
            log(f"stopping_early:wallclock step:{step + 1}")
            break

    synchronize_device(device)
    train_time = time.perf_counter() - t0
    bpb, bits_per_token = evaluate_bpb(
        args,
        model,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        device,
        compute_dtype,
    )
    bytes_zlib = compressed_state_bytes(model)
    result = {
        "run_id": args.run_id,
        "seed": args.seed,
        "val_var_bpb": bpb,
        "bits_per_token": bits_per_token,
        "train_loss": float(np.mean(losses[-min(len(losses), 20) :])) if losses else None,
        "train_seconds": train_time,
        "model_params": n_params,
        "compressed_state_zlib_bytes": bytes_zlib,
        "self_condition": args.self_condition,
        "tie_output_head": args.tie_output_head,
        "mask_pattern": args.mask_pattern,
        "eval_steps": args.eval_steps,
    }
    log(f"compressed_state_zlib_bytes:{bytes_zlib}")
    log(f"final_var_bpb:{bpb:.8f} bits_per_token:{bits_per_token:.8f} train_seconds:{train_time:.2f}")
    with (Path(args.out_dir) / f"{args.run_id}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
