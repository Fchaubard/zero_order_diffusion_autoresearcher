"""
TRM-inspired Recursive Diffusion Transformer pretraining script. Single-GPU, single-file.
Recursive DiT with shared-weight blocks and two-level latent refinement (z_H, z_L),
adapted from TinyRecursiveModels for image generation on ImageNet ILSVRC2012.

IMPORTANT: NO NOISE SCHEDULES! NO TEACHER FORCING!
The TRM recursion IS the denoising process. z_H and z_L are initialized from
random noise and refined through recursive shared-weight blocks to produce
a clean image. The model learns end-to-end: noise → recursion → clean image.
Variable recursion depth during training enables test-time compute scaling.

Supports BPTT (backprop through full recursion) and 1.5-SPSA (zero-order).

Usage:
    uv run train.py                              # truncated BPTT (default)
    uv run train.py --solver spsa                # SPSA with default params
    uv run train.py --solver spsa --use-curvature  # 1.5-SPSA with curvature
    uv run train.py --solver spsa --search-strategy local  # SPSA with adaptive LR
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import argparse
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import wandb

from prepare import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES, TIME_BUDGET, FlowMatching, make_dataloader, evaluate_fid

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train TRM-inspired Recursive DiT with truncated BPTT or 1.5-SPSA zero-order optimization",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Solver selection
solver_group = parser.add_argument_group("solver")
solver_group.add_argument("--solver", type=str, default="tbptt",
    choices=["tbptt", "spsa"],
    help="Training solver: tbptt (truncated BPTT, TRM-style) or spsa (zero-order, no teacher forcing)")

# Model architecture
model_group = parser.add_argument_group("model architecture")
model_group.add_argument("--h-cycles", type=int, default=2,
    help="Number of outer (H-level) recursive cycles (TRM-style)")
model_group.add_argument("--l-cycles", type=int, default=3,
    help="Number of inner (L-level) recursive cycles per H-cycle")
model_group.add_argument("--l-layers", type=int, default=1,
    help="Number of transformer layers in the shared L-level block")
model_group.add_argument("--n-embd", type=int, default=768,
    help="Model embedding dimension")
model_group.add_argument("--patch-size", type=int, default=4,
    help="Patch size for patch embedding")
model_group.add_argument("--head-dim", type=int, default=64,
    help="Target head dimension for attention")

# Training (common)
train_group = parser.add_argument_group("training")
train_group.add_argument("--total-batch-size", type=int, default=256,
    help="Total images per optimizer step")
train_group.add_argument("--device-batch-size", type=int, default=64,
    help="Per-device batch size (reduce if OOM)")
train_group.add_argument("--lr", type=float, default=2e-3,
    help="Learning rate")
train_group.add_argument("--weight-decay", type=float, default=0.0,
    help="Weight decay for transformer parameters (TBPTT AdamW)")
train_group.add_argument("--warmup-ratio", type=float, default=0.1,
    help="Fraction of time budget for LR warmup")
train_group.add_argument("--warmdown-ratio", type=float, default=0.3,
    help="Fraction of time budget for LR warmdown")
train_group.add_argument("--final-lr-frac", type=float, default=0.0,
    help="Final LR as fraction of initial LR")
train_group.add_argument("--sigma-min", type=float, default=1e-4,
    help="Flow matching minimum variance for numerical stability")
train_group.add_argument("--time-budget", type=int, default=TIME_BUDGET,
    help="Training time budget in seconds")
train_group.add_argument("--seed", type=int, default=42,
    help="Random seed for reproducibility")
train_group.add_argument("--warmup-steps", type=int, default=10,
    help="Steps before counting toward time budget (compilation warmup)")
train_group.add_argument("--fail-threshold", type=float, default=100.0,
    help="Loss threshold for fast fail (abort if loss exceeds this)")
train_group.add_argument("--num-workers", type=int, default=4,
    help="Dataloader workers (reduce to 1 when running 8 experiments in parallel)")
train_group.add_argument("--p-uncond", type=float, default=0.0,
    help="Probability of dropping class labels for classifier-free guidance training")
train_group.add_argument("--cfg-scale", type=float, default=1.0,
    help="Classifier-free guidance scale at inference (1.0 = no guidance)")
train_group.add_argument("--logit-normal-t", action="store_true",
    help="Use logit-normal timestep sampling instead of uniform")

# Backprop-specific
bp_group = parser.add_argument_group("tbptt-specific")
bp_group.add_argument("--adam-beta1", type=float, default=0.9,
    help="Adam beta1")
bp_group.add_argument("--adam-beta2", type=float, default=0.95,
    help="Adam beta2")

# SPSA-specific
spsa_group = parser.add_argument_group("SPSA-specific")
spsa_group.add_argument("--epsilon", type=float, default=None,
    help="SPSA perturbation size (default: tied to --lr)")
spsa_group.add_argument("--n-perts", type=int, default=40,
    help="Number of perturbations per SPSA step")
spsa_group.add_argument("--use-curvature", action="store_true",
    help="Enable 1.5-SPSA with curvature scaling")
spsa_group.add_argument("--saturating-alpha", type=float, default=0.1,
    help="Exponent for saturating curvature in 1.5-SPSA")
spsa_group.add_argument("--lambda-reg", type=float, default=1.0,
    help="Minimum curvature regularization in 1.5-SPSA")
spsa_group.add_argument("--memory-efficient", action="store_true",
    help="Memory efficient mode: regenerate perturbation directions via RNG")
spsa_group.add_argument("--spsa-accum-steps", type=int, default=1,
    help="Batch accumulation steps for SPSA gradient estimation stability")
spsa_group.add_argument("--denoising-steps", type=int, default=20,
    help="Number of ODE steps for full denoising during SPSA training (T)")
spsa_group.add_argument("--spsa-weight-decay", type=float, default=0.0,
    help="Weight decay for SPSA parameter updates")

# SPSA search strategy
search_group = parser.add_argument_group("SPSA search strategy")
search_group.add_argument("--search-strategy", type=str, default="none",
    choices=["none", "line", "local"],
    help="LR search strategy: none (use schedule), line (full range), local (10x up/down)")
search_group.add_argument("--search-n-points", type=int, default=20,
    help="Number of points for line search")
search_group.add_argument("--search-n-seeds", type=int, default=3,
    help="Number of seeds to average over for each LR evaluation")
search_group.add_argument("--search-lr-min", type=float, default=1e-7,
    help="Minimum LR for line search range")
search_group.add_argument("--search-lr-max", type=float, default=1e-1,
    help="Maximum LR for line search range")
search_group.add_argument("--search-patience", type=int, default=20,
    help="Steps without improvement before triggering re-search")
search_group.add_argument("--search-diverge-threshold", type=float, default=1.5,
    help="Loss EMA increase ratio to trigger re-search")
search_group.add_argument("--search-ema-alpha", type=float, default=0.1,
    help="EMA smoothing factor for plateau detection in search")

args = parser.parse_args()

# Derived: epsilon defaults to lr if not set
SPSA_EPSILON = args.epsilon if args.epsilon is not None else args.lr

# ---------------------------------------------------------------------------
# TRM-inspired Recursive Diffusion Transformer Model
# ---------------------------------------------------------------------------

@dataclass
class RecursiveDiTConfig:
    img_size: int = 64
    patch_size: int = 4
    in_channels: int = 3
    n_embd: int = 768
    n_head: int = 12
    num_classes: int = 1000
    h_cycles: int = 2
    l_cycles: int = 3
    l_layers: int = 1


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.n_embd,
                              kernel_size=config.patch_size, stride=config.patch_size)
        self.num_patches = (config.img_size // config.patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)             # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, n_embd, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, n_embd, bias=True),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd, bias=True),
        )
        self.freq_dim = freq_dim

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args_t = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args_t), torch.sin(args_t)], dim=-1)
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, n_embd)

    def forward(self, labels):
        return self.embedding(labels)


class RecursiveDiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning and SwiGLU MLP (TRM-style)."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        # Self-attention
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # SwiGLU MLP (from TRM architecture)
        inter = (-(round(4 * config.n_embd * 2 / 3) // -256)) * 256
        self.gate_up_proj = nn.Linear(config.n_embd, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, config.n_embd, bias=False)

        # AdaLN modulation: 6 params per dim (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True),
        )

    def forward(self, x, c):
        # c is conditioning vector (B, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with AdaLN
        B, N, C = x.shape
        h = norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        q = self.c_q(h).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(h).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(h).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, C)
        h = self.c_proj(h)
        x = x + gate_msa.unsqueeze(1) * h

        # SwiGLU MLP with AdaLN
        h = norm(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        h = self.down_proj(F.silu(gate) * up)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class RecursiveDiT(nn.Module):
    """
    TRM-inspired Recursive Diffusion Transformer.
    Uses shared-weight L-level blocks applied recursively with two-level
    latent states (z_H, z_L) and input injection at each cycle.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.n_embd))

        # Time and class conditioning
        self.time_embed = TimestepEmbedder(config.n_embd)
        self.label_embed = LabelEmbedder(config.num_classes, config.n_embd)

        # Shared L-level blocks (reused across all recursive cycles, TRM-style)
        self.l_blocks = nn.ModuleList([RecursiveDiTBlock(config) for _ in range(config.l_layers)])

        # Spatial smoothing: 3x3 depthwise conv for local patch blending (gated, starts as no-op)
        self.spatial_smooth = nn.Conv2d(config.n_embd, config.n_embd, 3, padding=1, groups=config.n_embd, bias=False)
        self.spatial_gate = nn.Parameter(torch.zeros(config.n_embd))
        self._patch_grid = config.img_size // config.patch_size

        # Fixed initial latent states (TRM-style: buffers, not parameters)
        self.register_buffer('z_H_init', torch.empty(1, 1, config.n_embd))
        self.register_buffer('z_L_init', torch.empty(1, 1, config.n_embd))

        # Output head
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True),
        )
        self.final_proj = nn.Linear(config.n_embd, config.patch_size ** 2 * config.in_channels, bias=True)

    def unpatchify(self, x):
        """Convert patch tokens back to image."""
        p = self.config.patch_size
        h = w = self.config.img_size // p
        c = self.config.in_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def l_level(self, x, input_injection, c):
        """Apply shared L-level blocks with input injection."""
        x = x + input_injection
        for block in self.l_blocks:
            x = block(x, c)
        return x

    @torch.no_grad()
    def init_weights(self):
        # Standard Xavier init for linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Fixed initial latent states (TRM-style truncated normal, std=1)
        self.z_H_init.normal_(0, 1).clamp_(-2, 2)
        self.z_L_init.normal_(0, 1).clamp_(-2, 2)

        # Patch embedding
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # Label embedding
        nn.init.normal_(self.label_embed.embedding.weight, std=0.02)

        # Zero-out AdaLN modulation outputs so gates start at zero
        for block in self.l_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-out final projection (DiT zero-init)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def estimate_flops(self):
        """Estimated FLOPs per image (forward + backward)."""
        num_patches = self.patch_embed.num_patches
        l_block_params = sum(p.numel() for p in self.l_blocks.parameters())
        other_params = sum(p.numel() for p in self.parameters()) - l_block_params
        # Total L-level passes: h_cycles * l_cycles (z_L updates) + h_cycles (z_H updates)
        total_passes = self.config.h_cycles * self.config.l_cycles + self.config.h_cycles
        attn_flops_per_pass = 4 * num_patches * num_patches * self.config.n_embd * self.config.l_layers
        # Shared blocks are reused across passes, FLOPs scale with passes
        return 6 * other_params + total_passes * (6 * l_block_params + attn_flops_per_pass)

    def num_scaling_params(self):
        patch_embed = sum(p.numel() for p in self.patch_embed.parameters())
        pos_embed = self.pos_embed.numel()
        time_embed = sum(p.numel() for p in self.time_embed.parameters())
        label_embed = sum(p.numel() for p in self.label_embed.parameters())
        l_blocks = sum(p.numel() for p in self.l_blocks.parameters())
        final = sum(p.numel() for p in self.final_adaLN.parameters()) + \
                sum(p.numel() for p in self.final_proj.parameters())
        total = patch_embed + pos_embed + time_embed + label_embed + l_blocks + final
        return {
            'patch_embed': patch_embed, 'pos_embed': pos_embed,
            'time_embed': time_embed, 'label_embed': label_embed,
            'l_blocks': l_blocks, 'final': final, 'total': total,
        }

    def setup_optimizer(self, lr=1e-4, weight_decay=0.0, adam_betas=(0.9, 0.999)):
        # Separate param groups: embeddings (no decay) vs block weights (decay)
        embed_params = list(self.patch_embed.parameters()) + [self.pos_embed] + \
                       list(self.time_embed.parameters()) + list(self.label_embed.parameters())
        block_params = list(self.l_blocks.parameters()) + [self.spatial_gate] + \
                       list(self.spatial_smooth.parameters()) + \
                       list(self.final_adaLN.parameters()) + list(self.final_proj.parameters())
        assert len(embed_params) + len(block_params) == len(list(self.parameters()))
        param_groups = [
            dict(params=embed_params, lr=lr, betas=adam_betas, weight_decay=0.0),
            dict(params=block_params, lr=lr, betas=adam_betas, weight_decay=weight_decay),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _decode(self, z_H):
        """Decode z_H to image space. No conditioning — pure readout of the answer."""
        # No AdaLN conditioning for decode (following TRM: z_H is the answer, just read it)
        x = norm(z_H)
        x = self.final_proj(x)
        x = self.unpatchify(x)
        return x

    def _recurse(self, z_H, z_L, c_class, num_thtl=None):
        """
        TRM recursive refinement — the recursion IS the denoiser.
        NO external noise schedules. NO teacher forcing.

        z_H and z_L start from random noise and get refined to produce a clean image.
        Each macro-cycle corresponds to a denoising "step" — the model receives a
        progress signal (via time_embed) telling it which stage of denoising it's in.
        This is NOT an external noise schedule — it's an internal recursion progress
        indicator that helps the shared block adapt its behavior at each stage.

        Following TRM: H-updates are UNCONDITIONED (pure accumulator).
        L-updates receive class conditioning AND recursion progress.
        """
        t_htl = num_thtl if num_thtl is not None else self.config.h_cycles
        t_l = self.config.l_cycles
        t_h = 1  # TODO: ablate T_H > 1

        # Zero conditioning for H-updates
        c_zero = torch.zeros_like(c_class)

        for _htl in range(t_htl):
            # Loop 1: T_L rounds of L-consolidation (refine working memory)
            for _l in range(t_l):
                z_L = self.l_level(z_L, z_H, c_class)
            # Loop 2: T_H rounds of H-consolidation (update answer)
            for _h in range(t_h):
                z_H = self.l_level(z_H, z_L, c_zero)

        return self._decode(z_H)

    def forward(self, x, t, class_labels=None, num_thtl=None):
        """
        TRM-Diffusion forward pass.

        IMPORTANT: We do NOT use noise schedules or teacher forcing.
        The TRM recursion IS the denoising process.
        z_H and z_L are initialized from random noise (the "carry")
        and refined through recursive shared-weight blocks to produce a clean image.

        For compatibility with prepare.py's evaluate_fid (which uses ODE solver),
        we return (clean_prediction - x) as a "velocity". The ODE solver then
        iteratively moves x toward the clean prediction.

        Args:
            x: (B, C, H, W) — current state (noise at start, refined image later)
            t: (B,) — IGNORED during training. Used only for conditioning compatibility.
            class_labels: (B,) — class indices for conditional generation
            num_thtl: optional override for number of macro-cycles (for variable depth training)
        """
        B = x.shape[0]
        device = x.device
        N = self.patch_embed.num_patches
        C = self.config.n_embd

        # Conditioning: class labels only. Recursion progress is added inside _recurse.
        c_class = torch.zeros(B, self.config.n_embd, device=device)
        if class_labels is not None:
            label_emb = self.label_embed(class_labels)
            if self.training and self.p_uncond > 0:
                drop_mask = torch.rand(B, device=device) < self.p_uncond
                label_emb = label_emb * (~drop_mask).float().unsqueeze(-1)
            c_class = c_class + label_emb

        # Initialize the carry from input — this is where variety comes from.
        z_H = self.patch_embed(x) + self.pos_embed  # carry seeded from input noise
        z_L = self.z_L_init.expand(B, N, -1)        # fixed working memory init

        # The TRM recursion IS the denoiser
        clean_pred = self._recurse(z_H, z_L, c_class, num_thtl=num_thtl)

        if not self.training and self.cfg_scale > 1.0 and class_labels is not None:
            # CFG: also run unconditional and combine
            c_uncond = torch.zeros(B, self.config.n_embd, device=device)
            z_H_u = self.patch_embed(x) + self.pos_embed
            z_L_u = self.z_L_init.expand(B, N, -1)
            uncond_pred = self._recurse(z_H_u, z_L_u, c_uncond, num_thtl=num_thtl)
            clean_pred = uncond_pred + self.cfg_scale * (clean_pred - uncond_pred)

        # Return as "velocity" for ODE solver compatibility with prepare.py
        #
        # During TRAINING (t=0): velocity = clean_pred - noise
        # The training loop does: clean_pred = noise + velocity = clean_pred. Correct.
        #
        # During EVAL (ODE solver, 50 steps with dt=1/50):
        # The model only knows how to denoise PURE NOISE. The ODE solver feeds
        # increasingly clean images, which the model wasn't trained on.
        # Solution: scale velocity so ONE step reaches clean_pred, then subsequent
        # steps get near-zero velocity (model sees ~clean input, predicts ~clean).
        # Simple velocity: clean_pred - x
        # During training: loss uses noise + velocity = clean_pred
        # During direct eval: same formula, clean_pred = noise + velocity
        return clean_pred - x

# ---------------------------------------------------------------------------
# Triton Kernels (SPSA bit-packed perturbations)
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_and_apply(w_ptr, packed_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    """Apply bit-packed Rademacher perturbation to weights: w += alpha * sign(packed_bits)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    w = tl.load(w_ptr + offsets, mask=mask)
    byte_idx = offsets // 8
    bit_idx = offsets % 8
    packed_byte = tl.load(packed_ptr + byte_idx, mask=mask)
    bit = (packed_byte >> bit_idx) & 1
    sign = tl.where(bit == 1, 1.0, -1.0)
    tl.store(w_ptr + offsets, w + alpha * sign, mask=mask)


@triton.jit
def _unpack_and_accumulate(grad_ptr, packed_ptr, n_elements, coeff, BLOCK_SIZE: tl.constexpr):
    """Accumulate SPSA gradient estimate: grad += coeff * sign(packed_bits)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    byte_idx = offsets // 8
    bit_idx = offsets % 8
    packed_byte = tl.load(packed_ptr + byte_idx, mask=mask)
    bit = (packed_byte >> bit_idx) & 1
    sign = tl.where(bit == 1, 1.0, -1.0)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    tl.store(grad_ptr + offsets, grad + coeff * sign, mask=mask)

# ---------------------------------------------------------------------------
# SPSA Trainer
# ---------------------------------------------------------------------------

TRITON_BLOCK_SIZE = 1024  # Block size for triton kernels


class SPSATrainer:
    """
    1.5-SPSA zero-order optimizer with bit-packed Rademacher perturbations.
    Estimates gradients via finite differences without backpropagation.
    Supports curvature scaling (1.5-SPSA), memory-efficient mode,
    and adaptive LR search (line search, local search).
    """

    def __init__(self, model, lr, epsilon, n_perts, use_curvature,
                 saturating_alpha, lambda_reg, memory_efficient,
                 accum_steps, weight_decay):
        self.lr = lr
        self.epsilon = epsilon
        self.n_perts = n_perts
        self.use_curvature = use_curvature
        self.saturating_alpha = saturating_alpha
        self.lambda_reg = lambda_reg
        self.memory_efficient = memory_efficient
        self.accum_steps = accum_steps
        self.weight_decay = weight_decay

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.total = sum(p.numel() for p in self.params)
        self.packed_size = (self.total + 7) // 8

        # Compute offsets for bit-packed perturbations
        self.param_info = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.param_info.append({
                'param': p,
                'offset': offset,
                'packed_offset': offset // 8,
                'numel': numel,
                'grid': ((numel + TRITON_BLOCK_SIZE - 1) // TRITON_BLOCK_SIZE,),
            })
            offset += numel

        # Gradient accumulator per param (bf16) — skip if memory_efficient
        if not memory_efficient:
            self.grads = [torch.zeros(info['numel'], device='cuda', dtype=torch.bfloat16)
                          for info in self.param_info]
        else:
            self.grads = None

        mode_str = " [MEMORY EFFICIENT]" if memory_efficient else ""
        accum_str = f", accum={accum_steps}" if accum_steps > 1 else ""
        curv_str = " [1.5-SPSA]" if use_curvature else ""
        print(f"SPSATrainer: {self.total/1e6:.2f}M params, packed={self.packed_size/1e6:.1f}MB"
              f"{curv_str}{mode_str}{accum_str}")

    def step(self, loss_fn, iteration):
        """Perform one SPSA step. loss_fn(batch_idx) -> float."""
        if self.memory_efficient:
            return self._step_memory_efficient(loss_fn, iteration)

        for g in self.grads:
            g.zero_()

        total_loss = 0.0

        # For 1.5-SPSA, get clean loss once per iteration
        if self.use_curvature:
            loss_clean = 0.0
            for batch_idx in range(self.accum_steps):
                loss_clean += loss_fn(batch_idx)
            loss_clean /= self.accum_steps

        for pert_idx in range(self.n_perts):
            # Generate bit-packed Rademacher random (8x less memory than full float)
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            # Apply +epsilon perturbation
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_plus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_plus += loss_fn(batch_idx)
            loss_plus /= self.accum_steps

            # Apply -2*epsilon (net: -epsilon from original)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2 * self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_minus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_minus += loss_fn(batch_idx)
            loss_minus /= self.accum_steps

            # Restore to original (+epsilon to undo the -2*epsilon)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            # Compute gradient coefficient
            if self.use_curvature:
                curv = abs(loss_plus - 2 * loss_clean + loss_minus) / (self.epsilon ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts)

            # Accumulate gradient
            for i, info in enumerate(self.param_info):
                _unpack_and_accumulate[info['grid']](
                    self.grads[i], packed[info['packed_offset']:],
                    info['numel'], grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            total_loss += (loss_plus + loss_minus) / 2
            del packed

        # Apply update: theta -= lr * grad
        for info, grad in zip(self.param_info, self.grads):
            info['param'].data.view(-1).sub_(grad, alpha=self.lr)

        # Apply weight decay
        if self.weight_decay > 0:
            for info in self.param_info:
                info['param'].data.mul_(1 - self.lr * self.weight_decay)

        return total_loss / self.n_perts

    def _step_memory_efficient(self, loss_fn, iteration):
        """Memory-efficient step: regenerate directions via RNG instead of caching grads."""
        total_loss = 0.0
        grad_coeffs = []

        if self.use_curvature:
            loss_clean = 0.0
            for batch_idx in range(self.accum_steps):
                loss_clean += loss_fn(batch_idx)
            loss_clean /= self.accum_steps

        for pert_idx in range(self.n_perts):
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_plus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_plus += loss_fn(batch_idx)
            loss_plus /= self.accum_steps

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2 * self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_minus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_minus += loss_fn(batch_idx)
            loss_minus /= self.accum_steps

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            if self.use_curvature:
                curv = abs(loss_plus - 2 * loss_clean + loss_minus) / (self.epsilon ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * self.epsilon * self.n_perts)

            grad_coeffs.append(grad_coeff)
            total_loss += (loss_plus + loss_minus) / 2
            del packed

        # Second pass: replay RNG to apply update
        for pert_idx, grad_coeff in enumerate(grad_coeffs):
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -self.lr * grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)
            del packed

        if self.weight_decay > 0:
            for info in self.param_info:
                info['param'].data.mul_(1 - self.lr * self.weight_decay)

        return total_loss / self.n_perts

    def probe_loss_at_lr(self, loss_fn, test_lr, seed=0, accum_batches=1):
        """Probe loss after a hypothetical SPSA step at test_lr. Does NOT permanently update weights."""
        if self.memory_efficient:
            return self._probe_loss_at_lr_memory_efficient(loss_fn, test_lr, seed, accum_batches)

        test_eps = test_lr  # Tied: epsilon = lr

        for g in self.grads:
            g.zero_()

        if self.use_curvature:
            loss_clean = loss_fn()

        for pert_idx in range(self.n_perts):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_plus = 0.0
            for _ in range(accum_batches):
                loss_plus += loss_fn()
            loss_plus /= accum_batches

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2 * test_eps, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_minus = 0.0
            for _ in range(accum_batches):
                loss_minus += loss_fn()
            loss_minus /= accum_batches

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            if self.use_curvature:
                curv = abs(loss_plus - 2 * loss_clean + loss_minus) / (test_eps ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts)

            for i, info in enumerate(self.param_info):
                _unpack_and_accumulate[info['grid']](
                    self.grads[i], packed[info['packed_offset']:],
                    info['numel'], grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)
            del packed

        # Temporarily apply step
        for info, grad in zip(self.param_info, self.grads):
            info['param'].data.view(-1).sub_(grad, alpha=test_lr)

        loss_after = 0.0
        for _ in range(accum_batches):
            loss_after += loss_fn()
        loss_after /= accum_batches

        # Restore weights
        for info, grad in zip(self.param_info, self.grads):
            info['param'].data.view(-1).add_(grad, alpha=test_lr)

        return loss_after

    def _probe_loss_at_lr_memory_efficient(self, loss_fn, test_lr, seed=0, accum_batches=1):
        """Memory-efficient version of probe_loss_at_lr."""
        test_eps = test_lr
        grad_coeffs = []

        if self.use_curvature:
            loss_clean = loss_fn()

        for pert_idx in range(self.n_perts):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_plus = 0.0
            for _ in range(accum_batches):
                loss_plus += loss_fn()
            loss_plus /= accum_batches

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2 * test_eps, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_minus = 0.0
            for _ in range(accum_batches):
                loss_minus += loss_fn()
            loss_minus /= accum_batches

            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_eps, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            if self.use_curvature:
                curv = abs(loss_plus - 2 * loss_clean + loss_minus) / (test_eps ** 2)
                curvature = max(curv ** self.saturating_alpha, self.lambda_reg)
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts * curvature)
            else:
                grad_coeff = (loss_plus - loss_minus) / (2 * test_eps * self.n_perts)

            grad_coeffs.append(grad_coeff)
            del packed

        # Apply step temporarily
        for pert_idx, grad_coeff in enumerate(grad_coeffs):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -test_lr * grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)
            del packed

        loss_after = 0.0
        for _ in range(accum_batches):
            loss_after += loss_fn()
        loss_after /= accum_batches

        # Restore weights
        for pert_idx, grad_coeff in enumerate(grad_coeffs):
            torch.manual_seed(seed * 10000 + pert_idx)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
            for info in self.param_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], test_lr * grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)
            del packed

        return loss_after

    def line_search_lr(self, loss_fn, lr_min, lr_max, n_points, seed=0,
                       n_seeds=1, resample_batch_fn=None):
        """Line search for optimal LR in log space."""
        import numpy as np
        log_min = math.log10(lr_min)
        log_max = math.log10(lr_max)
        log_lrs = np.linspace(log_min, log_max, n_points)
        lrs = [10 ** log_lr for log_lr in log_lrs]
        print(f"  Line search: testing {n_points} points in [{lr_min:.0e}, {lr_max:.0e}] (n_seeds={n_seeds})")

        results = []
        for i, lr_candidate in enumerate(lrs):
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)
                loss = self.probe_loss_at_lr(loss_fn, lr_candidate, probe_seed, accum_batches=1)
                total_loss += loss
            avg_loss = total_loss / n_seeds
            results.append((lr_candidate, avg_loss))
            print(f"    [{i+1}/{n_points}] lr={lr_candidate:.2e} -> loss={avg_loss:.4f}")

        best_lr, best_loss = min(results, key=lambda x: x[1])
        print(f"  Line search complete: best lr={best_lr:.2e} (loss={best_loss:.4f})")
        return best_lr

    def local_search_lr(self, loss_fn, current_lr, seed=0, n_seeds=1,
                        resample_batch_fn=None):
        """Local search: test current_lr * 10 and current_lr / 10."""
        candidates = [current_lr / 10, current_lr, current_lr * 10]
        print(f"  Local search: testing {current_lr/10:.1e}, {current_lr:.1e}, {current_lr*10:.1e} (n_seeds={n_seeds})")

        results = []
        for i, lr_candidate in enumerate(candidates):
            total_loss = 0.0
            for s in range(n_seeds):
                probe_seed = seed + i * 100000 + s * 1000
                if resample_batch_fn is not None:
                    resample_batch_fn(probe_seed)
                loss = self.probe_loss_at_lr(loss_fn, lr_candidate, probe_seed, accum_batches=1)
                total_loss += loss
            avg_loss = total_loss / n_seeds
            results.append((lr_candidate, avg_loss))
            print(f"    lr={lr_candidate:.2e} -> loss={avg_loss:.4f}")

        best_lr, best_loss = min(results, key=lambda x: x[1])
        current_loss = results[1][1]  # Middle candidate is current_lr
        improved = best_lr != current_lr

        if improved:
            print(f"  Local search: {current_lr:.2e} -> {best_lr:.2e} (loss {current_loss:.4f} -> {best_loss:.4f})")
        else:
            print(f"  Local search: staying at {current_lr:.2e} (already best)")
        return best_lr, improved

# ---------------------------------------------------------------------------
# Setup: model, optimizer/trainer, dataloader, flow matching
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

num_heads = args.n_embd // args.head_dim
config = RecursiveDiTConfig(
    img_size=IMG_SIZE, patch_size=args.patch_size, in_channels=IMG_CHANNELS,
    n_embd=args.n_embd, n_head=num_heads, num_classes=NUM_CLASSES,
    h_cycles=args.h_cycles, l_cycles=args.l_cycles, l_layers=args.l_layers,
)
print(f"Solver: {args.solver}")
print(f"Model config: {asdict(config)}")

# Wandb logging
run_name = f"{args.solver}_h{args.h_cycles}l{args.l_cycles}d{args.l_layers}_n{args.n_embd}"
wandb.init(
    project="trm-recursive-diffusion",
    name=run_name,
    config={**asdict(config), "solver": args.solver, "lr": args.lr,
            "total_batch_size": args.total_batch_size, "time_budget": args.time_budget,
            "p_uncond": args.p_uncond, "cfg_scale": args.cfg_scale,
            "logit_normal_t": args.logit_normal_t,
            **({k: getattr(args, k) for k in ["n_perts", "denoising_steps", "epsilon",
               "use_curvature", "search_strategy"] if hasattr(args, k)} if args.solver == "spsa" else {})},
)

with torch.device("meta"):
    model = RecursiveDiT(config)
model.to_empty(device=device)
model.init_weights()
model.p_uncond = args.p_uncond
model.cfg_scale = args.cfg_scale

# EMA model for evaluation (Polyak averaging)
import copy
ema_model = copy.deepcopy(model)
ema_model.eval()
ema_decay = 0.999

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_image = model.estimate_flops()
print(f"Estimated FLOPs per image: {num_flops_per_image:e}")

assert args.total_batch_size % args.device_batch_size == 0
grad_accum_steps = args.total_batch_size // args.device_batch_size

flow_matching = FlowMatching(sigma_min=args.sigma_min)

# Solver-specific setup
if args.solver == "tbptt":
    optimizer = model.setup_optimizer(
        lr=args.lr, weight_decay=args.weight_decay,
        adam_betas=(args.adam_beta1, args.adam_beta2),
    )
    # model = torch.compile(model, dynamic=False)  # disabled: nvcc permission issue
else:
    # SPSA: no BPTT, no torch.compile (in-place weight perturbation)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    trainer = SPSATrainer(
        model=model, lr=args.lr, epsilon=SPSA_EPSILON,
        n_perts=args.n_perts, use_curvature=args.use_curvature,
        saturating_alpha=args.saturating_alpha, lambda_reg=args.lambda_reg,
        memory_efficient=args.memory_efficient,
        accum_steps=args.spsa_accum_steps, weight_decay=args.spsa_weight_decay,
    )

train_loader = make_dataloader("train", args.device_batch_size, num_workers=args.num_workers)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {args.time_budget}s")
if args.solver == "tbptt":
    print(f"Gradient accumulation steps: {grad_accum_steps}")
else:
    print(f"SPSA: n_perts={args.n_perts}, epsilon={SPSA_EPSILON:.2e}, denoising_steps={args.denoising_steps}")
    print(f"SPSA: accum_steps={args.spsa_accum_steps}, search={args.search_strategy}")
    if args.use_curvature:
        print(f"SPSA: 1.5-SPSA curvature: alpha={args.saturating_alpha}, lambda={args.lambda_reg}")

# Schedules (all based on progress = training_time / time_budget)

def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return progress / args.warmup_ratio if args.warmup_ratio > 0 else 1.0
    elif progress < 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / args.warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * args.final_lr_frac

# ---------------------------------------------------------------------------
# SPSA loss functions and probe setup
# ---------------------------------------------------------------------------

if args.solver == "spsa":
    # Batches held for consistent ±epsilon evaluation within each SPSA step
    spsa_batches = []
    # Noise seed for deterministic ODE noise (same noise for +/- perturbations)
    noise_seed = [0]

    def spsa_loss_fn(batch_idx=0):
        """SPSA training loss: full ODE denoising, no teacher forcing."""
        x_b, y_b = spsa_batches[batch_idx % len(spsa_batches)]
        with torch.no_grad(), autocast_ctx:
            return flow_matching.denoising_loss(
                model, x_b, class_labels=y_b,
                denoising_steps=args.denoising_steps,
                noise_seed=noise_seed[0] + batch_idx,
            )

    # Probe setup for LR search
    probe_data = [None, None]

    def resample_probe_batch(seed):
        x_b, y_b, _ = next(train_loader)
        probe_data[0] = x_b
        probe_data[1] = y_b

    # Initialize probe batch
    resample_probe_batch(0)

    def probe_loss_fn():
        """Fixed-batch loss for LR probing."""
        with torch.no_grad(), autocast_ctx:
            return flow_matching.denoising_loss(
                model, probe_data[0], class_labels=probe_data[1],
                denoising_steps=args.denoising_steps,
                noise_seed=42,  # fixed noise for consistent probing
            )

    # Initial LR search if strategy != none and != local
    if args.search_strategy == "line":
        print("=" * 70)
        print("INITIAL LR SEARCH")
        print("=" * 70)
        best_lr = trainer.line_search_lr(
            probe_loss_fn, args.search_lr_min, args.search_lr_max,
            n_points=args.search_n_points, seed=99999,
            n_seeds=args.search_n_seeds, resample_batch_fn=resample_probe_batch,
        )
        trainer.lr = best_lr
        if args.epsilon is None:
            trainer.epsilon = best_lr
            SPSA_EPSILON = best_lr
        print(f"Using lr={trainer.lr:.2e}, eps={trainer.epsilon:.2e}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# SPSA search state
if args.solver == "spsa" and args.search_strategy != "none":
    search_stall_count = 0
    search_loss_ema = None
    search_best_ema = None
    search_last_loss = None
    search_count = 1 if args.search_strategy == "line" else 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    if args.solver == "tbptt":
        # =====================================================================
        # TRM RECURSIVE DENOISING — NO NOISE SCHEDULES! NO TEACHER FORCING!
        #
        # The TRM recursion IS the denoiser. We start from random noise,
        # the model's recursive shared-weight blocks refine it into a clean image.
        # Loss = reconstruction error between model output and clean target.
        #
        # IMPORTANT: We do NOT use flow_matching.forward_sample(), timestep
        # sampling, velocity targets, or any noise schedule. The entire
        # "denoising" happens inside the model's recursive forward pass.
        # =====================================================================
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                batch_size = x.shape[0]

                # ============================================================
                # NOISE CURRICULUM: start easy, get harder over training
                #
                # noise_level goes from 0 → 1 over the training budget.
                # carry = (1 - noise_level) * clean_image + noise_level * noise
                #
                # Early training: carry ≈ clean image (easy correction task)
                #   → model learns what sharp images look like, no mode collapse
                # Late training: carry ≈ pure noise (full generation task)
                #   → model uses its learned priors to generate from scratch
                #
                # This is NOT a noise schedule — there is no timestep conditioning.
                # The noise level is a CURRICULUM parameter that changes over
                # training time. At test time, always start from pure noise.
                # ============================================================
                noise_level = min(total_training_time / args.time_budget, 1.0)  # 0 → 1 over training
                noise = torch.randn_like(x)
                carry = (1.0 - noise_level) * x + noise_level * noise

                # Variable recursion depth during training
                import random
                num_thtl = random.choice([2, 3, 4, 5])

                # Forward: carry → model recursion → clean image prediction
                velocity_pred = model(carry, torch.zeros(batch_size, device=x.device),
                                      class_labels=y, num_thtl=num_thtl)
                clean_pred = carry + velocity_pred

                # Loss: reconstruct the clean image
                loss = F.mse_loss(clean_pred, x)

            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        # Progress and schedules
        progress = min(total_training_time / args.time_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)
        # (spatial smooth removed — speed wins at higher LR)
        # Progressive EMA: decay increases from 0.99 to 0.9999 over training
        current_decay = 0.99 + progress * 0.0099  # 0.99 -> 0.9999
        with torch.no_grad():
            for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(current_decay).add_(p_model.data, alpha=1 - current_decay)
        train_loss_f = train_loss.item()

    else:
        # ----- SPSA: zero-order, no teacher forcing -----
        # Load batches for this step (consistent across ±epsilon evaluations)
        spsa_batches.clear()
        for _ in range(args.spsa_accum_steps):
            x_b, y_b, epoch = next(train_loader)
            spsa_batches.append((x_b, y_b))

        noise_seed[0] = step * 100

        # SPSA step
        train_loss_f = trainer.step(spsa_loss_fn, step)

        # Update LR based on schedule (when not using search)
        progress = min(total_training_time / args.time_budget, 1.0)
        if args.search_strategy == "none":
            lrm = get_lr_multiplier(progress)
            trainer.lr = args.lr * lrm
            if args.epsilon is None:
                trainer.epsilon = trainer.lr
        else:
            lrm = trainer.lr / args.lr if args.lr > 0 else 1.0

        # Adaptive search: plateau/divergence detection
        if args.search_strategy != "none":
            if search_loss_ema is None:
                search_loss_ema = train_loss_f
                search_best_ema = train_loss_f
                search_last_loss = train_loss_f
            else:
                search_loss_ema = args.search_ema_alpha * train_loss_f + \
                                  (1 - args.search_ema_alpha) * search_loss_ema

            need_search = False

            # Divergence check
            if search_loss_ema > search_last_loss * args.search_diverge_threshold:
                print(f"\nDIVERGENCE: EMA {search_loss_ema:.4f} > {search_last_loss:.4f} * {args.search_diverge_threshold}")
                need_search = True

            # Plateau check
            if search_loss_ema < search_best_ema - 0.01:
                search_best_ema = search_loss_ema
                search_stall_count = 0
            else:
                search_stall_count += 1
                if search_stall_count >= args.search_patience:
                    print(f"\nPLATEAU: EMA {search_loss_ema:.4f} stalled for {args.search_patience} steps")
                    need_search = True

            if need_search:
                search_count += 1
                if args.search_strategy == "local":
                    new_lr, improved = trainer.local_search_lr(
                        probe_loss_fn, trainer.lr,
                        seed=99999 + search_count * 1000000,
                        n_seeds=args.search_n_seeds,
                        resample_batch_fn=resample_probe_batch,
                    )
                    if not improved:
                        print("EARLY STOP: local search found no improvement")
                        break
                    trainer.lr = new_lr
                    if args.epsilon is None:
                        trainer.epsilon = new_lr
                else:  # line
                    new_lr = trainer.line_search_lr(
                        probe_loss_fn, args.search_lr_min, args.search_lr_max,
                        n_points=args.search_n_points,
                        seed=99999 + search_count * 1000000,
                        n_seeds=args.search_n_seeds,
                        resample_batch_fn=resample_probe_batch,
                    )
                    trainer.lr = new_lr
                    if args.epsilon is None:
                        trainer.epsilon = new_lr

                search_last_loss = search_loss_ema
                search_best_ema = search_loss_ema
                search_stall_count = 0
                print(f"  New lr={trainer.lr:.2e}, eps={trainer.epsilon:.2e}")

    # Fast fail: abort if loss is exploding
    if train_loss_f > args.fail_threshold:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > args.warmup_steps:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    img_per_sec = int(args.total_batch_size / dt) if dt > 0 else 0
    mfu = 100 * num_flops_per_image * args.total_batch_size / dt / H100_BF16_PEAK_FLOPS if dt > 0 else 0
    remaining = max(0, args.time_budget - total_training_time)

    solver_tag = "TBPTT" if args.solver == "tbptt" else "SPSA"
    print(f"\rstep {step:05d} ({pct_done:.1f}%) [{solver_tag}] | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | img/sec: {img_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    wandb.log({"train/loss": debiased_smooth_loss, "train/raw_loss": train_loss_f,
               "train/lr_multiplier": lrm, "train/img_per_sec": img_per_sec,
               "train/mfu_pct": mfu, "train/epoch": epoch, "train/progress": progress}, step=step)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > args.warmup_steps and total_training_time >= args.time_budget:
        break

print()  # newline after \r training log

total_images = step * args.total_batch_size

# Final eval — DIRECT generation (no ODE solver, model produces images in one shot)
model.eval()

# Custom FID eval that calls model directly instead of through ODE solver
@torch.no_grad()
def evaluate_fid_direct(model, batch_size, num_classes=NUM_CLASSES,
                        num_samples=10000):
    """Generate images directly via TRM recursion (no ODE solver)."""
    from prepare import InceptionFeatureExtractor, compute_fid_from_stats, STATS_DIR, FID_NUM_SAMPLES
    import numpy as np

    ref_mu = np.load(os.path.join(STATS_DIR, "fid_mu.npy"))
    ref_sigma = np.load(os.path.join(STATS_DIR, "fid_sigma.npy"))

    device = next(model.parameters()).device
    extractor = InceptionFeatureExtractor(device=str(device))

    all_features = []
    n_generated = 0
    while n_generated < num_samples:
        current_batch = min(batch_size, num_samples - n_generated)
        labels = torch.randint(0, num_classes, (current_batch,), device=device)

        # Generate directly: noise → model → clean image (ONE call, no ODE)
        noise = torch.randn(current_batch, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
        t_dummy = torch.zeros(current_batch, device=device)
        velocity = model(noise, t_dummy, class_labels=labels)
        # During eval with one-shot: clean_pred = noise + velocity (when velocity scaling = *50)
        # But we changed to training/eval split... let me just extract clean_pred directly
        # Actually the model returns (clean_pred - x) * 50 at eval. So:
        # clean_pred = noise + velocity / 50... no that's wrong too.
        # Let me just call model in training mode to get clean_pred - noise:
        pass

        # Simplest: temporarily set model to training mode for velocity = clean - noise
        model.train()
        velocity = model(noise, t_dummy, class_labels=labels)
        model.eval()
        samples = noise + velocity  # = clean_pred

        samples = torch.clamp(samples, -1, 1)
        samples_01 = (samples + 1) / 2
        features = extractor.extract_features(samples_01)
        all_features.append(features)
        n_generated += current_batch
        print(f"\r  FID eval (direct): generated {n_generated}/{num_samples} samples", end="", flush=True)

    print()
    all_features = np.concatenate(all_features, axis=0)[:num_samples]
    gen_mu = np.mean(all_features, axis=0)
    gen_sigma = np.cov(all_features, rowvar=False)
    return compute_fid_from_stats(ref_mu, ref_sigma, gen_mu, gen_sigma)

val_fid = evaluate_fid_direct(model, args.device_batch_size)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_image * args.total_batch_size * max(0, step - args.warmup_steps) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"solver:           {args.solver}")
print(f"val_fid:          {val_fid:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_images_M:   {total_images / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"h_cycles:         {args.h_cycles}")
print(f"l_cycles:         {args.l_cycles}")
print(f"l_layers:         {args.l_layers}")
if args.solver == "spsa":
    print(f"denoising_steps:  {args.denoising_steps}")
    print(f"n_perts:          {args.n_perts}")
    print(f"final_lr:         {trainer.lr:.2e}")
    print(f"final_epsilon:    {trainer.epsilon:.2e}")

wandb.log({"eval/val_fid": val_fid, "eval/peak_vram_mb": peak_vram_mb,
           "eval/mfu_percent": steady_state_mfu, "eval/total_images_M": total_images / 1e6,
           "eval/num_steps": step, "eval/num_params_M": num_params / 1e6})
wandb.finish()
