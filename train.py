"""
Diffusion Transformer pretraining script. Single-GPU, single-file.
One-layer DiT with flow matching on ImageNet ILSVRC2012.
Supports backprop (teacher-forced) and 1.5-SPSA (zero-order, no teacher forcing).

Usage:
    uv run train.py                              # backprop (default)
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
import random
import argparse
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import numpy as np
from prepare import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES, TIME_BUDGET, FlowMatching, make_dataloader, evaluate_fid, InceptionFeatureExtractor

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train DiT with backprop or 1.5-SPSA zero-order optimization",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Solver selection
solver_group = parser.add_argument_group("solver")
solver_group.add_argument("--solver", type=str, default="backprop",
    choices=["backprop", "spsa"],
    help="Training solver: backprop (teacher-forced) or spsa (zero-order, no teacher forcing)")

# Model architecture
model_group = parser.add_argument_group("model architecture")
model_group.add_argument("--depth", type=int, default=1,
    help="Number of transformer layers")
model_group.add_argument("--n-embd", type=int, default=768,
    help="Model embedding dimension")
model_group.add_argument("--patch-size", type=int, default=4,
    help="Patch size for patch embedding")
model_group.add_argument("--head-dim", type=int, default=64,
    help="Target head dimension for attention")
model_group.add_argument("--repeat-blocks", type=int, default=1,
    help="Number of times to loop through transformer blocks (weight sharing). "
         "Effective depth = depth × repeat_blocks. Same params, more compute, deeper model.")
model_group.add_argument("--attn-type", type=str, default="softmax",
    choices=["softmax", "linear", "none"],
    help="Attention type. softmax=standard, linear=no softmax (SPSA-friendly), "
         "none=skip attention entirely (MLP-only, use token mixing)")
model_group.add_argument("--mlp-ratio", type=int, default=4,
    help="MLP hidden dim ratio (hidden = n_embd * mlp_ratio)")

# Training (common)
train_group = parser.add_argument_group("training")
train_group.add_argument("--total-batch-size", type=int, default=256,
    help="Total images per optimizer step")
train_group.add_argument("--device-batch-size", type=int, default=64,
    help="Per-device batch size (reduce if OOM)")
train_group.add_argument("--lr", type=float, default=1e-4,
    help="Learning rate")
train_group.add_argument("--weight-decay", type=float, default=0.0,
    help="Weight decay for transformer parameters (backprop AdamW)")
train_group.add_argument("--warmup-ratio", type=float, default=0.1,
    help="Fraction of time budget for LR warmup")
train_group.add_argument("--warmdown-ratio", type=float, default=0.3,
    help="Fraction of time budget for LR warmdown")
train_group.add_argument("--final-lr-frac", type=float, default=0.0,
    help="Final LR as fraction of initial LR")
train_group.add_argument("--lr-schedule", type=str, default="linear",
    choices=["linear", "cosine", "cosine_warmdown"],
    help="LR decay schedule type (linear warmdown or cosine)")
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

# Backprop-specific
bp_group = parser.add_argument_group("backprop-specific")
bp_group.add_argument("--adam-beta1", type=float, default=0.9,
    help="Adam beta1")
bp_group.add_argument("--adam-beta2", type=float, default=0.999,
    help="Adam beta2")
bp_group.add_argument("--grad-clip", type=float, default=0.0,
    help="Max gradient norm for clipping (0 = disabled)")
bp_group.add_argument("--ema-decay", type=float, default=0.0,
    help="EMA decay rate for model weights (0 = disabled, try 0.9999)")

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
spsa_group.add_argument("--t-schedule", type=str, default="fixed",
    choices=["fixed", "linear", "lognormal", "exponential", "curriculum", "curriculum_exp", "curriculum_step", "curriculum_mix", "reverse", "cyclic", "stochastic", "stochastic_pert", "phased", "curriculum_stoch", "curriculum_sawtooth", "adaptive", "curriculum_weighted"],
    help="T schedule: fixed, linear (ramp T_min->T_max), lognormal (sample), exponential (more time at low T), curriculum (T_min for 60%% then ramp), curriculum_step (T_min then jump to T_max), curriculum_mix (T_min then random T_min/T_max), reverse (T_max->T_min), cyclic (alternate T_min/T_max), stochastic (random T per step), stochastic_pert (random T per perturbation), phased (sequential T phases with LR restart per phase)")
spsa_group.add_argument("--t-min", type=int, default=2,
    help="Minimum T for linear schedule (start of training)")
spsa_group.add_argument("--t-max", type=int, default=50,
    help="Maximum T for linear schedule (end of training)")
spsa_group.add_argument("--t-lognormal-mu", type=float, default=2.0,
    help="Mu parameter for lognormal T sampling (log-space mean)")
spsa_group.add_argument("--t-lognormal-sigma", type=float, default=1.0,
    help="Sigma parameter for lognormal T sampling (log-space std)")
spsa_group.add_argument("--curriculum-frac", type=float, default=0.6,
    help="Fraction of training to stay at t_min before ramping to t_max (for curriculum schedule)")
spsa_group.add_argument("--spsa-loss-type", type=str, default="teacher",
    choices=["teacher", "denoising", "trajectory", "progressive", "inception", "minifid", "traj_div", "contrastive", "cosine", "huber", "combo", "rank", "mmd", "mmd_inception",
             "ssim", "fft", "multiscale", "ssim_mse", "direct_fid",
             "multi_step", "multi_step_exp",
             "loss_ensemble", "ssim_mse_fft", "hist_match",
             "denoising_midpoint", "denoising_rk4", "denoising_logmse",
             "denoising_multiscale", "denoising_heun", "denoising_cosine_steps",
             "denoising_warm_restart", "denoising_selfdistill",
             "ssim_mse_light", "mse_clamp2",
             "denoising_discrete", "denoising_multires", "denoising_huber",
             "denoising_lowres", "denoising_mae",
             "denoising_edge", "denoising_tv"],
    help="SPSA loss type for zero-order training")
spsa_group.add_argument("--loss-warmup-frac", type=float, default=0.0,
    help="Fraction of training to use denoising MSE before switching to --spsa-loss-type. "
         "E.g. 0.3 = use MSE for first 30%%, then switch to target loss (enables warm start for perceptual losses)")
spsa_group.add_argument("--vary-noise", action="store_true",
    help="Vary noise seed each step (prevents overfitting to one noise realization in fixed-batch mode)")
spsa_group.add_argument("--multi-noise", action="store_true",
    help="Use different noise seeds per perturbation within each step. Gradient averages over noise "
         "realizations AND parameter perturbations, reducing overfitting to one noise instance.")
spsa_group.add_argument("--adaptive-perts", action="store_true",
    help="Decay n_perts from initial value to 1/4 over training. More perturbations early (reliable "
         "gradients when model is random), fewer late (more steps for fine-grained updates).")
spsa_group.add_argument("--adaptive-perts-min-frac", type=float, default=0.25,
    help="Minimum fraction of n_perts to decay to (default: 0.25 = 1/4)")
spsa_group.add_argument("--sign-update", action="store_true",
    help="Use sign of gradient (signSGD) instead of raw gradient. More robust to outlier "
         "perturbations. Each parameter gets a unit-magnitude update in the estimated direction.")
spsa_group.add_argument("--augment-fixed", action="store_true",
    help="Apply random horizontal flip to fixed batch each step (adds diversity without new data)")
spsa_group.add_argument("--forward-fd", action="store_true",
    help="Use forward-difference SPSA instead of central-difference (1 eval per pert instead of 2, ~2x faster)")
spsa_group.add_argument("--eps-schedule", type=str, default="fixed",
    choices=["fixed", "cosine_decay", "linear_decay", "linear_warmup", "t_coupled"],
    help="Epsilon schedule: fixed (constant), cosine_decay (eps-max -> epsilon via cosine), "
         "linear_decay (eps-max -> epsilon linearly), linear_warmup (eps-min -> epsilon over warmup), "
         "t_coupled (scale epsilon by sqrt(T) during curriculum ramp — maintains gradient SNR as T increases)")
spsa_group.add_argument("--eps-max", type=float, default=None,
    help="Starting epsilon for decay schedules (decays to --epsilon). Default: 10x epsilon")
spsa_group.add_argument("--spsa-grad-clip", type=float, default=0.0,
    help="Clip SPSA gradient coefficient magnitude per perturbation (0 = disabled). "
         "Prevents divergence from outlier perturbations, enabling higher lr.")
spsa_group.add_argument("--spsa-weight-decay", type=float, default=0.0,
    help="Weight decay for SPSA parameter updates")
spsa_group.add_argument("--guided-pert", type=float, default=0.0,
    help="Fraction of perturbation from gradient EMA sign (0=pure random, 0.5=half guided). "
         "Biases perturbations toward known-good directions while maintaining exploration.")
spsa_group.add_argument("--antithetic", action="store_true",
    help="Use antithetic perturbation pairs: for each random direction, also use its complement. "
         "Reduces gradient variance. Halves unique directions but improves estimate quality.")
spsa_group.add_argument("--no-zero-init", action="store_true",
    help="Skip zero initialization of final layers (better for SPSA gradient signal)")
spsa_group.add_argument("--layerwise-spsa", action="store_true",
    help="Perturb one module at a time instead of all params (better gradient estimates)")
spsa_group.add_argument("--fixed-batch-size", type=int, default=0,
    help="If > 0, pre-load this many images for SPSA training")
spsa_group.add_argument("--fixed-batch-mode", type=str, default="cycle",
    choices=["cycle", "all"],
    help="cycle: train on 1 image at a time; all: train on all fixed images every step")
spsa_group.add_argument("--batch-refresh-pct", type=float, default=0.0,
    help="If > 0, refresh fixed batch every this fraction of training (e.g., 0.1 = every 10%%)")
spsa_group.add_argument("--batch-trickle-interval", type=int, default=0,
    help="If > 0, replace 1 random image in fixed batch every N steps. Gradual replacement "
         "avoids the distribution shift that causes full batch-refresh to diverge. "
         "E.g., 200 = replace 1/48 images every 200 steps, full turnover in ~9600 steps.")
spsa_group.add_argument("--loss-explosion-guard", action="store_true",
    help="Skip gradient updates when loss spikes >2x the running EMA. "
         "Prevents divergence from bad SPSA gradient estimates.")
spsa_group.add_argument("--checkpoint-rollback", action="store_true",
    help="Periodically checkpoint model weights. If loss diverges (>1.5x EMA), "
         "roll back to last good checkpoint and halve LR. Prevents catastrophic "
         "divergence that wastes entire training runs.")
spsa_group.add_argument("--swa-frac", type=float, default=0.0,
    help="If > 0, perform Stochastic Weight Averaging over the last swa_frac of training. "
         "E.g., 0.1 = average weights from 90%% to 100%% of training. Uses uniform average "
         "(not exponential). Good for reducing evaluation variance from SPSA noise.")
spsa_group.add_argument("--spsa-adam", action="store_true",
    help="Use Adam optimizer for SPSA gradient updates (momentum + adaptive LR)")
spsa_group.add_argument("--spsa-adam-beta1", type=float, default=0.9,
    help="Adam beta1 for SPSA momentum")
spsa_group.add_argument("--spsa-adam-beta2", type=float, default=0.999,
    help="Adam beta2 for SPSA second moment")
spsa_group.add_argument("--pert-recycle", type=int, default=1,
    help="Reuse same perturbation directions for N consecutive steps (common random numbers). "
         "N=1 is standard (fresh perts each step). N=5 means same 100 directions for 5 steps. "
         "Reduces gradient variance via temporal averaging while maintaining update frequency.")
spsa_group.add_argument("--median-clip", type=float, default=0.0,
    help="Clip SPSA gradient coefficients to ±k*median(|coeff|) across perturbations (0=disabled). "
         "Adaptive outlier removal: k=3 clips extreme perturbations without tuning absolute threshold.")
spsa_group.add_argument("--spsa-topk", type=float, default=0.0,
    help="Fraction of perturbations to use for gradient estimation (0=disabled, 0.5=top 50%%). "
         "Selects perturbations with largest |loss_diff|, discarding noisy low-signal ones. "
         "Free variance reduction: no extra forward passes, just better gradient aggregation.")
spsa_group.add_argument("--curriculum-polish", type=float, default=0.0,
    help="Fraction of training at end to spend polishing at T=t_min (0=disabled). "
         "After curriculum reaches t_max, drops T back to t_min for final refinement. "
         "E.g., 0.05 = last 5%% of training at T=1 for sharpening early denoising steps.")
spsa_group.add_argument("--elite-perts", type=int, default=0,
    help="Number of elite perturbation seeds to carry over from previous step (0=disabled). "
         "Tracks the K best perturbation directions (lowest loss) and reuses them next step, "
         "filling remaining slots with fresh random perturbations. Exploits promising directions.")
spsa_group.add_argument("--lr-layer-scale", action="store_true",
    help="Scale learning rate per layer: deeper layers get smaller LR updates. "
         "Linear decay from 2x (first layer) to 0.5x (last layer). "
         "Matches update magnitude to layer sensitivity in SPSA.")
spsa_group.add_argument("--sign-consensus", type=int, default=0,
    help="Only update parameters where gradient sign agrees over last K steps (0=disabled). "
         "Maintains a running sign buffer and masks out parameters with flipping signs. "
         "Reduces noise from unreliable SPSA gradient estimates at per-parameter level.")
spsa_group.add_argument("--freeze-pattern", type=str, default="",
    help="Comma-separated patterns of module names to freeze (exclude from SPSA perturbation). "
         "E.g., 'label_embed,time_embed' freezes conditioning layers. "
         "'blocks.0' freezes first transformer block. Empty=train all.")

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
# Diffusion Transformer Model
# ---------------------------------------------------------------------------

@dataclass
class DiTConfig:
    img_size: int = 64
    patch_size: int = 4
    in_channels: int = 3
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 1
    num_classes: int = 1000
    repeat_blocks: int = 1
    attn_type: str = "softmax"
    mlp_ratio: int = 4


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


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""
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

        # MLP
        mlp_hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, mlp_hidden, bias=False)
        self.c_fc_proj = nn.Linear(mlp_hidden, config.n_embd, bias=False)
        self.attn_type = config.attn_type

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
        if self.attn_type == "none":
            # Skip attention, just use linear projection as token mixing
            h = self.c_proj(self.c_v(h))
        else:
            q = self.c_q(h).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
            k = self.c_k(h).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
            v = self.c_v(h).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
            if self.attn_type == "linear":
                # Linear attention: no softmax, direct QK^TV
                # More SPSA-friendly: perturbations have linear effect on output
                scale = 1.0 / (N * self.head_dim ** 0.5)
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                h = torch.matmul(attn, v)
            else:
                h = F.scaled_dot_product_attention(q, k, v)
            h = h.transpose(1, 2).reshape(B, N, C)
            h = self.c_proj(h)
        x = x + gate_msa.unsqueeze(1) * h

        # MLP with AdaLN
        h = norm(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.c_fc(h)
        h = F.gelu(h)
        h = self.c_fc_proj(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiT(nn.Module):
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

        # Transformer blocks
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_layer)])

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

    @torch.no_grad()
    def init_weights(self, zero_init=True):
        # Standard Xavier init for linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Patch embedding
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # Label embedding
        nn.init.normal_(self.label_embed.embedding.weight, std=0.02)

        if zero_init:
            # Zero-out AdaLN modulation outputs so gates start at zero
            for block in self.blocks:
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)

            # Zero-out final projection (DiT zero-init)
            nn.init.zeros_(self.final_proj.weight)
            nn.init.zeros_(self.final_proj.bias)
            nn.init.zeros_(self.final_adaLN[-1].weight)
            nn.init.zeros_(self.final_adaLN[-1].bias)
        else:
            # Scale down output layers instead of zeroing (small initial output)
            for block in self.blocks:
                block.adaLN_modulation[-1].weight.data.mul_(0.01)
            self.final_proj.weight.data.mul_(0.01)
            self.final_adaLN[-1].weight.data.mul_(0.01)

    def estimate_flops(self):
        """Estimated FLOPs per image (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        num_patches = self.patch_embed.num_patches
        # Attention FLOPs: 2 * num_patches^2 * n_embd per layer (Q@K and attn@V)
        attn_flops = 0
        for _ in self.blocks:
            attn_flops += 4 * num_patches * num_patches * self.config.n_embd
        return 6 * nparams + attn_flops

    def num_scaling_params(self):
        patch_embed = sum(p.numel() for p in self.patch_embed.parameters())
        pos_embed = self.pos_embed.numel()
        time_embed = sum(p.numel() for p in self.time_embed.parameters())
        label_embed = sum(p.numel() for p in self.label_embed.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        final = sum(p.numel() for p in self.final_adaLN.parameters()) + \
                sum(p.numel() for p in self.final_proj.parameters())
        total = patch_embed + pos_embed + time_embed + label_embed + blocks + final
        return {
            'patch_embed': patch_embed, 'pos_embed': pos_embed,
            'time_embed': time_embed, 'label_embed': label_embed,
            'blocks': blocks, 'final': final, 'total': total,
        }

    def setup_optimizer(self, lr=1e-4, weight_decay=0.0, adam_betas=(0.9, 0.999)):
        # Separate param groups: embeddings (no decay) vs transformer weights (decay)
        embed_params = list(self.patch_embed.parameters()) + [self.pos_embed] + \
                       list(self.time_embed.parameters()) + list(self.label_embed.parameters())
        block_params = list(self.blocks.parameters()) + list(self.final_adaLN.parameters()) + \
                       list(self.final_proj.parameters())
        assert len(embed_params) + len(block_params) == len(list(self.parameters()))
        param_groups = [
            dict(params=embed_params, lr=lr, betas=adam_betas, weight_decay=0.0),
            dict(params=block_params, lr=lr, betas=adam_betas, weight_decay=weight_decay),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, x, t, class_labels=None):
        """
        x: (B, C, H, W) noisy images
        t: (B,) timesteps in [0, 1]
        class_labels: (B,) class indices
        Returns: (B, C, H, W) predicted velocity
        """
        # Patch embed + positional
        x = self.patch_embed(x) + self.pos_embed

        # Conditioning: time + class
        c = self.time_embed(t)
        if class_labels is not None:
            c = c + self.label_embed(class_labels)

        # Transformer blocks (optionally repeated for weight-shared depth)
        for _ in range(self.config.repeat_blocks):
            for block in self.blocks:
                x = block(x, c)

        # Output projection with AdaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_proj(x)
        x = self.unpatchify(x)

        return x

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
                 accum_steps, weight_decay, layerwise=False,
                 use_adam=False, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                 grad_clip=0.0, forward_fd=False, guided_pert=0.0, sign_update=False,
                 pert_recycle=1, median_clip=0.0, antithetic=False,
                 loss_explosion_guard=False, topk=0.0):
        self.lr = lr
        self.epsilon = epsilon
        self.n_perts = n_perts
        self.use_curvature = use_curvature
        self.saturating_alpha = saturating_alpha
        self.lambda_reg = lambda_reg
        self.memory_efficient = memory_efficient
        self.accum_steps = accum_steps
        self.weight_decay = weight_decay
        self.layerwise = layerwise
        self.use_adam = use_adam
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.adam_step = 0
        self.grad_clip = grad_clip
        self.forward_fd = forward_fd
        self.guided_pert = guided_pert
        self.pert_recycle = pert_recycle
        self.median_clip = median_clip
        self.sign_update = sign_update
        self.antithetic = antithetic
        self.loss_explosion_guard = loss_explosion_guard
        self.topk = topk  # fraction of perturbations to keep (0 = all)
        self.elite_perts = 0  # set from args after construction
        self._elite_seeds = []  # seeds of best perturbations from last step
        self.sign_consensus = 0  # set from args after construction
        self._sign_history = None  # ring buffer of gradient signs
        self.lr_layer_scale = False  # set from args after construction
        self._loss_ema = None  # EMA of loss for explosion detection

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.total = sum(p.numel() for p in self.params)
        self.packed_size = (self.total + 7) // 8

        # For layerwise SPSA: group params by module
        if layerwise:
            self.param_groups = []
            for name, module in model.named_modules():
                module_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
                if module_params:
                    total = sum(p.numel() for p in module_params)
                    self.param_groups.append({
                        'name': name,
                        'params': module_params,
                        'total': total,
                        'packed_size': (total + 7) // 8,
                    })
            self.current_group = 0
            print(f"  Layerwise SPSA: {len(self.param_groups)} parameter groups")
            for g in self.param_groups:
                print(f"    {g['name']}: {g['total']:,} params")

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

        # Adam state for momentum-based SPSA
        if use_adam and not memory_efficient:
            self.m = [torch.zeros(info['numel'], device='cuda', dtype=torch.float32)
                      for info in self.param_info]
            self.v = [torch.zeros(info['numel'], device='cuda', dtype=torch.float32)
                      for info in self.param_info]
        else:
            self.m = None
            self.v = None

        # Guided perturbations: maintain gradient EMA sign for biasing perturbation directions
        if guided_pert > 0 and not memory_efficient:
            # Store sign of EMA gradient as packed bits (same format as perturbations)
            self.grad_ema_packed = torch.zeros(self.packed_size, device='cuda', dtype=torch.uint8)
            self.grad_ema_initialized = False
        else:
            self.grad_ema_packed = None

        mode_str = " [MEMORY EFFICIENT]" if memory_efficient else ""
        accum_str = f", accum={accum_steps}" if accum_steps > 1 else ""
        curv_str = " [1.5-SPSA]" if use_curvature else ""
        adam_str = " [ADAM]" if use_adam else ""
        guided_str = f" [GUIDED {guided_pert:.0%}]" if guided_pert > 0 else ""
        print(f"SPSATrainer: {self.total/1e6:.2f}M params, packed={self.packed_size/1e6:.1f}MB"
              f"{curv_str}{mode_str}{accum_str}{adam_str}{guided_str}")

    def step(self, loss_fn, iteration, per_pert_hook=None):
        """Perform one SPSA step. loss_fn(batch_idx) -> float.
        per_pert_hook(pert_idx, iteration): optional callback before each perturbation pair."""
        if self.layerwise:
            return self._step_layerwise(loss_fn, iteration)
        if self.memory_efficient:
            return self._step_memory_efficient(loss_fn, iteration)

        for g in self.grads:
            g.zero_()

        total_loss = 0.0

        # For forward-FD or 1.5-SPSA, get clean loss once per iteration
        if self.use_curvature or self.forward_fd:
            loss_clean = 0.0
            for batch_idx in range(self.accum_steps):
                loss_clean += loss_fn(batch_idx)
            loss_clean /= self.accum_steps

        # Track per-perturbation loss for elite selection
        if self.elite_perts > 0:
            _pert_losses = []  # (loss_avg, seed) pairs for elite tracking
            self._elite_seeds_snapshot = list(self._elite_seeds)  # snapshot for deferred replay

        for pert_idx in range(self.n_perts):
            # Per-perturbation hook (e.g., set T for this perturbation)
            if per_pert_hook is not None:
                per_pert_hook(pert_idx, iteration)

            # Generate bit-packed Rademacher random (8x less memory than full float)
            # With pert_recycle > 1, reuse same perturbation directions across consecutive steps
            seed_iter = iteration // self.pert_recycle if self.pert_recycle > 1 else iteration
            # Elite perturbations: reuse best seeds from previous step
            if self.elite_perts > 0 and pert_idx < len(self._elite_seeds):
                pert_seed = self._elite_seeds[pert_idx]
                is_antithetic = False  # elite seeds are always "positive" direction
            elif self.antithetic and pert_idx >= self.n_perts // 2:
                base_idx = pert_idx - self.n_perts // 2
                pert_seed = seed_iter * 10000 + base_idx
                is_antithetic = True
            else:
                pert_seed = seed_iter * 10000 + pert_idx
                is_antithetic = False

            torch.manual_seed(pert_seed)
            packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
            if is_antithetic:
                packed = ~packed  # bitwise complement = opposite direction

            # Guided perturbations: bias toward gradient EMA sign direction
            if self.guided_pert > 0 and self.grad_ema_packed is not None and self.grad_ema_initialized:
                # Generate per-byte guidance mask: each bit independently uses EMA with prob guided_pert
                # Efficient approach: use threshold on random bytes
                guide_rand = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
                threshold = int(self.guided_pert * 256)
                # For each bit position, if the corresponding random value < threshold, use EMA
                # Simple approximation: if random byte < threshold*256/256, replace entire byte
                guide_mask = (guide_rand < threshold)
                packed = torch.where(guide_mask, self.grad_ema_packed, packed)

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

            if self.forward_fd:
                # Forward-difference: use (L+ - L0) / epsilon
                # Only need to undo +epsilon (not -2*epsilon)
                for info in self.param_info:
                    flat = info['param'].data.view(-1)
                    _unpack_and_apply[info['grid']](
                        flat, packed[info['packed_offset']:],
                        info['numel'], -self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)
                grad_coeff = (loss_plus - loss_clean) / (self.epsilon * self.n_perts)
            else:
                # Central difference: use (L+ - L-) / (2*epsilon)
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

            # Clip gradient coefficient to prevent outlier perturbations from diverging
            if self.grad_clip > 0:
                grad_coeff = max(-self.grad_clip, min(self.grad_clip, grad_coeff))

            if self.median_clip > 0 or self.topk > 0:
                # Deferred accumulation: store coefficients for median clipping or top-K later
                if not hasattr(self, '_deferred_coeffs'):
                    self._deferred_coeffs = []
                self._deferred_coeffs.append(grad_coeff)
            else:
                # Accumulate gradient immediately
                for i, info in enumerate(self.param_info):
                    _unpack_and_accumulate[info['grid']](
                        self.grads[i], packed[info['packed_offset']:],
                        info['numel'], grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            if self.forward_fd:
                total_loss += loss_plus
                pert_avg_loss = loss_plus
            else:
                total_loss += (loss_plus + loss_minus) / 2
                pert_avg_loss = (loss_plus + loss_minus) / 2

            # Track per-perturbation loss for elite selection
            if self.elite_perts > 0:
                _pert_losses.append((pert_avg_loss, pert_seed))

            del packed

        # Update elite seeds: keep the K perturbations with lowest average loss
        if self.elite_perts > 0 and _pert_losses:
            _pert_losses.sort(key=lambda x: x[0])
            self._elite_seeds = [seed for _, seed in _pert_losses[:self.elite_perts]]

        # Deferred gradient accumulation: replay perturbations with processed coefficients
        if (self.median_clip > 0 or self.topk > 0) and hasattr(self, '_deferred_coeffs') and self._deferred_coeffs:
            coeffs = self._deferred_coeffs
            abs_coeffs = [abs(c) for c in coeffs]

            # Apply top-K filtering: zero out perturbations with smallest |loss_diff|
            if self.topk > 0:
                k = max(1, int(len(coeffs) * self.topk))
                abs_sorted = sorted(abs_coeffs, reverse=True)
                threshold = abs_sorted[min(k, len(abs_sorted)) - 1]
                # Rescale kept coefficients: multiply by n_perts/k to correct gradient magnitude
                scale = len(coeffs) / k
                coeffs = [c * scale if abs(c) >= threshold else 0.0 for c in coeffs]

            # Apply median clipping on top of top-K (if both are set)
            if self.median_clip > 0:
                live_abs = sorted([abs(c) for c in coeffs if c != 0.0])
                if live_abs:
                    median_val = live_abs[len(live_abs) // 2]
                    clip_val = self.median_clip * median_val
                    coeffs = [max(-clip_val, min(clip_val, c)) for c in coeffs]

            # Second pass: regenerate perturbations and accumulate with processed coefficients
            for pert_idx, coeff in enumerate(coeffs):
                if coeff == 0.0:
                    continue  # Skip zeroed-out perturbations (top-K filtered)
                seed_iter = iteration // self.pert_recycle if self.pert_recycle > 1 else iteration
                # Reconstruct same seed logic as the forward pass
                _replay_elite = getattr(self, '_elite_seeds_snapshot', [])
                if len(_replay_elite) > 0 and pert_idx < len(_replay_elite):
                    pert_seed = _replay_elite[pert_idx]
                    is_antithetic = False
                elif self.antithetic and pert_idx >= self.n_perts // 2:
                    base_idx = pert_idx - self.n_perts // 2
                    pert_seed = seed_iter * 10000 + base_idx
                    is_antithetic = True
                else:
                    pert_seed = seed_iter * 10000 + pert_idx
                    is_antithetic = False
                torch.manual_seed(pert_seed)
                packed = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
                if is_antithetic:
                    packed = ~packed
                if self.guided_pert > 0 and self.grad_ema_packed is not None and self.grad_ema_initialized:
                    guide_rand = torch.randint(0, 256, (self.packed_size,), device='cuda', dtype=torch.uint8)
                    threshold = int(self.guided_pert * 256)
                    guide_mask = (guide_rand < threshold)
                    packed = torch.where(guide_mask, self.grad_ema_packed, packed)
                for i, info in enumerate(self.param_info):
                    _unpack_and_accumulate[info['grid']](
                        self.grads[i], packed[info['packed_offset']:],
                        info['numel'], coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)
                del packed
            self._deferred_coeffs = []

        # Loss explosion guard: if loss spikes >2x the EMA, skip this update
        avg_loss = total_loss / self.n_perts
        if self.loss_explosion_guard:
            if self._loss_ema is None:
                self._loss_ema = avg_loss
            else:
                if avg_loss > 2.0 * self._loss_ema:
                    # Loss exploded — skip gradient update, restore EMA slowly
                    self._loss_ema = 0.95 * self._loss_ema + 0.05 * avg_loss
                    return avg_loss  # Return loss but don't update parameters
                self._loss_ema = 0.99 * self._loss_ema + 0.01 * avg_loss

        # Sign consensus: mask out parameters with flipping gradient signs
        if self.sign_consensus > 0 and self.grads is not None:
            K = self.sign_consensus
            if self._sign_history is None:
                # Initialize: list of K sign tensors per param group, index into ring buffer
                self._sign_history = [[torch.zeros(info['numel'], device='cuda', dtype=torch.int8)
                                       for _ in range(K)] for info in self.param_info]
                self._sign_idx = 0
            # Record current gradient sign
            for i, grad in enumerate(self.grads):
                self._sign_history[i][self._sign_idx % K] = grad.sign().to(torch.int8)
            self._sign_idx += 1
            # After K steps of history, apply consensus mask
            if self._sign_idx >= K:
                for i, grad in enumerate(self.grads):
                    # Sum signs over K steps: +K means all positive, -K means all negative
                    sign_sum = sum(self._sign_history[i]).float()
                    # Mask: only keep gradient where |sign_sum| > 0.6*K (60% agreement)
                    mask = (sign_sum.abs() > 0.6 * K).to(grad.dtype)
                    self.grads[i] = grad * mask

        # Apply update
        if self.use_adam and self.m is not None:
            self.adam_step += 1
            for i, (info, grad) in enumerate(zip(self.param_info, self.grads)):
                g = grad.float()
                self.m[i].mul_(self.adam_beta1).add_(g, alpha=1 - self.adam_beta1)
                if self.adam_beta2 > 0:
                    # Full Adam: momentum + adaptive LR
                    self.v[i].mul_(self.adam_beta2).addcmul_(g, g, value=1 - self.adam_beta2)
                    m_hat = self.m[i] / (1 - self.adam_beta1 ** self.adam_step)
                    v_hat = self.v[i] / (1 - self.adam_beta2 ** self.adam_step)
                    update = m_hat / (v_hat.sqrt() + self.adam_eps)
                else:
                    # Momentum-only SGD: just EMA of gradients, no normalization
                    update = self.m[i] / (1 - self.adam_beta1 ** self.adam_step)
                if self.weight_decay > 0:
                    update.add_(info['param'].data.view(-1).float(), alpha=self.weight_decay)
                lr_i = self.lr * info.get('lr_scale', 1.0) if self.lr_layer_scale else self.lr
                info['param'].data.view(-1).sub_(update, alpha=lr_i)
        else:
            # Plain SGD (optionally with sign update)
            for info, grad in zip(self.param_info, self.grads):
                lr_i = self.lr * info.get('lr_scale', 1.0) if self.lr_layer_scale else self.lr
                if self.sign_update:
                    info['param'].data.view(-1).sub_(grad.sign(), alpha=lr_i)
                else:
                    info['param'].data.view(-1).sub_(grad, alpha=lr_i)
            if self.weight_decay > 0:
                for info in self.param_info:
                    info['param'].data.mul_(1 - self.lr * self.weight_decay)

        # Update gradient EMA packed bits for guided perturbations
        if self.guided_pert > 0 and self.grad_ema_packed is not None:
            # Use Adam momentum (m) if available, else raw gradient
            grad_source = self.m if (self.use_adam and self.m is not None) else self.grads
            if grad_source is not None:
                # Pack sign of gradient EMA into bytes
                all_signs = []
                for g in grad_source:
                    all_signs.append((g >= 0).to(torch.uint8))
                sign_bits = torch.cat(all_signs)
                # Pad to multiple of 8
                n_full = (len(sign_bits) // 8) * 8
                if n_full > 0:
                    sign_bytes = sign_bits[:n_full].view(-1, 8)
                    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device='cuda', dtype=torch.uint8)
                    self.grad_ema_packed[:n_full // 8] = (sign_bytes * powers).sum(dim=1).to(torch.uint8)
                self.grad_ema_initialized = True

        return total_loss / self.n_perts

    def _step_layerwise(self, loss_fn, iteration):
        """Layerwise SPSA: perturb one parameter group at a time, rotate each step."""
        group = self.param_groups[self.current_group % len(self.param_groups)]
        self.current_group += 1

        # Build param_info for this group only
        group_info = []
        offset = 0
        for p in group['params']:
            numel = p.numel()
            group_info.append({
                'param': p,
                'offset': offset,
                'packed_offset': offset // 8,
                'numel': numel,
                'grid': ((numel + TRITON_BLOCK_SIZE - 1) // TRITON_BLOCK_SIZE,),
            })
            offset += numel

        packed_size = group['packed_size']
        grads = [torch.zeros(info['numel'], device='cuda', dtype=torch.bfloat16) for info in group_info]
        total_loss = 0.0

        if self.use_curvature:
            loss_clean = 0.0
            for batch_idx in range(self.accum_steps):
                loss_clean += loss_fn(batch_idx)
            loss_clean /= self.accum_steps

        for pert_idx in range(self.n_perts):
            torch.manual_seed(iteration * 10000 + pert_idx)
            packed = torch.randint(0, 256, (packed_size,), device='cuda', dtype=torch.uint8)

            for info in group_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_plus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_plus += loss_fn(batch_idx)
            loss_plus /= self.accum_steps

            for info in group_info:
                flat = info['param'].data.view(-1)
                _unpack_and_apply[info['grid']](
                    flat, packed[info['packed_offset']:],
                    info['numel'], -2 * self.epsilon, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            loss_minus = 0.0
            for batch_idx in range(self.accum_steps):
                loss_minus += loss_fn(batch_idx)
            loss_minus /= self.accum_steps

            for info in group_info:
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

            for i, info in enumerate(group_info):
                _unpack_and_accumulate[info['grid']](
                    grads[i], packed[info['packed_offset']:],
                    info['numel'], grad_coeff, BLOCK_SIZE=TRITON_BLOCK_SIZE)

            total_loss += (loss_plus + loss_minus) / 2
            del packed

        for info, grad in zip(group_info, grads):
            info['param'].data.view(-1).sub_(grad, alpha=self.lr)

        if self.weight_decay > 0:
            for info in group_info:
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
config = DiTConfig(
    img_size=IMG_SIZE, patch_size=args.patch_size, in_channels=IMG_CHANNELS,
    n_embd=args.n_embd, n_head=num_heads, n_layer=args.depth, num_classes=NUM_CLASSES,
    repeat_blocks=args.repeat_blocks,
    attn_type=args.attn_type,
    mlp_ratio=args.mlp_ratio,
)
print(f"Solver: {args.solver}")
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = DiT(config)
model.to_empty(device=device)
zero_init = not getattr(args, 'no_zero_init', False)
model.init_weights(zero_init=zero_init)
if not zero_init:
    print("Using non-zero init (small random outputs)")

# Note: no-zero-init is already handled by model.init_weights(zero_init=False)
# which scales output layers by 0.01x instead of zeroing them

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
if args.solver == "backprop":
    optimizer = model.setup_optimizer(
        lr=args.lr, weight_decay=args.weight_decay,
        adam_betas=(args.adam_beta1, args.adam_beta2),
    )
    model = torch.compile(model, dynamic=False)
    # EMA setup
    ema_model = None
    if args.ema_decay > 0:
        import copy
        ema_model = copy.deepcopy(model._orig_mod if hasattr(model, '_orig_mod') else model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
else:
    # SPSA: no backprop, no torch.compile (in-place weight perturbation)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    # Freeze specified modules (reduce SPSA dimensionality)
    if args.freeze_pattern:
        freeze_patterns = [pat.strip() for pat in args.freeze_pattern.split(",")]
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(pat in name for pat in freeze_patterns):
                param.requires_grad = False
                frozen_count += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen {frozen_count:,} params ({len(freeze_patterns)} patterns). Trainable: {trainable:,}")
    trainer = SPSATrainer(
        model=model, lr=args.lr, epsilon=SPSA_EPSILON,
        n_perts=args.n_perts, use_curvature=args.use_curvature,
        saturating_alpha=args.saturating_alpha, lambda_reg=args.lambda_reg,
        memory_efficient=args.memory_efficient,
        accum_steps=args.spsa_accum_steps, weight_decay=args.spsa_weight_decay,
        layerwise=args.layerwise_spsa,
        use_adam=args.spsa_adam, adam_beta1=args.spsa_adam_beta1,
        adam_beta2=args.spsa_adam_beta2,
        grad_clip=args.spsa_grad_clip,
        forward_fd=args.forward_fd,
        guided_pert=args.guided_pert,
        sign_update=args.sign_update,
        pert_recycle=args.pert_recycle,
        median_clip=args.median_clip,
        antithetic=args.antithetic,
        loss_explosion_guard=args.loss_explosion_guard,
        topk=args.spsa_topk,
    )
    if args.elite_perts > 0:
        trainer.elite_perts = args.elite_perts
        print(f"  Elite perturbations: {args.elite_perts} seeds carried over each step")
    if args.sign_consensus > 0:
        trainer.sign_consensus = args.sign_consensus
        print(f"  Sign consensus: only update params with consistent sign over {args.sign_consensus} steps")
    if args.lr_layer_scale:
        trainer.lr_layer_scale = True
        # Compute per-param LR scale: linearly from 2.0 (first param) to 0.5 (last param)
        n_params = len(trainer.param_info)
        for idx, info in enumerate(trainer.param_info):
            frac = idx / max(n_params - 1, 1)  # 0 to 1
            info['lr_scale'] = 2.0 - 1.5 * frac  # 2.0 -> 0.5
        scales = [info['lr_scale'] for info in trainer.param_info]
        print(f"  Layer-wise LR: scale {scales[0]:.2f}x -> {scales[-1]:.2f}x across {n_params} params")

train_loader = make_dataloader("train", args.device_batch_size)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {args.time_budget}s")
if args.solver == "backprop":
    print(f"Gradient accumulation steps: {grad_accum_steps}")
else:
    print(f"SPSA: n_perts={args.n_perts}, epsilon={SPSA_EPSILON:.2e}, denoising_steps={args.denoising_steps}")
    print(f"SPSA: accum_steps={args.spsa_accum_steps}, search={args.search_strategy}")
    if args.use_curvature:
        print(f"SPSA: 1.5-SPSA curvature: alpha={args.saturating_alpha}, lambda={args.lambda_reg}")

# Schedules (all based on progress = training_time / time_budget)

def get_lr_multiplier(progress):
    if args.lr_schedule == "cosine":
        # Cosine schedule: warmup then cosine decay to final_lr_frac
        if progress < args.warmup_ratio:
            return progress / args.warmup_ratio if args.warmup_ratio > 0 else 1.0
        else:
            decay_progress = (progress - args.warmup_ratio) / (1.0 - args.warmup_ratio)
            return args.final_lr_frac + 0.5 * (1.0 - args.final_lr_frac) * (1.0 + math.cos(math.pi * decay_progress))
    elif args.lr_schedule == "cosine_warmdown":
        # Linear warmup + constant + COSINE warmdown (spends more time near peak LR)
        if progress < args.warmup_ratio:
            return progress / args.warmup_ratio if args.warmup_ratio > 0 else 1.0
        elif progress < 1.0 - args.warmdown_ratio:
            return 1.0
        else:
            # Cosine decay from 1.0 to final_lr_frac
            wd_progress = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
            return args.final_lr_frac + 0.5 * (1.0 - args.final_lr_frac) * (1.0 + math.cos(math.pi * wd_progress))
    else:
        # Linear warmup + constant + linear warmdown
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
    # Dynamic T: set per step, used by loss function (same T for ±epsilon)
    current_T = [args.denoising_steps]
    # Pre-generated random state for deterministic teacher loss across ±epsilon
    spsa_teacher_t = [None]       # timesteps (same for all perturbation evals)
    spsa_teacher_noise = [None]   # noise (same for all perturbation evals)

    # Inception feature matching setup
    spsa_inception = [None]
    spsa_ref_mu = [None]
    spsa_ref_sigma = [None]
    if args.spsa_loss_type in ("inception", "minifid", "mmd_inception", "direct_fid"):
        import os as _os
        STATS_DIR = _os.path.join(_os.path.expanduser("~"), ".cache", "autoresearch", "stats")
        spsa_inception[0] = InceptionFeatureExtractor(device=str(device))
        spsa_ref_mu[0] = np.load(_os.path.join(STATS_DIR, "fid_mu.npy"))
        spsa_ref_sigma[0] = np.load(_os.path.join(STATS_DIR, "fid_sigma.npy"))
        print(f"Inception feature matching: loaded reference stats (dim={spsa_ref_mu[0].shape[0]})")

    # Track which loss type is active (for logging during warmup phase)
    active_loss_type = [args.spsa_loss_type]

    # Self-distillation targets: computed once per step at unperturbed model
    selfdistill_targets = [None]  # Will hold target images from T=20 generation

    def spsa_loss_fn(batch_idx=0):
        """SPSA training loss with multiple loss type options."""
        T = current_T[0]  # Dynamic T, set per training step
        x_b, y_b = spsa_batches[batch_idx % len(spsa_batches)]

        # Loss warmup: use denoising MSE for first loss_warmup_frac of training
        # This provides strong gradient signal when model output is noise,
        # then switches to the target (possibly non-differentiable) loss
        loss_type = args.spsa_loss_type
        if args.loss_warmup_frac > 0:
            progress_now = min(total_training_time / args.time_budget, 1.0)
            if progress_now < args.loss_warmup_frac:
                loss_type = "denoising"  # warm start with MSE
            active_loss_type[0] = loss_type

        with torch.no_grad(), autocast_ctx:
            if loss_type == "teacher":
                # Use pre-generated t and noise for determinism across ±eps
                t = spsa_teacher_t[0]
                noise = spsa_teacher_noise[0]
                sigma_min = flow_matching.sigma_min
                t_exp = t.float().view(-1, 1, 1, 1)
                x_t = t_exp * x_b + (1 - (1 - sigma_min) * t_exp) * noise
                velocity_target = x_b - (1 - sigma_min) * noise
                predicted = model(x_t, t, class_labels=y_b)
                return F.mse_loss(predicted, velocity_target).item()
            elif loss_type == "denoising":
                return flow_matching.denoising_loss(
                    model, x_b, class_labels=y_b,
                    denoising_steps=T,
                    noise_seed=noise_seed[0] + batch_idx,
                )
            elif loss_type == "denoising_midpoint":
                # Midpoint method (2nd order) ODE solver for denoising
                # Only possible with zero-order: doesn't need differentiability!
                # Better ODE integration = more accurate loss = less noisy SPSA gradient
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    # k1 = velocity at current point
                    k1 = model(x, t_tensor, class_labels=y_b)
                    # Midpoint: evaluate at t + dt/2 with x + k1*dt/2
                    t_mid = torch.full((x_b.shape[0],), min(t_val + dt/2, 1.0), device=dev)
                    x_mid = x + k1 * (dt / 2)
                    k2 = model(x_mid, t_mid, class_labels=y_b)
                    # Update using midpoint velocity
                    x = x + k2 * dt
                return F.mse_loss(x, x_b).item()
            elif loss_type == "denoising_rk4":
                # 4th-order Runge-Kutta ODE solver for denoising
                # Dramatically better integration accuracy, only possible with zero-order
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    t_mid = torch.full((x_b.shape[0],), min(t_val + dt/2, 1.0), device=dev)
                    t_end = torch.full((x_b.shape[0],), min(t_val + dt, 1.0), device=dev)
                    k1 = model(x, t_tensor, class_labels=y_b)
                    k2 = model(x + k1 * (dt/2), t_mid, class_labels=y_b)
                    k3 = model(x + k2 * (dt/2), t_mid, class_labels=y_b)
                    k4 = model(x + k3 * dt, t_end, class_labels=y_b)
                    x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
                return F.mse_loss(x, x_b).item()
            elif loss_type == "denoising_logmse":
                # Log-MSE loss: log(MSE) gives larger gradient signal when loss is small
                # Changes the SPSA gradient landscape - gradient magnitude doesn't decay with loss
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                mse = F.mse_loss(x, x_b).item()
                import math
                return math.log(mse + 1e-8)
            elif loss_type == "denoising_multiscale":
                # Multi-scale MSE: compute MSE at original + downsampled resolutions
                # Captures both fine detail and global structure
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # Scale 1: full resolution
                loss_full = F.mse_loss(x, x_b).item()
                # Scale 2: 2x downsampled
                x_down2 = F.avg_pool2d(x, 2)
                target_down2 = F.avg_pool2d(x_b, 2)
                loss_2x = F.mse_loss(x_down2, target_down2).item()
                # Scale 3: 4x downsampled
                x_down4 = F.avg_pool2d(x, 4)
                target_down4 = F.avg_pool2d(x_b, 4)
                loss_4x = F.mse_loss(x_down4, target_down4).item()
                # Weighted combination (more weight on fine detail)
                return 0.5 * loss_full + 0.3 * loss_2x + 0.2 * loss_4x
            elif loss_type == "denoising_heun":
                # Heun's method (improved Euler / trapezoidal rule)
                # 2nd order like midpoint but uses trapezoidal average of endpoints
                # Formula: x_{n+1} = x_n + dt/2 * (k1 + k2)
                # where k1 = f(t_n, x_n), k2 = f(t_{n+1}, x_n + dt*k1)
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    k1 = model(x, t_tensor, class_labels=y_b)
                    # Predictor: Euler step to get x_predicted
                    x_pred = x + k1 * dt
                    t_end = torch.full((x_b.shape[0],), min(t_val + dt, 1.0), device=dev)
                    k2 = model(x_pred, t_end, class_labels=y_b)
                    # Corrector: trapezoidal average
                    x = x + (k1 + k2) * (dt / 2)
                return F.mse_loss(x, x_b).item()
            elif loss_type == "denoising_cosine_steps":
                # Cosine-spaced ODE timesteps: cluster steps near t=0 and t=1
                # where dynamics change fastest. Better integration for same # of steps.
                import math as _math
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                # Generate cosine-spaced timesteps from 0 to 1
                # t_i = 0.5 * (1 - cos(pi * i / T)) maps [0,T] -> [0,1] with clustering at endpoints
                t_points = [0.5 * (1.0 - _math.cos(_math.pi * i / T)) for i in range(T + 1)]
                for i in range(T):
                    t_val = t_points[i]
                    dt_i = t_points[i + 1] - t_points[i]
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt_i
                return F.mse_loss(x, x_b).item()
            elif loss_type == "denoising_warm_restart":
                # Warm restart denoising: run ODE, compute MSE, then use the
                # current prediction as a better starting point and run again.
                # 2 full ODE passes: first rough, second refined. Only possible
                # with zero-order since we don't need to backprop through both passes.
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                # First pass: standard Euler
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # x is now the first-pass prediction
                # Second pass: blend prediction with noise and re-denoise
                # Start from 50% noise + 50% prediction (like starting from t=0.5)
                x_blend = 0.5 * noise + 0.5 * x
                half_T = max(T // 2, 1)
                dt2 = 0.5 / half_T  # Only integrate from t=0.5 to t=1.0
                for i in range(half_T):
                    t_val = 0.5 + i * dt2
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x_blend, t_tensor, class_labels=y_b)
                    x_blend = x_blend + velocity * dt2
                # Loss is MSE of the refined prediction
                return F.mse_loss(x_blend, x_b).item()
            elif loss_type == "denoising_selfdistill":
                # Self-distillation: combine standard denoising loss with a
                # consistency loss that teaches the model to reproduce its own
                # high-T generation at lower T. Targets computed ONCE at unperturbed
                # model before perturbations, so they're deterministic for ±eps.
                # Standard denoising component
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                mse_real = F.mse_loss(x, x_b).item()
                # Self-distillation component (if targets available)
                if selfdistill_targets[0] is not None:
                    mse_distill = F.mse_loss(x, selfdistill_targets[0]).item()
                    # Blend: 70% real target, 30% self-distillation
                    return 0.7 * mse_real + 0.3 * mse_distill
                return mse_real
            elif loss_type in ("multi_step", "multi_step_exp"):
                # Multi-step denoising: sum MSE(x_t, x_clean) at each ODE step
                # multi_step: uniform weighting
                # multi_step_exp: exponentially increasing weight toward later steps
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                total_loss = 0.0
                total_weight = 0.0
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                    step_loss = F.mse_loss(x, x_b).item()
                    if loss_type == "multi_step_exp":
                        # Exponential weight: 2^(i/T) — later steps weighted more
                        w = 2.0 ** (i / T)
                    else:
                        w = 1.0
                    total_loss += w * step_loss
                    total_weight += w
                return total_loss / total_weight
            elif loss_type == "trajectory":
                # Per-step velocity matching: compare predicted velocity to
                # flow matching target at each ODE step. Gives T signals.
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                sigma_min = flow_matching.sigma_min
                v_target = x_b - (1 - sigma_min) * noise
                x = noise.clone()
                dt = 1.0 / T
                total_loss = 0.0
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    total_loss += F.mse_loss(velocity, v_target).item()
                    x = x + velocity * dt
                return total_loss / T
            elif loss_type == "progressive":
                # Per-step state matching: compare ODE state to ideal
                # intermediate x_target(t) = t*x0 + (1-(1-σ)*t)*noise
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                sigma_min = flow_matching.sigma_min
                x = noise.clone()
                dt = 1.0 / T
                total_loss = 0.0
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                    # Target state at t_{i+1}
                    t_next = (i + 1) / T
                    x_target = t_next * x_b + (1 - (1 - sigma_min) * t_next) * noise
                    total_loss += F.mse_loss(x, x_target).item()
                return total_loss / T
            elif loss_type == "inception":
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    samples = torch.clamp(x.float(), -1, 1)
                    samples_01 = (samples + 1) / 2
                    gen_features = spsa_inception[0].extract_features(samples_01)
                gen_mu = np.mean(gen_features, axis=0)
                # Normalize by feature dim to keep loss in reasonable range
                return float(np.mean((gen_mu - spsa_ref_mu[0]) ** 2))
            elif loss_type == "minifid":
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt_ode = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt_ode
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    samples = torch.clamp(x.float(), -1, 1)
                    samples_01 = (samples + 1) / 2
                    gen_features = spsa_inception[0].extract_features(samples_01)
                gen_mu = np.mean(gen_features, axis=0)
                diff = gen_mu - spsa_ref_mu[0]
                return float(np.mean(diff * diff))
            elif loss_type == "traj_div":
                # Trajectory loss with diversity penalty
                # Penalizes generated images for being too similar (anti-collapse)
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                sigma_min = flow_matching.sigma_min
                v_target = x_b - (1 - sigma_min) * noise
                x = noise.clone()
                dt_ode = 1.0 / T
                total_loss = 0.0
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    total_loss += F.mse_loss(velocity, v_target).item()
                    x = x + velocity * dt_ode
                traj_loss = total_loss / T
                # Diversity: std of generated images across batch (higher = more diverse)
                batch_std = x.std(dim=0).mean().item()
                # We want high diversity, so subtract it from loss
                return traj_loss - 0.1 * batch_std
            elif loss_type == "contrastive":
                # Contrastive loss: each generated image should be closer to its
                # own target than to other targets. Mean prediction fails this
                # because the mean is equidistant from all targets.
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # Flatten to (B, D) for pairwise distances
                B = x.shape[0]
                x_flat = x.reshape(B, -1).float()
                target_flat = x_b.reshape(B, -1).float()
                # Pairwise L2 distances: dist[i,j] = ||gen_i - target_j||^2
                # For each generated image i, we want dist[i,i] < dist[i,j] for j != i
                dists = torch.cdist(x_flat, target_flat, p=2)  # (B, B)
                # Contrastive: log-softmax along target dim, negative of diagonal
                # Lower loss = generated images closer to their own targets
                temperature = 0.1  # scale distances
                log_probs = F.log_softmax(-dists / temperature, dim=1)
                # We want the diagonal to have high probability
                contrastive_loss = -log_probs.diag().mean().item()
                return contrastive_loss
            elif loss_type == "cosine":
                # Cosine similarity loss: cares about direction, not magnitude
                # Mean prediction has low cosine similarity to most targets
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # Negative cosine similarity (lower = more similar)
                x_flat = x.reshape(x.shape[0], -1).float()
                target_flat = x_b.reshape(x_b.shape[0], -1).float()
                cos_sim = F.cosine_similarity(x_flat, target_flat, dim=1)
                return -cos_sim.mean().item()
            elif loss_type == "huber":
                # Huber loss: less sensitive to outliers than MSE
                # May provide better gradients for SPSA
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                return F.smooth_l1_loss(x, x_b).item()
            elif loss_type == "combo":
                # Combined: MSE for signal + contrastive to prevent mean collapse
                # Key insight: use feature-normalized contrastive (cosine-based)
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                B = x.shape[0]
                x_flat = x.reshape(B, -1).float()
                target_flat = x_b.reshape(B, -1).float()
                # MSE component
                mse = F.mse_loss(x, x_b).item()
                # Contrastive on normalized features (cosine distance)
                x_norm = F.normalize(x_flat, dim=1)
                t_norm = F.normalize(target_flat, dim=1)
                # Cosine similarity matrix
                sim = x_norm @ t_norm.T  # (B, B), values in [-1, 1]
                # InfoNCE with temperature
                temperature = 0.5
                log_probs = F.log_softmax(sim / temperature, dim=1)
                contrastive = -log_probs.diag().mean().item()
                # Combined: MSE drives toward targets, contrastive prevents collapse
                return mse + 0.1 * contrastive
            elif loss_type == "rank":
                # Margin ranking loss: penalize when gen image is closer to
                # wrong target than to correct target
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                B = x.shape[0]
                x_flat = x.reshape(B, -1).float()
                target_flat = x_b.reshape(B, -1).float()
                # Per-sample MSE to each target
                # dists[i,j] = mean((gen_i - target_j)^2)
                dists = torch.cdist(x_flat, target_flat, p=2).pow(2) / x_flat.shape[1]
                # For each sample, loss = max(0, d(gen_i, target_i) - min_j≠i(d(gen_i, target_j)) + margin)
                diag = dists.diag()  # correct distances
                # Set diagonal to inf to find min over other targets
                dists_off = dists.clone()
                dists_off.fill_diagonal_(float('inf'))
                min_other = dists_off.min(dim=1).values  # closest wrong target
                margin = 0.01
                # Hinge loss: penalize when correct target isn't closest by margin
                rank_loss = torch.clamp(diag - min_other + margin, min=0).mean().item()
                # Also add plain MSE to provide signal toward targets
                mse = diag.mean().item()
                return mse + rank_loss
            elif loss_type == "mmd":
                # Maximum Mean Discrepancy on raw pixels
                # Compares distribution of generated vs real images
                # Mean prediction gives high MMD because point mass ≠ distribution
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                B = x.shape[0]
                x_flat = x.reshape(B, -1).float()
                y_flat = x_b.reshape(B, -1).float()
                # Gaussian kernel MMD^2
                # k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
                # Use median heuristic for sigma
                with torch.no_grad():
                    all_dists = torch.cdist(
                        torch.cat([x_flat, y_flat], dim=0),
                        torch.cat([x_flat, y_flat], dim=0),
                        p=2
                    )
                    sigma = all_dists.median().item() + 1e-6
                sigma2 = 2 * sigma * sigma
                # K_xx, K_yy, K_xy
                xx_dists = torch.cdist(x_flat, x_flat, p=2).pow(2)
                yy_dists = torch.cdist(y_flat, y_flat, p=2).pow(2)
                xy_dists = torch.cdist(x_flat, y_flat, p=2).pow(2)
                K_xx = torch.exp(-xx_dists / sigma2)
                K_yy = torch.exp(-yy_dists / sigma2)
                K_xy = torch.exp(-xy_dists / sigma2)
                # MMD^2 = E[K_xx] + E[K_yy] - 2*E[K_xy]
                # Exclude diagonal for unbiased estimate
                mask = 1 - torch.eye(B, device=dev)
                mmd2 = (K_xx * mask).sum() / (B * (B-1)) + \
                       (K_yy * mask).sum() / (B * (B-1)) - \
                       2 * K_xy.mean()
                return mmd2.item()
            elif loss_type == "mmd_inception":
                # MMD on InceptionV3 features (more semantically meaningful)
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    gen_samples = torch.clamp(x.float(), -1, 1)
                    gen_01 = (gen_samples + 1) / 2
                    real_01 = (x_b.float() + 1) / 2
                    gen_feat = torch.from_numpy(
                        spsa_inception[0].extract_features(gen_01)
                    ).float().to(dev)
                    real_feat = torch.from_numpy(
                        spsa_inception[0].extract_features(real_01)
                    ).float().to(dev)
                B = gen_feat.shape[0]
                # MMD with Gaussian kernel on inception features
                all_feat = torch.cat([gen_feat, real_feat], dim=0)
                all_dists = torch.cdist(all_feat, all_feat, p=2)
                sigma = all_dists.median().item() + 1e-6
                sigma2 = 2 * sigma * sigma
                xx_dists = torch.cdist(gen_feat, gen_feat, p=2).pow(2)
                yy_dists = torch.cdist(real_feat, real_feat, p=2).pow(2)
                xy_dists = torch.cdist(gen_feat, real_feat, p=2).pow(2)
                K_xx = torch.exp(-xx_dists / sigma2)
                K_yy = torch.exp(-yy_dists / sigma2)
                K_xy = torch.exp(-xy_dists / sigma2)
                mask = 1 - torch.eye(B, device=dev)
                mmd2 = (K_xx * mask).sum() / (B * (B-1)) + \
                       (K_yy * mask).sum() / (B * (B-1)) - \
                       2 * K_xy.mean()
                return mmd2.item()
            elif loss_type == "ssim":
                # SSIM loss: Structural Similarity Index
                # NOT differentiable through ODE — only possible with zero-order!
                # SSIM captures luminance, contrast, and structure — perceptually better than MSE
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # Compute SSIM between generated and target
                # SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
                x_f = x.float()
                y_f = x_b.float()
                C1 = 0.01 ** 2  # (K1*L)^2 where L=2 for [-1,1] range
                C2 = 0.03 ** 2
                # Use 11x11 Gaussian window via avg pooling (simpler, still captures structure)
                kernel_size = 11
                pad = kernel_size // 2
                # Channel-wise processing
                mu_x = F.avg_pool2d(x_f, kernel_size, stride=1, padding=pad)
                mu_y = F.avg_pool2d(y_f, kernel_size, stride=1, padding=pad)
                sigma_x2 = F.avg_pool2d(x_f * x_f, kernel_size, stride=1, padding=pad) - mu_x * mu_x
                sigma_y2 = F.avg_pool2d(y_f * y_f, kernel_size, stride=1, padding=pad) - mu_y * mu_y
                sigma_xy = F.avg_pool2d(x_f * y_f, kernel_size, stride=1, padding=pad) - mu_x * mu_y
                ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                           ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
                # SSIM is in [-1, 1], higher is better. Return 1 - SSIM as loss.
                return (1 - ssim_map.mean()).item()
            elif loss_type == "ssim_mse":
                # Combined SSIM + MSE: perceptual quality + pixel accuracy
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                mse = F.mse_loss(x_f, y_f).item()
                # SSIM
                C1, C2 = 0.01**2, 0.03**2
                ks, pad = 11, 5
                mu_x = F.avg_pool2d(x_f, ks, stride=1, padding=pad)
                mu_y = F.avg_pool2d(y_f, ks, stride=1, padding=pad)
                sigma_x2 = F.avg_pool2d(x_f**2, ks, stride=1, padding=pad) - mu_x**2
                sigma_y2 = F.avg_pool2d(y_f**2, ks, stride=1, padding=pad) - mu_y**2
                sigma_xy = F.avg_pool2d(x_f*y_f, ks, stride=1, padding=pad) - mu_x*mu_y
                ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
                ssim_loss = (1 - ssim.mean()).item()
                return mse + ssim_loss
            elif loss_type == "ssim_mse_light":
                # SSIM+MSE with reduced SSIM weight (0.1) to prevent divergence at high T
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                mse = F.mse_loss(x_f, y_f).item()
                # SSIM with small weight
                C1, C2 = 0.01**2, 0.03**2
                ks, pad = 11, 5
                mu_x = F.avg_pool2d(x_f, ks, stride=1, padding=pad)
                mu_y = F.avg_pool2d(y_f, ks, stride=1, padding=pad)
                sigma_x2 = F.avg_pool2d(x_f**2, ks, stride=1, padding=pad) - mu_x**2
                sigma_y2 = F.avg_pool2d(y_f**2, ks, stride=1, padding=pad) - mu_y**2
                sigma_xy = F.avg_pool2d(x_f*y_f, ks, stride=1, padding=pad) - mu_x*mu_y
                ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
                ssim_loss = (1 - ssim.mean()).item()
                return mse + 0.1 * ssim_loss
            elif loss_type == "mse_clamp2":
                # Progressive SSIM: pure MSE first 30%, then gradually blend SSIM in
                # This avoids SSIM instability early when model is rough
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                mse = F.mse_loss(x_f, y_f).item()
                # Blend in SSIM after 30% of training (progress available via T ramp)
                ssim_weight = max(0.0, min(0.2, (T - 2) * 0.1))  # 0 when T<=2, ramps to 0.2 at T=4
                if ssim_weight > 0:
                    C1, C2 = 0.01**2, 0.03**2
                    ks, pad = 11, 5
                    mu_x = F.avg_pool2d(x_f, ks, stride=1, padding=pad)
                    mu_y = F.avg_pool2d(y_f, ks, stride=1, padding=pad)
                    sigma_x2 = F.avg_pool2d(x_f**2, ks, stride=1, padding=pad) - mu_x**2
                    sigma_y2 = F.avg_pool2d(y_f**2, ks, stride=1, padding=pad) - mu_y**2
                    sigma_xy = F.avg_pool2d(x_f*y_f, ks, stride=1, padding=pad) - mu_x*mu_y
                    ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
                    ssim_loss = (1 - ssim.mean()).item()
                    return mse + ssim_weight * ssim_loss
                return mse
            elif loss_type == "denoising_discrete":
                # MSE on discretized pixel values (round to 0-255 then back)
                # NOT differentiable — zero-order exclusive!
                # Directly optimizes for the pixel values that matter in final images
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # Discretize to 0-255 range (images are in [-1, 1])
                x_disc = torch.round((x.float() + 1) * 127.5).clamp(0, 255) / 127.5 - 1
                y_disc = torch.round((x_b.float() + 1) * 127.5).clamp(0, 255) / 127.5 - 1
                return F.mse_loss(x_disc, y_disc).item()
            elif loss_type == "denoising_multires":
                # Multi-resolution MSE: full + half resolution averaged
                # Captures both fine detail and coarse structure
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                mse_full = F.mse_loss(x_f, y_f).item()
                mse_half = F.mse_loss(F.avg_pool2d(x_f, 2), F.avg_pool2d(y_f, 2)).item()
                return mse_full + 0.5 * mse_half
            elif loss_type == "denoising_huber":
                # Huber loss (L1 for large errors, L2 for small) — robust to outliers
                # Non-differentiable at the L1/L2 boundary — zero-order exclusive!
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                return F.huber_loss(x_f, y_f, delta=0.5).item()
            elif loss_type == "denoising_lowres":
                # MSE on 2× downscaled images — smoother loss landscape
                # Helps early training by ignoring fine detail
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = F.avg_pool2d(x.float(), 2)
                y_f = F.avg_pool2d(x_b.float(), 2)
                return F.mse_loss(x_f, y_f).item()
            elif loss_type == "denoising_mae":
                # Mean Absolute Error — L1 loss, robust to outlier pixels
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                return F.l1_loss(x_f, y_f).item()
            elif loss_type == "denoising_edge":
                # MSE + gradient-domain (edge) loss
                # Computes spatial gradients (finite differences) of both predicted and target,
                # then adds MSE on the gradients. This emphasizes edge accuracy which directly
                # impacts FID — sharper edges = better perceived quality.
                # Zero-order exclusive: gradient-domain loss is non-differentiable through ODE.
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                # Pixel loss
                mse = F.mse_loss(x_f, y_f)
                # Edge loss: finite-difference spatial gradients
                dx_pred = x_f[:, :, 1:, :] - x_f[:, :, :-1, :]
                dx_targ = y_f[:, :, 1:, :] - y_f[:, :, :-1, :]
                dy_pred = x_f[:, :, :, 1:] - x_f[:, :, :, :-1]
                dy_targ = y_f[:, :, :, 1:] - y_f[:, :, :, :-1]
                edge = F.mse_loss(dx_pred, dx_targ) + F.mse_loss(dy_pred, dy_targ)
                return (mse + 0.5 * edge).item()
            elif loss_type == "denoising_tv":
                # Total variation regularized denoising loss
                # MSE + TV penalty on predicted image. TV encourages piecewise-smooth
                # outputs, reducing noise/artifacts from imperfect denoising.
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                mse = F.mse_loss(x_f, y_f)
                # TV of the ERROR, not the image — penalizes spatially-correlated errors
                err = x_f - y_f
                tv = torch.mean(torch.abs(err[:, :, 1:, :] - err[:, :, :-1, :])) + \
                     torch.mean(torch.abs(err[:, :, :, 1:] - err[:, :, :, :-1]))
                return (mse + 0.1 * tv).item()
            elif loss_type == "fft":
                # Frequency domain loss: penalize spectral differences
                # NOT differentiable through ODE — zero-order exclusive!
                # Images have specific frequency characteristics that MSE ignores
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                # 2D FFT of generated and target
                x_f = x.float()
                y_f = x_b.float()
                fft_gen = torch.fft.fft2(x_f)
                fft_real = torch.fft.fft2(y_f)
                # Compare magnitude spectra (phase is noisy, magnitude captures structure)
                mag_gen = torch.abs(fft_gen)
                mag_real = torch.abs(fft_real)
                # Log-magnitude for better dynamic range
                log_mag_gen = torch.log1p(mag_gen)
                log_mag_real = torch.log1p(mag_real)
                return F.mse_loss(log_mag_gen, log_mag_real).item()
            elif loss_type == "multiscale":
                # Multi-scale loss: compare at original + downsampled resolutions
                # Captures both fine detail and global structure
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                total_loss = 0.0
                # Scale 1: full resolution (64x64)
                total_loss += F.mse_loss(x_f, y_f).item()
                # Scale 2: half resolution (32x32)
                x_half = F.avg_pool2d(x_f, 2)
                y_half = F.avg_pool2d(y_f, 2)
                total_loss += F.mse_loss(x_half, y_half).item()
                # Scale 3: quarter resolution (16x16)
                x_quarter = F.avg_pool2d(x_f, 4)
                y_quarter = F.avg_pool2d(y_f, 4)
                total_loss += F.mse_loss(x_quarter, y_quarter).item()
                return total_loss / 3.0
            elif loss_type == "ssim_mse_fft":
                # Triple combo: MSE + SSIM + FFT spectral loss
                # Pixel accuracy + perceptual quality + frequency fidelity
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                # MSE
                mse = F.mse_loss(x_f, y_f).item()
                # SSIM
                C1, C2 = 0.01**2, 0.03**2
                ks, pad = 11, 5
                mu_x = F.avg_pool2d(x_f, ks, stride=1, padding=pad)
                mu_y = F.avg_pool2d(y_f, ks, stride=1, padding=pad)
                sigma_x2 = F.avg_pool2d(x_f**2, ks, stride=1, padding=pad) - mu_x**2
                sigma_y2 = F.avg_pool2d(y_f**2, ks, stride=1, padding=pad) - mu_y**2
                sigma_xy = F.avg_pool2d(x_f*y_f, ks, stride=1, padding=pad) - mu_x*mu_y
                ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
                ssim_loss = (1 - ssim.mean()).item()
                # FFT spectral
                fft_gen = torch.fft.fft2(x_f)
                fft_real = torch.fft.fft2(y_f)
                fft_loss = F.mse_loss(torch.log1p(torch.abs(fft_gen)), torch.log1p(torch.abs(fft_real))).item()
                return mse + ssim_loss + 0.1 * fft_loss  # FFT scaled down (different magnitude)
            elif loss_type == "loss_ensemble":
                # Weighted ensemble of MSE + SSIM + multiscale MSE
                # Combines pixel, perceptual, and structural at multiple scales
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                # Full-res MSE
                mse = F.mse_loss(x_f, y_f).item()
                # Half-res MSE (captures global structure)
                mse_half = F.mse_loss(F.avg_pool2d(x_f, 2), F.avg_pool2d(y_f, 2)).item()
                # SSIM
                C1, C2 = 0.01**2, 0.03**2
                ks, pad = 11, 5
                mu_x = F.avg_pool2d(x_f, ks, stride=1, padding=pad)
                mu_y = F.avg_pool2d(y_f, ks, stride=1, padding=pad)
                sigma_x2 = F.avg_pool2d(x_f**2, ks, stride=1, padding=pad) - mu_x**2
                sigma_y2 = F.avg_pool2d(y_f**2, ks, stride=1, padding=pad) - mu_y**2
                sigma_xy = F.avg_pool2d(x_f*y_f, ks, stride=1, padding=pad) - mu_x*mu_y
                ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
                ssim_loss = (1 - ssim.mean()).item()
                return mse + 0.5 * mse_half + ssim_loss
            elif loss_type == "hist_match":
                # Histogram matching loss: match pixel intensity distributions
                # IMPOSSIBLE with backprop — histogram binning is non-differentiable!
                # This captures color/brightness distribution matching
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                x_f = x.float()
                y_f = x_b.float()
                # MSE component (pixel accuracy)
                mse = F.mse_loss(x_f, y_f).item()
                # Histogram distance per channel (32 bins from -1 to 1)
                n_bins = 32
                hist_loss = 0.0
                for c in range(3):
                    gen_vals = x_f[:, c].reshape(-1)
                    ref_vals = y_f[:, c].reshape(-1)
                    gen_hist = torch.histc(gen_vals, bins=n_bins, min=-1.0, max=1.0)
                    ref_hist = torch.histc(ref_vals, bins=n_bins, min=-1.0, max=1.0)
                    # Normalize to probability distributions
                    gen_hist = gen_hist / (gen_hist.sum() + 1e-8)
                    ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
                    # Earth mover's distance approximation via cumulative difference
                    hist_loss += torch.abs(gen_hist.cumsum(0) - ref_hist.cumsum(0)).sum().item()
                hist_loss /= 3.0
                return mse + 0.1 * hist_loss  # Scale histogram loss
            elif loss_type == "direct_fid":
                # DIRECT FID as training loss — THE unique zero-order advantage!
                # Backprop CANNOT optimize this: it requires differentiating through
                # InceptionV3 + matrix square root. SPSA doesn't care!
                # We compute a mini-FID between generated batch and reference stats.
                dev = x_b.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(noise_seed[0] + batch_idx)
                noise = torch.randn(x_b.shape, device=dev, dtype=x_b.dtype, generator=gen)
                x = noise.clone()
                dt = 1.0 / T
                for i in range(T):
                    t_val = i / T
                    t_tensor = torch.full((x_b.shape[0],), t_val, device=dev)
                    velocity = model(x, t_tensor, class_labels=y_b)
                    x = x + velocity * dt
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    samples = torch.clamp(x.float(), -1, 1)
                    samples_01 = (samples + 1) / 2
                    gen_features = spsa_inception[0].extract_features(samples_01)
                gen_mu = np.mean(gen_features, axis=0)
                gen_sigma = np.cov(gen_features, rowvar=False)
                # Simplified FID: ||mu_gen - mu_ref||^2 + Tr(sigma_gen + sigma_ref - 2*sqrt(sigma_gen*sigma_ref))
                # For small batch, sigma is rank-deficient, so we use a regularized version:
                # FID ≈ ||mu_gen - mu_ref||^2 + Tr(sigma_gen) + Tr(sigma_ref) - 2*Tr(sqrt(sigma_gen@sigma_ref))
                diff = gen_mu - spsa_ref_mu[0]
                mean_term = float(np.dot(diff, diff))
                # For the covariance term, use trace norm (simpler, avoids sqrtm instability)
                trace_gen = float(np.trace(gen_sigma))
                trace_ref = float(np.trace(spsa_ref_sigma[0]))
                # Cross term via eigenvalue approximation
                try:
                    from scipy import linalg as sp_linalg
                    product = gen_sigma @ spsa_ref_sigma[0]
                    eigvals = np.real(sp_linalg.eigvals(product))
                    eigvals = np.maximum(eigvals, 0)  # numerical stability
                    cross_term = float(np.sum(np.sqrt(eigvals)))
                except:
                    cross_term = 0.0  # fallback: just use mean term
                fid_approx = mean_term + trace_gen + trace_ref - 2 * cross_term
                return fid_approx

    # Probe setup for LR search
    probe_data = [None, None]

    def resample_probe_batch(seed):
        x_b, y_b, _ = next(train_loader)
        probe_data[0] = x_b
        probe_data[1] = y_b

    # Initialize probe batch
    resample_probe_batch(0)

    def probe_loss_fn():
        """Fixed-batch loss for LR probing. Delegates to spsa_loss_fn."""
        # Temporarily set probe data as spsa_batches for probe eval
        old_batches = list(spsa_batches)
        spsa_batches.clear()
        spsa_batches.append((probe_data[0], probe_data[1]))
        result = spsa_loss_fn(batch_idx=0)
        spsa_batches.clear()
        spsa_batches.extend(old_batches)
        return result

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

# Pre-load fixed buffer if using fixed-batch mode
fixed_buffer = None
fixed_all_batch = None  # Pre-concatenated batch for "all" mode
if args.fixed_batch_size > 0 and args.solver == "spsa":
    fixed_buffer = []
    print(f"Pre-loading {args.fixed_batch_size} fixed images for SPSA training...")
    for i in range(args.fixed_batch_size):
        x_b, y_b, _ = next(train_loader)
        fixed_buffer.append((x_b[:1], y_b[:1]))
    if args.fixed_batch_mode == "all":
        # Concatenate all images into a single batch for simultaneous training
        fixed_all_batch = (
            torch.cat([fb[0] for fb in fixed_buffer], dim=0),
            torch.cat([fb[1] for fb in fixed_buffer], dim=0),
        )
        print(f"  Loaded {len(fixed_buffer)} images (all-at-once mode, batch_size={fixed_all_batch[0].shape[0]})")
    else:
        print(f"  Loaded {len(fixed_buffer)} images (cycling 1-at-a-time)")

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

    if args.solver == "backprop":
        # ----- Backprop: teacher-forced flow matching -----
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = flow_matching.train_loss(model, x, class_labels=y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        # Progress and schedules
        progress = min(total_training_time / args.time_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # EMA update
        if ema_model is not None:
            src_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            with torch.no_grad():
                for ema_p, src_p in zip(ema_model.parameters(), src_model.parameters()):
                    ema_p.lerp_(src_p.data, 1.0 - args.ema_decay)
        model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()

    else:
        # ----- SPSA: zero-order, no teacher forcing -----
        # Load batches for this step (consistent across ±epsilon evaluations)
        spsa_batches.clear()
        if args.fixed_batch_size > 0 and fixed_buffer is not None:
            if args.fixed_batch_mode == "all" and fixed_all_batch is not None:
                # Use all fixed images every step
                x_b, y_b = fixed_all_batch
                if args.augment_fixed:
                    # Random horizontal flip (deterministic per step for ±epsilon consistency)
                    aug_rng = torch.Generator(device=x_b.device)
                    aug_rng.manual_seed(step * 9999 + 13)
                    flip_mask = torch.rand(x_b.shape[0], 1, 1, 1, device=x_b.device, generator=aug_rng) > 0.5
                    x_b = torch.where(flip_mask, x_b.flip(-1), x_b)
                spsa_batches.append((x_b, y_b))
                noise_seed[0] = (42 + step) if args.vary_noise else 42
            else:
                # Cycle through images one at a time
                idx = step % len(fixed_buffer)
                x_b, y_b = fixed_buffer[idx]
                spsa_batches.append((x_b, y_b))
                noise_seed[0] = idx * 100 + 42
        else:
            for _ in range(args.spsa_accum_steps):
                x_b, y_b, epoch = next(train_loader)
                spsa_batches.append((x_b, y_b))
            noise_seed[0] = step * 100

        # Pre-generate deterministic t and noise for teacher loss
        # (must be identical across all ±epsilon evaluations within this step)
        if args.spsa_loss_type == "teacher":
            gen_teacher = torch.Generator(device='cuda')
            gen_teacher.manual_seed(step * 777 + 42)
            x_ref = spsa_batches[0][0]
            spsa_teacher_t[0] = torch.rand(x_ref.shape[0], device='cuda', generator=gen_teacher)
            spsa_teacher_noise[0] = torch.randn(x_ref.shape, device='cuda', dtype=x_ref.dtype, generator=gen_teacher)

        # Dynamic T scheduling (same T for ±epsilon within one step)
        if args.t_schedule == "fixed":
            current_T[0] = args.denoising_steps
        elif args.t_schedule == "linear":
            # Ramp from t_min to t_max over training
            progress = min(total_training_time / args.time_budget, 1.0)
            current_T[0] = max(1, int(args.t_min + (args.t_max - args.t_min) * progress))
        elif args.t_schedule == "lognormal":
            # Sample T ~ lognormal each step, clamp to [1, 200]
            # Use deterministic seed so ±epsilon see same T
            t_rng = torch.Generator()
            t_rng.manual_seed(step * 31337 + 7)
            log_t = args.t_lognormal_mu + args.t_lognormal_sigma * torch.randn(1, generator=t_rng).item()
            current_T[0] = max(1, min(200, int(round(math.exp(log_t)))))
        elif args.t_schedule == "exponential":
            # Exponential ramp: spend more time at low T
            # T = t_min * (t_max/t_min)^progress
            progress = min(total_training_time / args.time_budget, 1.0)
            ratio = max(args.t_max / max(args.t_min, 1), 1.0)
            current_T[0] = max(1, int(args.t_min * (ratio ** progress)))
        elif args.t_schedule == "curriculum":
            # Curriculum: stay at t_min for curriculum_frac of training, then linear ramp to t_max
            # Optional polish phase: drop back to t_min for final curriculum_polish fraction
            progress = min(total_training_time / args.time_budget, 1.0)
            cf = args.curriculum_frac
            pf = args.curriculum_polish
            if pf > 0 and progress >= (1.0 - pf):
                # Polish phase: back to t_min for final refinement
                current_T[0] = args.t_min
            elif progress < cf:
                current_T[0] = args.t_min
            else:
                # Ramp phase: compress ramp into (cf, 1-pf) range
                ramp_end = 1.0 - pf if pf > 0 else 1.0
                ramp_progress = (progress - cf) / (ramp_end - cf)
                ramp_progress = min(ramp_progress, 1.0)
                current_T[0] = max(1, int(args.t_min + (args.t_max - args.t_min) * ramp_progress))
        elif args.t_schedule == "curriculum_exp":
            # Curriculum + exponential ramp: stay at t_min for curriculum_frac,
            # then exponential ramp to t_max (spends more ramp time at low T)
            progress = min(total_training_time / args.time_budget, 1.0)
            cf = args.curriculum_frac
            if progress < cf:
                current_T[0] = args.t_min
            else:
                ramp_progress = (progress - cf) / (1.0 - cf)
                ratio = max(args.t_max / max(args.t_min, 1), 1.0)
                current_T[0] = max(1, int(args.t_min * (ratio ** ramp_progress)))
        elif args.t_schedule == "curriculum_step":
            # Curriculum step: stay at t_min for curriculum_frac, then JUMP to t_max
            # No linear ramp through intermediate T values — skip T=2,3
            progress = min(total_training_time / args.time_budget, 1.0)
            cf = args.curriculum_frac
            if progress < cf:
                current_T[0] = args.t_min
            else:
                current_T[0] = args.t_max
        elif args.t_schedule == "curriculum_mix":
            # Curriculum mix: stay at t_min for curriculum_frac, then randomly
            # sample T from {t_min, t_max} each step (keep reinforcing T=1)
            progress = min(total_training_time / args.time_budget, 1.0)
            cf = args.curriculum_frac
            if progress < cf:
                current_T[0] = args.t_min
            else:
                _mix_rng = random.Random(step * 31337 + 42)
                current_T[0] = args.t_min if _mix_rng.random() < 0.5 else args.t_max
        elif args.t_schedule == "reverse":
            # Reverse: start at t_max, ramp down to t_min
            progress = min(total_training_time / args.time_budget, 1.0)
            current_T[0] = max(1, int(args.t_max - (args.t_max - args.t_min) * progress))
        elif args.t_schedule == "cyclic":
            # Cyclic: alternate between t_min and t_max every 10% of training
            progress = min(total_training_time / args.time_budget, 1.0)
            cycle = int(progress * 10) % 2
            current_T[0] = args.t_min if cycle == 0 else args.t_max
        elif args.t_schedule == "stochastic":
            # Stochastic T: sample uniformly from [t_min, current_max_T]
            # current_max_T grows linearly like the linear schedule
            # Different T per step gives gradient diversity across diffusion depths
            progress = min(total_training_time / args.time_budget, 1.0)
            max_T = max(args.t_min, int(args.t_min + (args.t_max - args.t_min) * progress))
            _t_rng = random.Random(step * 31337 + 7)
            current_T[0] = _t_rng.randint(args.t_min, max(args.t_min, max_T))
        elif args.t_schedule == "phased":
            # Phased T: divide training into equal phases, one T per phase
            # LR restarts at each phase boundary for fresh momentum
            progress = min(total_training_time / args.time_budget, 1.0)
            n_phases = args.t_max - args.t_min + 1  # e.g., T=1→4 = 4 phases
            phase = min(int(progress * n_phases), n_phases - 1)
            current_T[0] = args.t_min + phase
        elif args.t_schedule == "curriculum_stoch":
            # Curriculum + stochastic: stay at t_min for curriculum_frac,
            # then RANDOMLY sample T from [t_min, current_max_T] where
            # current_max_T ramps linearly. Gives gradient diversity with curriculum structure.
            progress = min(total_training_time / args.time_budget, 1.0)
            cf = args.curriculum_frac
            if progress < cf:
                current_T[0] = args.t_min
            else:
                ramp_progress = (progress - cf) / (1.0 - cf)
                max_T = max(args.t_min, int(args.t_min + (args.t_max - args.t_min) * ramp_progress))
                _t_rng = random.Random(step * 31337 + 7)
                current_T[0] = _t_rng.randint(args.t_min, max(args.t_min, max_T))
        elif args.t_schedule == "curriculum_sawtooth":
            # Curriculum + sawtooth: stay at t_min for curriculum_frac,
            # then do 3 mini-ramps from t_min to progressively higher T values.
            # Each mini-ramp re-anchors at t_min, preventing drift at high T.
            progress = min(total_training_time / args.time_budget, 1.0)
            cf = args.curriculum_frac
            if progress < cf:
                current_T[0] = args.t_min
            else:
                ramp_progress = (progress - cf) / (1.0 - cf)
                n_teeth = 3
                tooth = min(int(ramp_progress * n_teeth), n_teeth - 1)
                tooth_progress = (ramp_progress * n_teeth) - tooth
                tooth_progress = min(tooth_progress, 1.0)
                # Each tooth reaches progressively higher T
                tooth_max = args.t_min + int((args.t_max - args.t_min) * (tooth + 1) / n_teeth)
                current_T[0] = max(1, int(args.t_min + (tooth_max - args.t_min) * tooth_progress))

        elif args.t_schedule == "curriculum_weighted":
            # Weighted phase schedule: spend exponentially decreasing time at each T
            # T=1: 50%, T=2: 25%, T=3: 15%, T=4: 10% (for T_min=1, T_max=4)
            # More time at low T where SPSA gradients are most reliable
            progress = min(total_training_time / args.time_budget, 1.0)
            n_T = args.t_max - args.t_min + 1  # number of T values
            # Geometric weights: 2^(n-i) for i-th phase
            weights = [2.0 ** (n_T - 1 - i) for i in range(n_T)]
            total_w = sum(weights)
            boundaries = []
            cumsum = 0.0
            for w in weights:
                cumsum += w / total_w
                boundaries.append(cumsum)
            # Find which phase we're in
            current_T[0] = args.t_max  # default to last
            for i, b in enumerate(boundaries):
                if progress < b:
                    current_T[0] = args.t_min + i
                    break
        elif args.t_schedule == "adaptive":
            # Adaptive T: increase T when loss stabilizes at current T
            # Uses loss EMA slope to detect convergence, then bumps T
            progress = min(total_training_time / args.time_budget, 1.0)
            if not hasattr(args, '_adaptive_T_state'):
                args._adaptive_T_state = {
                    'current_T': args.t_min,
                    'loss_ema': None,
                    'loss_ema_slow': None,
                    'steps_at_T': 0,
                    'min_steps': 500,  # minimum steps at each T before considering increase
                }
            st = args._adaptive_T_state
            st['steps_at_T'] += 1
            # train_loss_f may not exist on first step (it's set after trainer.step())
            try:
                _loss_val = train_loss_f
            except NameError:
                _loss_val = None
            if _loss_val is not None:
                if st['loss_ema'] is None:
                    st['loss_ema'] = _loss_val
                    st['loss_ema_slow'] = _loss_val
                else:
                    st['loss_ema'] = 0.99 * st['loss_ema'] + 0.01 * _loss_val
                    st['loss_ema_slow'] = 0.999 * st['loss_ema_slow'] + 0.001 * _loss_val
            # Increase T if: loss EMA has converged (fast ≈ slow) AND enough steps at this T
            # AND haven't reached t_max AND still have training time left
            converged = (st['loss_ema'] is not None and st['loss_ema_slow'] is not None
                         and abs(st['loss_ema'] - st['loss_ema_slow']) < 0.005 * st['loss_ema_slow'])
            if (converged and st['steps_at_T'] > st['min_steps']
                    and st['current_T'] < args.t_max and progress < 0.95):
                st['current_T'] = min(st['current_T'] + 1, args.t_max)
                st['steps_at_T'] = 0
                st['loss_ema'] = None
                st['loss_ema_slow'] = None
                print(f"\n  [Adaptive T] Increasing T to {st['current_T']} at progress {progress:.1%}")
            current_T[0] = st['current_T']

        # Batch refresh: periodically load new random images
        if args.batch_refresh_pct > 0 and args.fixed_batch_size > 0 and fixed_buffer is not None:
            progress_now = min(total_training_time / args.time_budget, 1.0)
            refresh_interval = args.batch_refresh_pct
            current_bucket = int(progress_now / refresh_interval) if refresh_interval > 0 else 0
            if not hasattr(args, '_last_refresh_bucket'):
                args._last_refresh_bucket = 0
            if current_bucket > args._last_refresh_bucket and progress_now > 0.01:
                args._last_refresh_bucket = current_bucket
                fixed_buffer.clear()
                for i in range(args.fixed_batch_size):
                    x_b, y_b, _ = next(train_loader)
                    fixed_buffer.append((x_b[:1], y_b[:1]))
                if args.fixed_batch_mode == "all":
                    fixed_all_batch = (
                        torch.cat([fb[0] for fb in fixed_buffer], dim=0),
                        torch.cat([fb[1] for fb in fixed_buffer], dim=0),
                    )
                print(f"  [Batch refresh at {progress_now:.1%}] Loaded {args.fixed_batch_size} new images")

        # Batch trickle: replace 1 image every N steps (gradual replacement)
        if args.batch_trickle_interval > 0 and args.fixed_batch_size > 0 and fixed_buffer is not None:
            if step > 0 and step % args.batch_trickle_interval == 0:
                import random as _rand
                replace_idx = _rand.randint(0, len(fixed_buffer) - 1)
                x_new, y_new, _ = next(train_loader)
                fixed_buffer[replace_idx] = (x_new[:1], y_new[:1])
                if args.fixed_batch_mode == "all":
                    fixed_all_batch = (
                        torch.cat([fb[0] for fb in fixed_buffer], dim=0),
                        torch.cat([fb[1] for fb in fixed_buffer], dim=0),
                    )

        # Adaptive perturbation count: decay n_perts over training
        if args.adaptive_perts:
            progress_now = min(total_training_time / args.time_budget, 1.0)
            min_perts = max(1, int(args.n_perts * args.adaptive_perts_min_frac))
            trainer.n_perts = max(min_perts, int(args.n_perts * (1 - (1 - args.adaptive_perts_min_frac) * progress_now)))

        # SPSA step — with optional per-perturbation hooks (T variation, noise variation)
        _pert_hook = None
        _needs_hook = (args.t_schedule == "stochastic_pert") or args.multi_noise
        if _needs_hook:
            _stoch_progress = min(total_training_time / args.time_budget, 1.0)
            _stoch_max_T = max(args.t_min, int(args.t_min + (args.t_max - args.t_min) * _stoch_progress))
            def _pert_hook(pert_idx, iteration):
                if args.t_schedule == "stochastic_pert":
                    # Each perturbation gets a different T sampled from expanding range
                    _rng = random.Random(iteration * 10000 + pert_idx * 777 + 42)
                    current_T[0] = _rng.randint(args.t_min, max(args.t_min, _stoch_max_T))
                if args.multi_noise:
                    # Each perturbation uses a different noise seed for ODE denoising
                    # ±epsilon evaluations still share the same noise (deterministic via pert_idx)
                    noise_seed[0] = iteration * 100 + pert_idx * 7 + 42
        # Self-distillation: compute high-quality targets at unperturbed model
        if args.spsa_loss_type == "denoising_selfdistill" and step > 100:
            with torch.no_grad(), autocast_ctx:
                x_b_sd, y_b_sd = spsa_batches[0]
                dev_sd = x_b_sd.device
                gen_sd = torch.Generator(device=dev_sd)
                gen_sd.manual_seed(noise_seed[0])
                noise_sd = torch.randn(x_b_sd.shape, device=dev_sd, dtype=x_b_sd.dtype, generator=gen_sd)
                x_sd = noise_sd.clone()
                # Use T=20 for high-quality generation (vs current T=1-4)
                T_hi = 20
                dt_sd = 1.0 / T_hi
                for i_sd in range(T_hi):
                    t_val_sd = i_sd / T_hi
                    t_sd = torch.full((x_b_sd.shape[0],), t_val_sd, device=dev_sd)
                    vel_sd = model(x_sd, t_sd, class_labels=y_b_sd)
                    x_sd = x_sd + vel_sd * dt_sd
                selfdistill_targets[0] = x_sd.detach()
        elif args.spsa_loss_type == "denoising_selfdistill":
            selfdistill_targets[0] = None  # No targets for first 100 steps

        train_loss_f = trainer.step(spsa_loss_fn, step, per_pert_hook=_pert_hook)

        # Checkpoint rollback: save good states, restore on divergence
        if args.checkpoint_rollback:
            if not hasattr(args, '_ckpt_state'):
                args._ckpt_state = {n: p.data.clone() for n, p in model.named_parameters()}
                args._ckpt_loss_ema = train_loss_f
                args._ckpt_step = step
                args._ckpt_lr = trainer.lr
                args._rollback_count = 0
            else:
                args._ckpt_loss_ema = 0.99 * args._ckpt_loss_ema + 0.01 * train_loss_f
                # Save checkpoint every 200 steps if loss is stable
                if step % 200 == 0 and train_loss_f < 1.5 * args._ckpt_loss_ema:
                    args._ckpt_state = {n: p.data.clone() for n, p in model.named_parameters()}
                    args._ckpt_step = step
                    args._ckpt_lr = trainer.lr
                # Detect divergence: loss > 1.5x EMA
                if train_loss_f > 1.5 * args._ckpt_loss_ema and step > 100:
                    args._rollback_count += 1
                    if args._rollback_count >= 3:  # 3 consecutive bad steps
                        print(f"\n  ROLLBACK at step {step}: loss {train_loss_f:.4f} > 1.5x EMA {args._ckpt_loss_ema:.4f}")
                        print(f"  Restoring checkpoint from step {args._ckpt_step}, halving LR {trainer.lr:.2e} → {trainer.lr/2:.2e}")
                        for n, p in model.named_parameters():
                            if n in args._ckpt_state:
                                p.data.copy_(args._ckpt_state[n])
                        trainer.lr = max(trainer.lr / 2, 1e-4)
                        args._rollback_count = 0
                        args._ckpt_loss_ema = train_loss_f * 0.8  # Reset EMA higher to avoid immediate re-trigger
                else:
                    args._rollback_count = 0

        # SWA: Stochastic Weight Averaging in the last swa_frac of training
        if args.swa_frac > 0:
            progress_now = min(total_training_time / args.time_budget, 1.0)
            if progress_now >= (1.0 - args.swa_frac):
                if not hasattr(args, '_swa_state'):
                    # Initialize SWA: deep copy current weights
                    import copy as _copy
                    args._swa_state = {n: p.data.clone().float() for n, p in model.named_parameters()}
                    args._swa_count = 1
                else:
                    # Running average: avg = avg + (new - avg) / count
                    args._swa_count += 1
                    for n, p in model.named_parameters():
                        args._swa_state[n].add_((p.data.float() - args._swa_state[n]) / args._swa_count)

        # Update LR based on schedule (when not using search)
        progress = min(total_training_time / args.time_budget, 1.0)
        if args.search_strategy == "none":
            if args.t_schedule == "phased":
                # Phased T: restart LR at each phase boundary with independent warmup/warmdown
                n_phases = args.t_max - args.t_min + 1
                phase_progress = (progress * n_phases) % 1.0  # 0→1 within each phase
                lrm = get_lr_multiplier(phase_progress)  # independent schedule per phase
            else:
                lrm = get_lr_multiplier(progress)
            trainer.lr = args.lr * lrm
            if args.epsilon is None:
                trainer.epsilon = max(trainer.lr, 1e-8)
            elif args.eps_schedule != "fixed" and args.epsilon is not None:
                # Adaptive epsilon: decay from eps_max to epsilon over training
                eps_hi = args.eps_max if args.eps_max is not None else args.epsilon * 10
                eps_lo = args.epsilon
                if args.eps_schedule == "cosine_decay":
                    eps_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
                    trainer.epsilon = eps_lo + (eps_hi - eps_lo) * eps_mult
                elif args.eps_schedule == "linear_decay":
                    trainer.epsilon = eps_hi + (eps_lo - eps_hi) * progress
                elif args.eps_schedule == "linear_warmup":
                    # Warmup epsilon from eps_lo/2 to eps_lo over first 10% of training
                    eps_min = eps_lo * 0.5
                    if progress < 0.1:
                        trainer.epsilon = eps_min + (eps_lo - eps_min) * (progress / 0.1)
                    else:
                        trainer.epsilon = eps_lo
                elif args.eps_schedule == "t_coupled":
                    # Scale epsilon by sqrt(current_T) — maintains gradient SNR as T increases.
                    # With more denoising steps, the loss landscape is smoother per-parameter,
                    # so larger perturbations are needed to measure the gradient signal.
                    import math as _math
                    trainer.epsilon = eps_lo * _math.sqrt(max(current_T[0], 1))
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

    solver_tag = "BP" if args.solver == "backprop" else "SPSA"
    loss_tag = ""
    if args.solver == "spsa" and args.loss_warmup_frac > 0:
        loss_tag = f" [{active_loss_type[0]}]"
    print(f"\rstep {step:05d} ({pct_done:.1f}%) [{solver_tag}]{loss_tag} | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | img/sec: {img_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

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

# Final eval
model.eval()
if args.solver == "backprop" and ema_model is not None:
    model_for_eval = ema_model.float()
    print("Using EMA model for evaluation")
else:
    model_for_eval = model._orig_mod.float() if hasattr(model, '_orig_mod') else model.float()

# Apply SWA weights if available
if hasattr(args, '_swa_state') and args._swa_state is not None:
    print(f"Applying SWA weights (averaged {args._swa_count} checkpoints over last {args.swa_frac*100:.0f}% of training)")
    for n, p in model_for_eval.named_parameters():
        if n in args._swa_state:
            p.data.copy_(args._swa_state[n])

class Float32Wrapper(nn.Module):
    """Wrapper that ensures model output is always float32 for FID evaluation."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).float()
    def parameters(self):
        return self.model.parameters()

val_fid = evaluate_fid(Float32Wrapper(model_for_eval), flow_matching, args.device_batch_size)

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
print(f"depth:            {args.depth}")
if args.solver == "spsa":
    print(f"denoising_steps:  {args.denoising_steps}")
    print(f"n_perts:          {args.n_perts}")
    print(f"final_lr:         {trainer.lr:.2e}")
    print(f"final_epsilon:    {trainer.epsilon:.2e}")
