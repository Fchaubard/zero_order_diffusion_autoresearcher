# NeurIPS Submission: Research Plan

## Title (working)
**"Recursive Diffusion Transformers: Shared-Weight Hierarchical Refinement for Efficient Image Generation"**

Alternative titles:
- "TRM-Diffusion: How Recursive Reasoning Models Learn to Denoise"
- "Asymmetric Recursion is All You Need: Efficient Diffusion with Shared-Weight Transformers"

---

## 1. Core Thesis

A single shared-weight transformer block, applied recursively through a two-level latent hierarchy (z_H "answer accumulator" / z_L "working memory"), can generate competitive images on ImageNet 64×64 with only 13.6M parameters. The key insight is that **the allocation of compute across recursive cycles matters more than the total depth**, and that **asymmetric recursion** (front-loading working memory refinement) consistently outperforms symmetric allocation across all training budgets.

---

## 2. Key Findings to Present

### 2.1 The Four Loops of Recursive Diffusion

The TRM-Diffusion architecture has **four distinct recursion loops**, each controlling a different aspect of compute allocation:

```
for outer_refinement in range(T_refine):            # Loop 4: Outer refinement (eval-time)
    for thtl in range(T_H_T_L):                     # Loop 3: H-cycle-L-cycle outer loops 
        for tl in range(T_L):                       # Loop 1: L-cycle repetitions
            z_L = shared_block(z_L + z_H + c)         # L consolidation 
        for th in range(T_H):                       # Loop 2: H-cycle repetitions
            z_H = shared_block(z_H + z_L)         # H consolidation
```


| Loop | Variable | Controls | Current Best | Status |
|------|----------|----------|--------------|--------|
| **Loop 1**: L-cycle reps | `T_L` | # unique layers in shared block | 1 | Fixed (adding layers adds params) |
| **Loop 2**: H-cycle reps | `T_H` | Working memory iterations per H-cycle | ??? | NOT EXPLORED AT ALL |
| **Loop 3**: L-H-cycle reps | `T_H_T_L` | Answer accumulation iterations | 2 (in asym) / 3 (in grid best) | **Partially explored — NEEDS FULL ABLATION** |
| **Loop 4**: Outer refinement | `T_refine` / `eval_mult` | Test-time compute scaling | 2× at eval | **Under-explored — NEEDS ABLATION** |

### 2.2 Architecture Findings
- **z_H/z_L hierarchy is essential**: H=1 (no hierarchy) → 80-95 FID regardless of training time (vs H=3 → 36-45 FID)
- **Asymmetric recursion**: Front-loading T_L in the first H-cycle (6+1) beats symmetric distribution (3+3) by ~3 FID
- **Recursion history memory**: Lightweight EMA of z_L states (α=0.05) provides ~1 FID improvement at zero parameter cost
- **Full BPTT is critical**: TRM's truncated BPTT is catastrophically wrong for diffusion (259→55 FID improvement)

### 2.3 Training Findings
- **Hybrid L1+L2 loss with luminance channel weighting**: ~4 FID improvement over pure L1
- **Classifier-free guidance with rescaled output**: rCFG scale 2.5 with p_uncond=0.1
- **Simplicity principle**: Parameter-free innovations beat learned ones under fixed training budget

### 2.4 Scaling Laws
- **H×L compute grid**: Systematic ablation at 1hr and 3hr showing identical optimal allocation
- **Training time scaling**: 1hr→3hr→20hr showing diminishing returns

---

## 3. Experiments Needed

### 3.1 CRITICAL — Four-Loop Ablation (NEW)

**Loop 1: L-module depth (l_layers)**
Currently fixed at 1. We should test l_layers=[1, 2, 4] to see if deeper (non-shared) blocks help.
Note: This changes parameter count, so we need to control for that (e.g., reduce n_embd to match 13.6M).

| Experiment | l_layers | n_embd (to match params) | Status |
|------------|----------|--------------------------|--------|
| l_layers=1, n_embd=768 | 1 | 768 (13.6M) | Current best |
| l_layers=2, n_embd=544 | 2 | 544 (~13.6M) | **TODO** |
| l_layers=4, n_embd=384 | 4 | 384 (~13.6M) | **TODO** |

**Loop 2: T_L (L-cycle repetitions) — detailed sweep**
We've tested this extensively but need clean final numbers.

| T_L (first H) | T_L (second H) | FID (1hr) | FID (20hr) | Status |
|----------------|----------------|-----------|------------|--------|
| 1 | 1 | ~50 | TODO | **RUNNING (H=2,L=1)** |
| 3 | 3 | ~48 | TODO | **RUNNING (H=2,L=3)** |
| 6 | 1 (asymmetric) | ~41 | TODO | **RUNNING (5 seeds)** |
| 9 | 9 | ~67 | — | Done (too slow) |

**Loop 3: T_H (H-cycle repetitions) — NEEDS FULL ABLATION**
From the H×L grid we have H=[1,3,9] but we need finer resolution.

| T_H | T_L | Total block apps | FID (1hr) | FID (3hr) | FID (20hr) | Status |
|-----|-----|------------------|-----------|-----------|------------|--------|
| 1 | 3 | 9 | 87.2 | 94.5 | — | Done |
| 2 | 3 | 12 | — | — | TODO | **RUNNING (default)** |
| 3 | 1 | 6 | 45.2 | 36.5 | TODO | **RUNNING** |
| 3 | 3 | 15 | 48.2 | 38.5 | — | Done |
| 4 | 1 | 8 | 45.6 | — | **TODO** | Need to run |
| 5 | 1 | 10 | 46.4 | — | **TODO** | Need to run |
| 6 | 1 | 12 | 49.0 | — | **TODO** | Need to run |
| 9 | 1 | 18 | 53.3 | 43.0 | — | Done |

**Loop 4: T_refine (outer refinement / eval-time compute)**
We've used eval_mult=2 (TTA 2×). Need systematic ablation.

| T_refine (eval) | T_refine (train) | FID | Status |
|-----------------|-------------------|-----|--------|
| 1 | 1 | **TODO** | Need to run |
| 2 | 1 | Current default | **RUNNING** |
| 3 | 1 | ~45.6 (1hr) | Done (partial) |
| 4 | 1 | **TODO** | Need to run |
| 2 | 2 | **TODO** | Need to run |

### 3.2 CRITICAL — Scaling Laws

**Model size scaling (fixed architecture ratios)**

| n_embd | Params | FID (1hr) | FID (20hr) | Status |
|--------|--------|-----------|------------|--------|
| 256 | ~1.6M | **TODO** | **TODO** | Need to run |
| 384 | ~3.5M | **TODO** | **TODO** | Need to run |
| 512 | ~6.3M | **TODO** | **TODO** | Need to run |
| 768 | 13.6M | ~41 | RUNNING | Current |
| 1024 | ~24M | **TODO** | **TODO** | Need to run |

**Training iteration scaling (fixed architecture)**
Use best config, measure FID at checkpoints: 10min, 30min, 1hr, 3hr, 6hr, 12hr, 20hr.
→ This can be extracted from the 20hr wandb logs! Just read intermediate FID evaluations or loss curves.

**Dataset size scaling**
Train on subsets of ImageNet: 10%, 25%, 50%, 100%.

| Dataset fraction | # images | FID (1hr) | Status |
|------------------|----------|-----------|--------|
| 10% | ~128K | **TODO** | Need to run |
| 25% | ~320K | **TODO** | Need to run |
| 50% | ~640K | **TODO** | Need to run |
| 100% | ~1.28M | ~41 | Current |

### 3.3 CRITICAL — Baseline Comparisons

| Model | Params | Architecture | FID (20hr) | Status |
|-------|--------|--------------|------------|--------|
| Non-recursive (1 block pass) | 13.6M | No recursion | RUNNING | **GPU 5** |
| Symmetric H=2, L=3 | 13.6M | TRM default | RUNNING | **GPU 6** |
| Symmetric H=3, L=1 | 13.6M | Grid optimal | RUNNING | **GPU 7** |
| **Ours (Asym 6+1 + history)** | 13.6M | Full config | RUNNING | **GPUs 0-4** |
| Standard DiT (unique layers) | 13.6M | 4 layers @ 384d | **TODO** | Need to build |

### 3.4 IMPORTANT — Ablation of Individual Components

Starting from the best config, remove one component at a time:

| Ablation | FID (20hr) | Δ FID | Status |
|----------|------------|-------|--------|
| Full best config | RUNNING | 0 | Baseline |
| − Asymmetric → symmetric 3+3 | RUNNING (GPU 6) | ? | **RUNNING** |
| − History memory | **TODO** | ? | Need to run |
| − Hybrid loss → pure L1 | **TODO** | ? | Need to run |
| − Channel weighting | **TODO** | ? | Need to run |
| − Rescaled CFG → standard CFG | **TODO** | ? | Need to run |
| − CFG entirely (no guidance) | **TODO** | ? | Need to run |
| − Dual-t → single-t | **TODO** | ? | Need to run |
| − Full BPTT → truncated BPTT | **TODO** | ? | Need to run |

### 3.5 NICE-TO-HAVE — Additional Analysis

- [ ] Generated sample images (cherry-picked and random) for qualitative evaluation
- [ ] Attention map visualization across recursion steps
- [ ] z_H and z_L trajectory analysis (how do they evolve through recursion?)
- [ ] Per-class FID breakdown (which classes benefit most from recursion?)
- [ ] Comparison with published small-model diffusion results from literature
- [ ] Training loss curves comparison across configs
- [ ] Wall-clock efficiency analysis (FID per GPU-hour)

---

## 4. Paper Structure (Draft Outline)

### Abstract (~150 words)
We adapt TinyRecursiveModels (TRM) — a shared-weight recursive transformer with two-level latent hierarchy — for image generation via flow matching. Through 860+ systematic experiments on ImageNet 64×64, we discover that the allocation of compute across recursive cycles is critical: asymmetric recursion (front-loading working memory refinement) consistently outperforms symmetric allocation. Our 13.6M parameter model achieves FID [XX] with 20 hours of A100 training, demonstrating that recursive weight sharing is a viable alternative to the standard deep-and-wide paradigm in diffusion models.

### 1. Introduction
- Diffusion models are large and expensive
- TRM showed recursive shared-weight models work for reasoning
- Can the same principle work for generation?
- Key question: how should compute be allocated across recursive cycles?

### 2. Background
- 2.1 Flow Matching (Lipman et al., 2022)
- 2.2 Diffusion Transformers (DiT, Peebles & Xie, 2023)
- 2.3 TinyRecursiveModels (shared-weight recursion, z_H/z_L hierarchy)
- 2.4 Classifier-Free Guidance (Ho & Salimans, 2022)

### 3. Method: Recursive Diffusion Transformer
- 3.1 Architecture: shared-weight block + two-level hierarchy
- 3.2 The four loops of recursion (T_L, T_H, T_HL, T_refine)
- 3.3 Asymmetric recursion and history memory
- 3.4 Training: hybrid loss, CFG, schedule

### 4. Experiments
- 4.1 Setup (ImageNet 64×64, evaluation protocol, hardware)
- 4.2 H×L compute allocation grid (1hr and 3hr)
- 4.3 Four-loop ablation
- 4.4 Component ablation (Table: remove each component)
- 4.5 Scaling laws (model size, training time, data size)
- 4.6 Comparison with baselines

### 5. Analysis
- 5.1 Why asymmetric recursion works (gradient flow analysis)
- 5.2 The role of z_H hierarchy (information aggregation)
- 5.3 Recursion history as implicit memory
- 5.4 Simplicity principle: why parameter-free wins

### 6. Related Work
- Efficient diffusion (distillation, pruning, quantization)
- Recursive/iterative models (Universal Transformers, DEQ, PonderNet)
- Weight sharing in transformers (ALBERT, etc.)
- Adaptive computation (ACT, early exit)

### 7. Conclusion

---

## 5. Experiment Priority & Schedule

### Phase 1: RUNNING NOW (20hr final runs)
- [x] 5 seeds of best config (GPUs 0-4)
- [x] Non-recursive baseline (GPU 5)
- [x] Symmetric H=2,L=3 (GPU 6)
- [x] Symmetric H=3,L=1 (GPU 7)
**ETA: ~20 hours from launch**

### Phase 2: Component Ablations (after Phase 1)
Priority: each removes ONE component from best config
- [ ] − History memory (asym 6+1 without z_mem)
- [ ] − Hybrid loss → pure L1+cosine
- [ ] − Channel weighting → uniform weights
- [ ] − CFG entirely (p_uncond=0, cfg_scale=1)
- [ ] − Dual-t → single-t
- [ ] − Rescaled CFG → standard CFG
- [ ] − Full BPTT → truncated BPTT (only backprop through last H-cycle)
**8 runs × 20hr = 1 batch**

### Phase 3: Four-Loop Detailed Ablation
- [ ] T_H sweep: H=[2,3,4,5] with L=1 (4 runs, 20hr each)
- [ ] T_refine sweep: eval_mult=[1,2,3,4] (4 runs, 20hr each)
- [ ] l_layers sweep: [1,2,4] with matched params (3 runs, 20hr each)
**~11 runs = 2 batches**

### Phase 4: Scaling Laws
- [ ] Model size: n_embd=[256,384,512,1024] (4 runs, 20hr each)
- [ ] Dataset size: [10%,25%,50%] of ImageNet (3 runs, 20hr each)
- [ ] Training time: extract from 20hr wandb logs (no extra runs needed!)
**7 runs = 1 batch**

### Phase 5: Polish
- [ ] Standard DiT baseline (unique layers, matched params)
- [ ] Generate sample images for paper figures
- [ ] Attention visualization
- [ ] Final paper writing

---

## 6. Timeline

| Day | Phase | Runs | GPU-hours |
|-----|-------|------|-----------|
| Day 0 (now) | Phase 1 launched | 8 × 20hr | 160 |
| Day 1 | Phase 2: Component ablations | 8 × 20hr | 160 |
| Day 2 | Phase 3: Four-loop ablation (batch 1) | 8 × 20hr | 160 |
| Day 3 | Phase 3 (batch 2) + Phase 4 | 8 × 20hr | 160 |
| Day 4 | Phase 5: Polish + sample generation | 8 × varies | ~80 |
| Days 5-7 | Paper writing | — | — |

**Total: ~720 GPU-hours on 8× A100-80GB**

---

## 7. Key Figures for Paper

1. **H×L heatmap** (1hr vs 3hr vs 20hr) — the centerpiece figure
2. **Four-loop ablation bar chart** — showing marginal value of each loop
3. **Component ablation waterfall** — starting from baseline, adding each component
4. **Scaling laws** — log-log plots of FID vs model size, training time, data size
5. **Generated samples** — grid of class-conditional 64×64 images
6. **Architecture diagram** — showing the four loops and z_H/z_L flow
7. **Training curves** — loss and FID over time for key configs

---

## 8. Literature to Cite

### Core
- Lipman et al. (2022) — Flow Matching
- Peebles & Xie (2023) — DiT
- Ho & Salimans (2022) — Classifier-Free Guidance
- TinyRecursiveModels (TRM) — the base architecture
- Lin et al. (2024) — Rescaled CFG / Common Diffusion Noise Schedules

### Recursive/Iterative Models
- Dehghani et al. (2018) — Universal Transformers
- Bai et al. (2019) — Deep Equilibrium Models (DEQ)
- Graves (2016) — Adaptive Computation Time
- Banino et al. (2021) — PonderNet

### Efficient Diffusion
- Song et al. (2023) — Consistency Models
- Salimans & Ho (2022) — Progressive Distillation
- Li et al. (2023) — SnapFusion

### Weight Sharing
- Lan et al. (2020) — ALBERT (cross-layer sharing)
- Reid et al. (2021) — Subformer

### Bio-Inspired
- Oja (1982) — Hebbian learning rule
- Graves et al. (2014) — Neural Turing Machines
- Friston (2005) — Predictive Coding
