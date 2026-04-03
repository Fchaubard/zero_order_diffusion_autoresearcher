# Experiment Tracker — NeurIPS Submission

## Status Key
- RUNNING = currently executing on GPU
- DONE = completed, results logged
- TODO = planned, not yet launched
- BLOCKED = waiting on dependency

---

## Phase 1: Final 20hr Runs (RUNNING)

| # | GPU | wandb_name | Config | Seed | Status | FID |
|---|-----|------------|--------|------|--------|-----|
| 1 | 0 | final_best_seed42 | Asym 6+1 + hist + hybrid+chw + rCFG 2.5 + wd20 | 42 | RUNNING | — |
| 2 | 1 | final_best_seed13 | Same | 13 | RUNNING | — |
| 3 | 2 | final_best_seed73 | Same | 73 | RUNNING | — |
| 4 | 3 | final_best_seed137 | Same | 137 | RUNNING | — |
| 5 | 4 | final_best_seed199 | Same | 199 | RUNNING | — |
| 6 | 5 | final_nonrecursive_baseline | 1 block pass, no hierarchy | 42 | RUNNING | — |
| 7 | 6 | final_symmetric_h2l3 | H=2, L=3 symmetric + best loss | 42 | RUNNING | — |
| 8 | 7 | final_symmetric_h3l1 | H=3, L=1 symmetric + best loss | 42 | RUNNING | — |

**ETA:** ~20hrs from 2026-03-31 launch

---

## Phase 2: Component Ablations (TODO — launch after Phase 1)

Each removes ONE component from the best config. All 20hr, seed=42.

| # | Component removed | What changes | Status | FID |
|---|-------------------|--------------|--------|-----|
| 9 | − History memory | Remove z_mem EMA | TODO | — |
| 10 | − Hybrid loss | Pure L1 (no L2 component) | TODO | — |
| 11 | − Channel weighting | Uniform channel weights | TODO | — |
| 12 | − CFG entirely | p_uncond=0, cfg_scale=1 | TODO | — |
| 13 | − Rescaled CFG | Standard CFG (no std rescaling) | TODO | — |
| 14 | − Dual-t loss | Single timestep per step | TODO | — |
| 15 | − Velocity clamping | Remove magnitude clamping | TODO | — |
| 16 | − Full BPTT | Truncated BPTT (only last H-cycle) | TODO | — |

---

## Phase 3: Four-Loop Ablation (TODO)

### 3a. T_H sweep (L=1, T_HL=3 fixed, 20hr)
| # | T_H | T_L | T_HL | Block apps | Status | FID |
|---|-----|-----|------|------------|--------|-----|
| 17 | 1 | 1 | 3 | 6 | TODO | — |
| 18 | 2 | 1 | 3 | 9 | TODO | — |
| 19 | 3 | 1 | 3 | 12 | TODO | — |
| 20 | 4 | 1 | 3 | 15 | TODO | — |

### 3b. T_refine sweep (best config, 20hr)
| # | eval_mult | Extra apps | Status | FID |
|---|-----------|------------|--------|-----|
| 21 | 1 | 0 | TODO | — |
| 22 | 2 | +2 | RUNNING (Phase 1) | — |
| 23 | 3 | +4 | TODO | — |
| 24 | 4 | +6 | TODO | — |

### 3c. l_layers sweep (param-matched, 20hr)
| # | l_layers | n_embd | Params | Status | FID |
|---|----------|--------|--------|--------|-----|
| 25 | 1 | 768 | 13.6M | RUNNING (Phase 1) | — |
| 26 | 2 | 544 | ~13.6M | TODO | — |
| 27 | 4 | 384 | ~13.6M | TODO | — |

---

## Phase 4: Scaling Laws (TODO)

### 4a. Model size scaling (20hr each)
| # | n_embd | Params | Status | FID |
|---|--------|--------|--------|-----|
| 28 | 256 | ~1.6M | TODO | — |
| 29 | 384 | ~3.5M | TODO | — |
| 30 | 512 | ~6.3M | TODO | — |
| 31 | 768 | 13.6M | RUNNING | — |
| 32 | 1024 | ~24M | TODO | — |

### 4b. Dataset size scaling (20hr each)
| # | Fraction | # images | Status | FID |
|---|----------|----------|--------|-----|
| 33 | 10% | ~128K | TODO | — |
| 34 | 25% | ~320K | TODO | — |
| 35 | 50% | ~640K | TODO | — |
| 36 | 100% | ~1.28M | RUNNING | — |

### 4c. Training time scaling
Extract from 20hr wandb logs — no extra runs needed.
Measure FID at: 1hr, 3hr, 6hr, 12hr, 20hr checkpoints.

---

## Phase 5: Cross-Loop Compute Efficiency (TODO)

Fixed total block applications = 12:
| # | Config | T_L | T_H | T_HL | Status | FID |
|---|--------|-----|-----|------|--------|-----|
| 37 | All L | 6 | 1 | 2 | ≈ RUNNING (asym) | — |
| 38 | All H | 1 | 4 | 3 | TODO | — |
| 39 | All HL | 1 | 1 | 6 | DONE (1hr) | 49.0 |
| 40 | Balanced | 2 | 2 | 3 | TODO | — |

---

## Previously Completed (from 860+ experiments)

### H×L Grid — 1 hour
```
         L=1       L=3       L=9
H=1    84.84     87.18     77.69
H=3    45.16     48.16     66.80
H=9    53.27     71.27     OOM
```

### H×L Grid — 3 hours
```
         L=1       L=3       L=9
H=1    (skip)    94.53     79.69
H=3    36.45     38.49     49.51
H=9    43.01     50.26     72.72
```

### Key 1-hour results
- Best overall: FID 40.31 (hybrid+chw+asym6+1+hist+rCFG2.5+wd20, seed=137)
- Best symmetric: FID 45.16 (H=3, L=1)
- Non-recursive: ~85 FID
- Original baseline: FID 359.63

---

## Total Experiment Count

| Phase | Runs | GPU-hours | Status |
|-------|------|-----------|--------|
| Historical (b1-b91) | 860 | ~860 | DONE |
| Phase 1 | 8 | 160 | RUNNING |
| Phase 2 | 8 | 160 | TODO |
| Phase 3 | 11 | 220 | TODO |
| Phase 4 | 8 | 160 | TODO |
| Phase 5 | 3 | 60 | TODO |
| **Total** | **~898** | **~1620** | — |
# Best schedule-free: FID 141 (curriculum cap=0.8 + 2-pass eval)
# Contractive training: multi-pass train hurts, 2-pass eval helps. Best FID 138.
