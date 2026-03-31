# The Four Loops of TRM-Diffusion

## Loop Taxonomy

```python
# Pseudocode showing all 4 loops
for t_refine in range(T_refine):                    # LOOP 4: Outer refinement (eval-time scaling)
    for t_hl in range(T_HL):                         # LOOP 3: Joint H-L macro-cycles
        for t_h in range(T_H):                       # LOOP 2: H-module consecutive repetitions
            z_H = shared_block(z_H, z_L, c)
        for t_l in range(T_L):                       # LOOP 1: L-module consecutive repetitions
            z_L = shared_block(z_L, z_H + input, c)
```

## Definitions

### Loop 1: T_L — L-module consecutive repetitions
**What it controls:** How many times z_L (working memory) is updated consecutively before z_H sees the result.
**Current best:** Asymmetric — 6 in the first macro-cycle, 1 in the second.
**Our finding:** More consecutive L-updates allow deeper working memory refinement before consolidation. Front-loading is better than uniform distribution.
**Status:** WELL EXPLORED (1hr, 3hr ablations)

### Loop 2: T_H — H-module consecutive repetitions
**What it controls:** How many times z_H (answer accumulator) is updated consecutively within each macro-cycle.
**Current best:** T_H = 1 (single z_H consolidation per macro-cycle)
**Your note:** This has NOT been explored in TRM research!
**Status:** UNDER-EXPLORED — NEEDS ABLATION
**Key question:** Does repeating z_H updates (T_H > 1) within a single macro-cycle improve answer quality? Or is one consolidation pass sufficient?

### Loop 3: T_HL — Joint H-L macro-cycles
**What it controls:** How many complete rounds of "L refines → H consolidates" occur.
**Current best:** T_HL = 2 (in our asymmetric config: one heavy round of 6L+1H, one light round of 1L+1H+1reverse)
**Equivalently:** In the symmetric H×L grid, H=3 means T_HL=3 macro-cycles (each with L=1 or L=3 inner iterations).
**Status:** PARTIALLY EXPLORED (H×L grid), but not cleanly separated from T_L and T_H.

### Loop 4: T_refine — Outer refinement / test-time compute
**What it controls:** Additional recursive passes at evaluation time that are NOT used during training. A form of test-time compute scaling.
**Current best:** eval_mult = 2 (2× more L and reverse passes at eval)
**Status:** PARTIALLY EXPLORED (tested 1x, 2x, 3x briefly), NEEDS SYSTEMATIC ABLATION
**Key question:** Can we get "free" FID improvement by spending more compute at inference?

## Mapping to Current Code

In our current best config (asymmetric 6+1), the loops are:
```
T_HL = 2 macro-cycles:
  Macro-cycle 1: T_L=6 (with history), T_H=1
  Macro-cycle 2: T_L=1*eval_mult, T_H=1, then T_L=1*eval_mult reverse
T_refine = 1 (eval_mult=2 is within the last macro-cycle, not a full outer loop)
```

In the H×L grid, the mapping is:
```
H parameter = T_HL (number of macro-cycles)
L parameter = T_L (L-reps per macro-cycle, uniform across all macro-cycles)
T_H = 1 (always, never tested > 1)
T_refine = 1 (eval_mult handles it within the last macro-cycle)
```

## What We Know vs. What We Need

| Loop | Explored? | Finding | Gap |
|------|-----------|---------|-----|
| T_L | Extensively | Asymmetric 6+1 > symmetric 3+3 | Need 20hr final numbers |
| T_H | **NOT AT ALL** | Always 1 | **CRITICAL GAP** — need T_H=[1,2,3,4] |
| T_HL | Partially | 2-3 macro-cycles optimal | Need finer resolution at 20hr |
| T_refine | Partially | eval_mult=2 default | Need [1,2,3,4,5] sweep |

## Proposed Ablation Grid

### Clean T_H ablation (holding T_L=1, T_HL=3 fixed)
| T_H | T_L | T_HL | Total block apps | Predicted FID | Status |
|-----|-----|------|------------------|---------------|--------|
| 1 | 1 | 3 | 6 | ~36 (3hr) | Done |
| 2 | 1 | 3 | 9 | ??? | **TODO** |
| 3 | 1 | 3 | 12 | ??? | **TODO** |
| 4 | 1 | 3 | 15 | ??? | **TODO** |

### Clean T_HL ablation (holding T_L=1, T_H=1 fixed)
| T_HL | T_L | T_H | Total block apps | FID (3hr) | Status |
|------|-----|-----|------------------|-----------|--------|
| 1 | 1 | 1 | 2 | ~95 | Approx (H=1,L=1) |
| 2 | 1 | 1 | 4 | ~50 (est) | **TODO** |
| 3 | 1 | 1 | 6 | 36.5 | Done |
| 4 | 1 | 1 | 8 | ~46 (1hr) | Done (H=4,L=1) |
| 5 | 1 | 1 | 10 | ~46 (1hr) | Done (H=5,L=1) |
| 6 | 1 | 1 | 12 | ~49 (1hr) | Done (H=6,L=1) |
| 9 | 1 | 1 | 18 | 43.0 (3hr) | Done |

### Clean T_refine ablation (holding architecture fixed at best config)
| T_refine | Extra block apps at eval | FID | Status |
|----------|--------------------------|-----|--------|
| 1 (eval_mult=1) | 0 | ??? | **TODO** |
| 2 (eval_mult=2) | +2 | Current default | RUNNING |
| 3 (eval_mult=3) | +4 | ~45.6 (1hr) | Done |
| 4 (eval_mult=4) | +6 | ??? | **TODO** |
| 5 (eval_mult=5) | +8 | ??? | **TODO** |

## Scaling Laws to Measure

For each loop, we want to plot: **FID vs. total compute (block applications)** at fixed training budget.

This reveals:
1. **Which loop has the best FID/compute ratio** (steepest improvement per additional block app)
2. **Where each loop saturates** (diminishing returns)
3. **Whether loops are complementary or redundant** (does combining T_H=2 with T_L=2 beat T_H=1 with T_L=4 at same total compute?)

### Cross-loop compute efficiency
At fixed total block applications = 12:
| Config | T_L | T_H | T_HL | FID |
|--------|-----|-----|------|-----|
| All in T_L | 12 | 1 | 1 | ??? |
| All in T_H | 1 | 12 | 1 | ??? |
| All in T_HL | 1 | 1 | 6 | ~49 (H=6,L=1) |
| Balanced | 2 | 2 | 3 | ??? |
| Our asymmetric | 6+1 | 1 | 2 | ~41 |

This table would be a KILLER figure for the paper — it directly shows that compute allocation strategy matters more than total compute.
