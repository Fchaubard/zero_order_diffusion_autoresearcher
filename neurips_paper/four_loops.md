# The Four Loops of TRM-Diffusion

## Loop Taxonomy

```python
# Pseudocode showing all 4 loops
for t_refine in range(T_refine):                    # LOOP 4: Outer refinement (eval-time)
    for t_htl in range(T_H_T_L):                    # LOOP 3: H-L macro-cycle repetitions
        for t_l in range(T_L):                      # LOOP 1: L-cycle repetitions
            z_L = shared_block(z_L + z_H + c)       #   L consolidation (working memory)
        for t_h in range(T_H):                      # LOOP 2: H-cycle repetitions
            z_H = shared_block(z_H + z_L)           #   H consolidation (answer accumulator)
```

The nesting order is critical: within each macro-cycle (T_H_T_L), the model first runs T_L rounds of L-consolidation (refining working memory z_L given the current answer z_H), then runs T_H rounds of H-consolidation (updating the answer z_H given the refined working memory z_L). This "refine-then-consolidate" pattern repeats T_H_T_L times, and the whole thing can optionally repeat T_refine times at eval.

## Definitions

### Loop 1: T_L — L-cycle repetitions (working memory refinement)
**What it controls:** How many times z_L is updated consecutively before z_H gets to consolidate.
**Analogy:** "Thinking steps" — how long the model ponders before committing an update to the answer.
**Current best:** Asymmetric — 6 in the first macro-cycle, 1 in the second.
**Our finding:** More consecutive L-updates allow deeper working memory refinement before consolidation. Front-loading (more L-cycles early, fewer late) is better than uniform distribution.
**Status:** WELL EXPLORED (1hr, 3hr ablations, asymmetric discovery)

### Loop 2: T_H — H-cycle repetitions (answer consolidation)
**What it controls:** How many times z_H is updated consecutively after each round of L-refinement.
**Analogy:** "Reflection steps" — how many times the model re-reads its working memory before moving on.
**Current best:** T_H = 1 (single consolidation per macro-cycle — always!)
**Critical observation:** This has NEVER been explored in TRM research! We've always assumed T_H=1.
**Status:** **COMPLETELY UNEXPLORED — CRITICAL GAP**
**Key question:** Does the answer accumulator benefit from multiple consecutive reads of the working memory? Or is one pass sufficient?

### Loop 3: T_H_T_L — H-L macro-cycle repetitions (the big loop)
**What it controls:** How many complete rounds of "L refines → H consolidates" occur.
**Analogy:** "Drafts" — how many complete think-then-commit cycles the model performs.
**Current best:** T_H_T_L = 2 in our asymmetric config (one heavy round + one light round), or T_H_T_L = 3 in the ablation grid best (H=3, L=1).
**Status:** PARTIALLY EXPLORED (H×L grid covers this), but confounded with T_L in our asymmetric experiments.

### Loop 4: T_refine — Outer refinement / test-time compute
**What it controls:** Additional full recursive passes at evaluation time that are NOT used during training. Pure test-time compute scaling.
**Analogy:** "Proofreading passes" — extra review cycles applied only when generating final output.
**Current best:** eval_mult = 2 (applies to the last macro-cycle's L and reverse passes)
**Status:** PARTIALLY EXPLORED (tested 1×, 2×, 3× briefly), NEEDS SYSTEMATIC ABLATION
**Key question:** Can we get "free" FID improvement by spending more compute at inference?

## Mapping to Current Code

In our current best config (asymmetric 6+1), the loops map as:
```
T_refine = 1 (eval_mult=2 applies within the last macro-cycle, not a full outer loop)
T_H_T_L = 2 macro-cycles:
  Macro-cycle 1: T_L=6 (with history memory), T_H=1
  Macro-cycle 2: T_L=1×eval_mult, T_H=1, then reverse T_L=1×eval_mult
```

In the H×L ablation grid, the mapping is:
```
"H" parameter = T_H_T_L (number of macro-cycles)
"L" parameter = T_L (L-reps per macro-cycle, uniform across all)
T_H = 1 (always, never tested > 1)
T_refine = 1
```

## What We Know vs. What We Need

| Loop | Variable | Explored? | Finding | Gap |
|------|----------|-----------|---------|-----|
| Loop 1 | T_L | Extensively | Asymmetric 6+1 > symmetric 3+3 | Need 20hr final numbers |
| Loop 2 | T_H | **NOT AT ALL** | Always fixed at 1 | **CRITICAL GAP** — need T_H=[1,2,3,4] |
| Loop 3 | T_H_T_L | Partially | 2-3 macro-cycles optimal | Need finer resolution at 20hr |
| Loop 4 | T_refine | Partially | eval_mult=2 default | Need [1,2,3,4,5] sweep |

## Proposed Ablation Grid

### Clean T_L ablation (holding T_H=1, T_H_T_L=3 fixed)
Already done via H×L grid at L=[1,3,9] with H=3.
Need 20hr runs for final numbers.

### Clean T_H ablation (holding T_L=1, T_H_T_L=3 fixed) — **THE KEY NEW EXPERIMENT**
Each macro-cycle does T_L=1 L-pass, then T_H H-passes.

| T_H | T_L | T_H_T_L | Block apps per macro-cycle | Total block apps | Status |
|-----|-----|---------|---------------------------|------------------|--------|
| 1 | 1 | 3 | 1L + 1H = 2 | 6 | Done (H=3,L=1) |
| 2 | 1 | 3 | 1L + 2H = 3 | 9 | **TODO** |
| 3 | 1 | 3 | 1L + 3H = 4 | 12 | **TODO** |
| 4 | 1 | 3 | 1L + 4H = 5 | 15 | **TODO** |

This directly answers: "should the answer accumulator re-read working memory multiple times?"

### Clean T_H_T_L ablation (holding T_L=1, T_H=1 fixed)
| T_H_T_L | Block apps | FID (3hr) | FID (20hr) | Status |
|---------|------------|-----------|------------|--------|
| 1 | 2 | ~95 | — | Done (terrible) |
| 2 | 4 | ~50 (est) | **TODO** | |
| 3 | 6 | 36.5 | RUNNING | Phase 1 (H=3,L=1) |
| 4 | 8 | ~46 (1hr) | **TODO** | |
| 5 | 10 | ~46 (1hr) | **TODO** | |
| 6 | 12 | ~49 (1hr) | — | Done |
| 9 | 18 | 43.0 | — | Done |

### Clean T_refine ablation (best architecture, 20hr)
| T_refine (eval_mult) | Extra block apps at eval | FID | Status |
|----------------------|--------------------------|-----|--------|
| 1 | 0 | ??? | **TODO** |
| 2 | +2 | Current default | RUNNING (Phase 1) |
| 3 | +4 | ~45.6 (1hr) | Partial |
| 4 | +6 | ??? | **TODO** |
| 5 | +8 | ??? | **TODO** |

## Scaling Laws to Measure

For each loop, we want to plot: **FID vs. total compute (block applications per forward pass)** at fixed training budget.

This reveals:
1. **Which loop has the best FID/compute ratio** (steepest improvement per additional block application)
2. **Where each loop saturates** (diminishing returns)
3. **Whether loops are complementary or redundant** (does T_H=2 + T_L=2 beat T_H=1 + T_L=4 at same total compute?)

### Cross-loop compute efficiency (THE KILLER TABLE)
At fixed total block applications ≈ 12:
| Config | T_L | T_H | T_H_T_L | Total apps | FID | How compute is spent |
|--------|-----|-----|---------|------------|-----|---------------------|
| All in T_L | 6 | 1 | 2 | 14 | ~41 | Deep working memory |
| All in T_H | 1 | 4 | 3 | 15 | ??? | Deep answer consolidation |
| All in T_H_T_L | 1 | 1 | 6 | 12 | ~49 | Many short cycles |
| Balanced | 2 | 2 | 3 | 12 | ??? | Even distribution |
| Our asymmetric | 6+1 | 1 | 2 | ~12 | ~41 | Front-loaded L |

This table directly shows that **compute allocation strategy matters more than total compute** — the central thesis of the paper.

## Connection to TRM Theory

In TRM's original formulation:
- **z_L** is "working memory" — a scratchpad for intermediate computation
- **z_H** is "answer accumulation" — the running answer that persists across cycles

Our findings map onto this beautifully:
1. **T_L matters most** → Working memory refinement is the bottleneck
2. **T_H=1 seems sufficient** → The answer only needs one look at the refined working memory
3. **T_H_T_L ≥ 2 is essential** → Multiple refine-consolidate cycles are needed
4. **Asymmetric T_L is optimal** → Early cycles need deep thinking, later cycles just need a quick check

This suggests that in TRM-style models, **the working memory (z_L) is doing most of the "work," while the answer accumulator (z_H) just needs to periodically "read" the result.** The optimal strategy is to let z_L think deeply first, then have z_H read once, then let z_L do a quick check, then z_H reads again.
