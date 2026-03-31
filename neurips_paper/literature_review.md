# Literature Review — Recursive Diffusion Transformers

## 1. Diffusion Models & Flow Matching

### Flow Matching
- **Lipman et al. (2022)** "Flow Matching for Generative Modeling" — Our training framework. OT conditional paths, velocity prediction, Euler ODE sampling.
- **Liu et al. (2022)** "Flow Straight and Fast" — Rectified flows, related optimal transport formulation.
- **Albergo & Vanden-Eijnden (2022)** "Building Normalizing Flows with Stochastic Interpolants" — Theoretical foundation.

### Diffusion Transformers
- **Peebles & Xie (2023)** "Scalable Diffusion Models with Transformers (DiT)" — The standard architecture we compare against. AdaLN-Zero conditioning, patch embedding. Our shared block is architecturally identical to a DiT block.
- **Esser et al. (2024)** "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)" — Logit-normal timestep sampling (we tested, neutral for us). MMDiT architecture.

### Classifier-Free Guidance
- **Ho & Salimans (2022)** "Classifier-Free Diffusion Guidance" — Core technique, ~6 FID improvement in our experiments.
- **Lin et al. (2024)** "Common Diffusion Noise Schedules and Sample Steps are Flawed" — Rescaled CFG. We found rescaled output normalization gives ~1 FID improvement.

### Efficient Diffusion
- **Song et al. (2023)** "Consistency Models" — Distillation for fast sampling. Related but orthogonal (we focus on training efficiency).
- **Salimans & Ho (2022)** "Progressive Distillation for Fast Sampling" — Step reduction.
- **Li et al. (2023)** "SnapFusion" — Efficient diffusion on mobile. Relevant comparison point for small models.

## 2. Recursive & Iterative Transformer Models

### Universal Transformers
- **Dehghani et al. (2018)** "Universal Transformers" — Weight sharing across layers with adaptive halting. The closest predecessor to TRM-style recursion. Key finding: shared weights + adaptive depth outperforms fixed-depth models on algorithmic tasks. **Our work extends this to generative modeling.**

### Deep Equilibrium Models
- **Bai et al. (2019)** "Deep Equilibrium Models" — Implicit layers that find fixed points. Related to infinite-depth weight sharing. Memory-efficient via implicit differentiation. **Relevant comparison: DEQ is the limit of infinite recursion with shared weights.**
- **Bai et al. (2020)** "Multiscale Deep Equilibrium Models" — Multiscale extension. Our z_H/z_L hierarchy is conceptually related to multiscale DEQ.

### Adaptive Computation
- **Graves (2016)** "Adaptive Computation Time for Recurrent Neural Networks" — ACT mechanism with halting probability. Directly inspired our (unsuccessful) stochastic exit experiment. **Key insight: ACT is hard to train for diffusion because every ODE step needs consistent depth.**
- **Banino et al. (2021)** "PonderNet: Learning to Ponder" — Improved ACT with geometric prior. Could be revisited for diffusion.

### TinyRecursiveModels (TRM)
- **[TRM paper]** — The base architecture. Shared-weight L-level blocks, z_H/z_L two-level hierarchy, truncated BPTT. **Our key finding: truncated BPTT is catastrophically wrong for diffusion; full BPTT is essential.**
- Key TRM insights that transfer: β₂=0.95 optimizer, shared weights, warm-start z_H from input.
- Key TRM insights that DON'T transfer: gradient truncation, symmetric recursion depth.

## 3. Weight Sharing in Transformers

- **Lan et al. (2020)** "ALBERT" — Cross-layer parameter sharing in BERT. Found that sharing all layers gives competitive performance at much lower parameter count. **Our model is the generative analog of ALBERT.**
- **Reid et al. (2021)** "Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers" — Sandwich-style sharing in language models.
- **Takase & Kiyono (2023)** "Lessons on Parameter Sharing across Layers in Transformers" — Systematic study of sharing strategies. Found that sharing is most beneficial when model is small relative to data.

## 4. Biological Inspiration

### Predictive Coding
- **Rao & Ballard (1999)** "Predictive coding in the visual cortex" — Hierarchical prediction errors. Our z_H/z_L hierarchy mirrors cortical layers with top-down predictions (z_H→z_L) and bottom-up refinement (z_L→z_H).
- **Friston (2005)** "A theory of cortical responses" — Free energy principle. The recursive refinement in our model can be interpreted as variational inference with iterative message passing.

### Hebbian Learning
- **Oja (1982)** "A simplified neuron model as a principal component analyzer" — Oja's rule. We tested Hebbian gating (b82) — it crashed, but the principle of correlation-gated updates is worth noting.

### Neural Turing Machines
- **Graves et al. (2014)** "Neural Turing Machines" — External memory for neural networks. Our recursion history memory (EMA of z_L states) is a lightweight analog — the model maintains a running memory of its own computation trajectory.

## 5. Closely Related Work (Need to Carefully Position Against)

### Recurrent Interface Networks (RIN)
- **Jabri et al. (2022)** "Scalable Adaptive Computation for Iterative Generation" — Iterative refinement with a recurrent interface. Very similar concept to our approach but different architecture. **MUST CITE AND DIFFERENTIATE.**

### DART
- **[Apple, 2025]** "Denoising Autoregressive Transformer" — Spatial ordering in denoising. We tested a DART-inspired approach (b82 idea) but didn't pursue it.

### Recurrent Diffusion
- **Chen et al. (2024)** "Recurrent Diffusion" — If this exists, we need to cite it. Search for any paper combining recurrence with diffusion.

## 6. Gaps in Literature Our Paper Fills

1. **No systematic study of recursion depth allocation in diffusion**: The H×L ablation grid and four-loop framework are novel.
2. **No asymmetric recursion for diffusion**: The front-loading insight is new.
3. **No study of TRM-style hierarchy for generation**: z_H/z_L for diffusion is new.
4. **No scaling law for recursive compute in diffusion**: Our compute allocation analysis is new.
5. **Simplicity principle for recursive models**: The finding that parameter-free innovations beat learned ones under short training budgets is a useful meta-result.

## 7. Papers to Search For (TODO)

- [ ] Search for "recurrent diffusion" or "iterative diffusion transformer"
- [ ] Search for "weight sharing diffusion"
- [ ] Search for "recursive image generation"
- [ ] Check if RIN (Jabri et al.) has been applied to flow matching
- [ ] Check Universal Transformer follow-ups in vision
- [ ] Search for "adaptive computation diffusion"
