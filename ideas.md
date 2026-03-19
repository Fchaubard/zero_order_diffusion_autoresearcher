# Ideas Log — Zero-Order Diffusion Autoresearcher

Priority queue: highest-confidence untried ideas at the top. Completed/failed ideas archived below.

---

## ============================================================
## PRIORITY QUEUE (untried or running, sorted by confidence)
## ============================================================

---
idea_id: `glce_batch_size_sweep`
Description: Sweep batch size for gl_class_ce w=1. B=1k with InceptionV3 is ~100s/step = only 108 steps in 3h. Smaller batches (B=128, B=256) are much faster through InceptionV3, giving more steps. More steps with noisier gradients vs fewer steps with cleaner gradients. The question is whether step count or gradient quality matters more.
Confidence: 7
Why: gl_class_ce w=1 at B=1k achieved 252.70 FID with only 108 steps. With B=128, each step should be ~10-20s, giving ~500-1000 steps. The InceptionV3 classification signal is class-level (not pixel-level), so smaller batches still cover diverse classes. More steps = more optimization = potentially much better FID.
Time of idea generation: 2026-03-19T00:00:00
Status: Implemented, not tried
HPPs: B=128/256/1000, np=50/100, with and without loss-lr-scale
Time of run start and end: queued (R159)
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `glce_weight_sweep`
Description: Sweep diversity_weight for gl_class_ce. w=1 got 252.70. w=5 still running (loss ~55, very slow). Test w=0.1 (hint of classification), w=0.5, w=2 to find the optimal balance between GL denoising quality and InceptionV3 classification guidance.
Confidence: 6
Why: The balance between GL and CE losses matters. Too little CE (w=0.1) might not break mean prediction. Too much CE (w=5) might overwhelm GL and produce InceptionV3-adversarial images instead of natural images. w=1 worked well as a first try.
Time of idea generation: 2026-03-19T00:00:00
Status: Implemented, not tried
HPPs: diversity_weight=0.1/0.5/1.0/2.0/5.0
Time of run start and end: queued (R159)
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `dc48_llrs_kalman`
Description: Data curriculum (start 48 images) + loss-lr-scale + kalman-grad. Combine the proven data curriculum approach with the champion 1hr features (loss-lr-scale gave 194.57→193.87 on fixed batch). Data curriculum provides the gradually expanding training set, loss-lr-scale adapts LR to loss magnitude, kalman-grad smooths gradient estimates.
Confidence: 6
Why: Data curriculum with 48 initial images shows clear learning (loss 1.29→0.69 at 17% in R156). Loss-lr-scale and kalman-grad are the top two features for fixed-batch SPSA. Combining them should give the best of both worlds. High confidence because each component is proven individually.
Time of idea generation: 2026-03-18T11:00:00
Status: Implemented, not tried
HPPs: `--data-curriculum 48 --loss-lr-scale --kalman-grad --n-perts 100 --total-batch-size 48 --device-batch-size 48 --seed 2 --time-budget 10800` + GL loss + standard curriculum T
Time of run start and end: queued (R158)
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `dc48_cubic_growth`
Description: Data curriculum with cubic growth schedule (pool_size = init + (max-init)*progress^3). Even slower pool expansion than quadratic. At 50% progress, cubic only reaches 12.5% of max vs 25% for quadratic. Gives the model MORE time in the low-noise regime.
Confidence: 5
Why: R156 shows smaller pools = better learning (datacur48 >> datacur5k). Cubic growth keeps the pool small for longer, maximizing the productive early-training phase. The risk is not having enough diversity by end of training.
Time of idea generation: 2026-03-18T11:00:00
Status: Implemented, not tried
HPPs: `--data-curriculum 48 --data-curriculum-growth cubic --n-perts 100 --total-batch-size 48 --device-batch-size 48 --seed 2`
Time of run start and end: queued (R158)
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `dc48_capped_pool`
Description: Data curriculum with capped maximum pool size (2K-5K instead of 50K). The pool grows from 48 to a moderate maximum, then stays fixed. This prevents the late-training noise explosion when the pool approaches full ImageNet size. Tested at max=1K, 2K, 5K.
Confidence: 5
Why: datacur5k in R156 barely learned (loss 1.30→1.26). datacur48 learned well (loss 1.29→0.69) but its pool is still small at 17%. Capping at 2K-5K provides diversity while keeping noise manageable. Essentially a compromise between fixed-batch (48 images, 193 FID) and full-dataset (1.3M images, 317 FID).
Time of idea generation: 2026-03-18T11:00:00
Status: Implemented, not tried
HPPs: `--data-curriculum 48 --data-curriculum-max 1000/2000/5000 --n-perts 100 --total-batch-size 48 --device-batch-size 48`
Time of run start and end: queued (R158)
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `dc200_llrs`
Description: Data curriculum starting from 200 images + loss-lr-scale. datacur200 showed the best end-loss in R156 (0.676 at 17%). Larger initial pool = more diversity from the start, while still being small enough for SPSA to learn.
Confidence: 5
Why: datacur200 outperformed datacur48 in end-loss (0.676 vs 0.685) despite starting from a larger pool. This suggests 200 images is a good balance between diversity and noise. Adding loss-lr-scale should further improve learning.
Time of idea generation: 2026-03-18T11:00:00
Status: Implemented, not tried
HPPs: `--data-curriculum 200 --loss-lr-scale --n-perts 100 --total-batch-size 200 --device-batch-size 200 --seed 2`
Time of run start and end: queued (R158)
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `grad_accumulation_fulldata`
Description: Use SPSA gradient accumulation (--spsa-accum-steps) on full ImageNet. Each SPSA step averages gradients from N independent mini-batches. With accum=4, each step sees 4x more data, reducing batch noise variance by 4x. The key insight: SPSA gradient on a single batch has variance proportional to d=563K. Accumulating over M batches reduces inter-batch noise by 1/M while SPSA noise stays proportional to d. If inter-batch noise dominates (which it does at ~317 FID), accum should help.
Confidence: 4
Why: Accumulation was tested briefly in R142 (--spsa-accum-steps 10) but with B=10K which is very slow. Need to test with moderate B (1K-2K) and moderate accum (2-8) for better throughput. The fundamental problem is that gradient variance from random batches swamps the true gradient signal. Accumulation directly reduces this. Score is moderate because the variance reduction may not be enough — 563K dims is huge.
Time of idea generation: 2026-03-18T10:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `antithetic_batches`
Description: Use antithetic sampling for SPSA batches. Instead of random batches each step, use pairs of batches that are "opposites" in some sense. For example, if batch 1 has classes [0-499], batch 2 has [500-999]. Or use stratified sampling to ensure each batch covers all 1000 classes equally. This reduces inter-batch variance which is the dominant noise source on full ImageNet.
Confidence: 3
Why: The mean prediction problem comes from batch-to-batch variance. If we ensure each batch is a representative mini-ImageNet (all classes), the gradient signal should be cleaner. However, SPSA still has 563K-dimensional noise, so this might not be enough.
Time of idea generation: 2026-03-18T10:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `class_conditional_model`
Description: Make the model class-conditional by embedding the class label and conditioning the denoising on it. ALREADY IMPLEMENTED: DiT model has LabelEmbedder (line 536) + AdaLN conditioning (line 545). Class labels are passed as class_labels=y_b in forward calls (line 2303). The model already receives class conditioning via additive fusion with time embeddings (line 741). Despite this, it still converges to mean prediction on random batches — meaning the problem isn't lack of class information, but rather that SPSA can't learn to USE the class information effectively.
Confidence: 0
Why: Already implemented and already failing. The model HAS class conditioning but still outputs the same mean prediction for all classes. The 563K-dim SPSA gradient noise prevents learning class-specific features.
Time of idea generation: 2026-03-18T10:00:00
Status: Failed
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `class_ce_inception`
Description: Cross-entropy classification loss using frozen InceptionV3. Generate images via ODE, classify with pretrained InceptionV3, compute cross-entropy vs true class labels. Class labels are STABLE across batches (unlike pixel targets), making this loss resistant to inter-batch noise. Mean prediction = classifier can't distinguish classes = HIGH cross-entropy. Diverse class-specific outputs = LOW cross-entropy.
Confidence: 8
Why: **BREAKTHROUGH RESULT**: gl_class_ce w=1 achieved 252.70 FID — 65 FID better than mean prediction! Only 108 steps due to InceptionV3 overhead but CLEAR learning. Pure class_ce failed (357 FID) — needs GL denoising component. The GL+CE combo works because GL provides pixel-level learning while CE provides class-level gradient signal that's stable across batches.
Time of idea generation: 2026-03-18T08:00:00
Status: Success
HPPs: `--spsa-loss-type class_ce --n-perts 100 --total-batch-size 1000 --device-batch-size 1000 --time-budget 10800`; also `gl_class_ce` variants with --diversity-weight 1.0 and 5.0
Time of run start and end: 2026-03-18T~09:30 - running
Results vs. Baseline: pending (R157 experiments)
wandb link: pending
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `data_curriculum`
Description: Data curriculum — start training on a fixed pool of N images (known to work on fixed batch), then linearly/quadratically grow the pool to full ImageNet by end of training. Pool size grows from N (e.g. 48) to 50K over training. Each step samples batch_size images from the current pool. This gradually transitions from the known-working regime (small fixed set → 193 FID) to the target regime (full dataset).
Confidence: 0
Why: FAILED COMPREHENSIVELY. datacur48 B=48 showed excellent loss trajectory (1.29→0.65) but FID was 326 — the model memorized the growing pool without learning generalizable features. All variants failed: datacur48 (326), datacur48 s38 (322), datacur48+vel-match (322), datacur48+kalman (pending), dc48+llrs+kalman (325), dc48_cubic (325), dc48_max2k (323), dc48_max5k (327). The growing pool creates an illusion of learning (loss decreases on pool images) but the model doesn't generalize to validation set.
Time of idea generation: 2026-03-18T07:00:00
Status: Failed
HPPs: Multiple configs: pool=48/200/1000/5000, B=48/200/1000/5000, with and without gl_diversity, vel-match
Time of run start and end: 2026-03-18T~09:30 - running (R156 experiments)
Results vs. Baseline: pending
wandb link: pending
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `gl_diversity_loss`
Description: Self-supervised diversity loss: GL denoising + pairwise L2 distance penalty on generated images. Mean prediction = all outputs identical = 0 pairwise distance = maximum penalty. Uses torch.cdist on flattened image outputs. diversity_weight controls balance between GL denoising quality and output diversity.
Confidence: 3
Why: This is target-independent — it only looks at model outputs, not targets. Mean prediction is the worst case for this loss (zero diversity). However, the diversity penalty might fight against the denoising loss (which pushes toward correct images). Also, pairwise L2 distance on 32x32 pixel space may not be a great diversity measure. Score is moderate.
Time of idea generation: 2026-03-18T06:00:00
Status: Running
HPPs: diversity_weight=0.1/0.5/5/10/50, B=1k/5k, np=100
Time of run start and end: 2026-03-18T~08:00 - running (R153 w=0.1 DONE: 326 FID; R153 w=0.5 running ~8%; R155 w=5/10/50 running ~0%)
Results vs. Baseline: R153 gl_div_w0.1: 326.20 FID (FAILED — diversity weight too low, still mean prediction)
wandb link: pending
Analysis: w=0.1 was too weak — the GL denoising component dominates and converges to mean prediction. Need much stronger diversity weight. Testing w=5, 10, 50 in R155.
Conclusion: pending (need higher weight results)
Next Ideas to Try: If high weights work, try adaptive weighting that starts high and decreases.
---

---
idea_id: `contrastive_warmup_then_GL`
Description: Two-phase loss schedule: use contrastive loss for first X% of training (to break mean prediction by forcing output diversity), then switch to GL denoising for remaining (1-X)% (to refine image quality). Contrastive loss creates gradient signal that pushes outputs apart, then GL takes over for pixel-level quality.
Confidence: 2
Why: Contrastive loss was shown to converge to ln(B)=6.908 = mean prediction optimum in R152b (394 FID for pure contrastive). However, early contrastive training MIGHT break the mean prediction basin before convergence. The short warmup prevents full convergence to the bad optimum. Score is low because the theoretical analysis shows contrastive loss actively ENCOURAGES mean prediction (equidistant outputs maximize InfoNCE entropy). Still testing as it's cheap to try.
Time of idea generation: 2026-03-18T05:00:00
Status: Running
HPPs: warmup_frac=0.10/0.30/0.50, B=1k, np=100
Time of run start and end: 2026-03-18T~09:30 - running (R154: 10% at 6%, 30% at 10%, 50% at 4%)
Results vs. Baseline: pending
wandb link: pending
Analysis: Theoretical concern: contrastive loss optimum IS mean prediction. Warmup may not help.
Conclusion: pending
Next Ideas to Try: If fails, try class_ce warmup instead (class labels don't have mean prediction optimum)
---

---
idea_id: `gl_spectral_loss`
Description: GL denoising + high-frequency spectral energy penalty. Mean prediction = blurry mean image = zero high-frequency content = maximum penalty. Uses FFT to measure ratio of high-freq to total energy. Forces model to generate images with sharp edges and texture (high-freq content).
Confidence: 2
Why: Spectral analysis can detect mean prediction (which is inherently low-frequency). However, random noise also has high-frequency content, so this might encourage noisy outputs rather than meaningful images. Also, the FFT computation adds overhead. Score is low.
Time of idea generation: 2026-03-18T06:00:00
Status: Implemented, not tried
HPPs: diversity_weight=5/20, B=1k, np=100 (queued in R155)
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `kalman_on_loss`
Description: Apply Kalman filtering to the loss values (not gradients) to get better estimates of the true loss for each perturbation direction. Currently loss from each perturbation is extremely noisy (different random batch each time). Kalman filter maintains a running estimate of the "true" loss surface and uses that for gradient estimation.
Confidence: 3
Why: Loss values from SPSA are noisy due to both batch sampling and perturbation randomness. Kalman filtering could reduce noise. On fixed batch, kalman-grad improved from 194.57→193.87 FID (R146). On full dataset the noise is much larger, so Kalman might help more — but it might also smooth away real signal.
Time of idea generation: 2026-03-17T12:00:00
Status: Implemented, not tried
HPPs: --kalman-grad on full dataset
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `reduced_dim_spsa`
Description: Instead of perturbing all 563K parameters, perturb in a low-dimensional subspace. Options: (a) random linear projection to d'=1000 dims, (b) PCA of gradient history, (c) perturb only a random subset of parameters each step. SPSA gradient variance is O(d) so reducing d from 563K to 1K would reduce variance by 563x.
Confidence: 3
Why: The curse of dimensionality is the root cause. SPSA gradient = (L+ - L-) / (2*eps) * z where z is 563K-dim. The signal-to-noise ratio is O(1/sqrt(d)). Going to d'=1000 gives 24x better SNR. However, (a) sparse perturbation was tested in R139b at 30% sparsity → 315.66 FID (terrible), (b) layerwise SPSA was tested → 314 FID (terrible). Need smarter dimensionality reduction, not naive sparsification.
Time of idea generation: 2026-03-18T10:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis: Prior attempts at dimensionality reduction (sparse-pert 0.3, layerwise) failed catastrophically. Need to perturb ALL parameters but in a correlated low-dim subspace.
Conclusion:
Next Ideas to Try:
---

---
idea_id: `momentum_spsa`
Description: Use exponential moving average of SPSA gradient estimates as the actual update direction. Momentum β=0.9 means current gradient is 10% new SPSA estimate + 90% historical. This smooths out batch noise across time. Different from Adam (which was tested and found bad) because we use SGD + momentum, no adaptive LR.
Confidence: 3
Why: Standard SPSA has no memory — each step uses a completely independent gradient estimate. Momentum accumulates signal over time. If the true gradient is roughly stationary (which it should be early in training), momentum provides sqrt(1/(1-β)) ≈ 3x variance reduction. However, the true gradient changes as the model learns, so high momentum might track a stale gradient.
Time of idea generation: 2026-03-18T10:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `multi_point_spsa`
Description: Instead of 2-point SPSA (L+, L-), use multi-point evaluation: evaluate loss at θ+εz₁, θ+εz₂, ..., θ+εzₖ, θ-εz₁, ..., θ-εzₖ and combine all 2k evaluations for a better gradient estimate. This is essentially what n_perts already does (100 perturbation directions), but we could increase to 400+ with smaller batches.
Confidence: 2
Why: Already doing n_perts=100. Increasing to 400 was tested in R150/R151 — no improvement on full dataset (all ~317 FID). More perturbations helps on fixed batch but doesn't solve the batch noise problem.
Time of idea generation: 2026-03-17T12:00:00
Status: Failed
HPPs: --n-perts 400 --total-batch-size 10000
Time of run start and end: 2026-03-17T~12:00 - 2026-03-17T~15:00 (R150/R151)
Results vs. Baseline: 316-322 FID (all mean prediction, no improvement over np=100)
wandb link:
Analysis: More perturbations does NOT help on full dataset. The bottleneck is inter-batch noise, not within-batch SPSA noise. On the same batch, 100 perturbations is already sufficient. On random batches, the loss landscape shifts too much between steps.
Conclusion: FAILED — increasing n_perts does not solve the full-dataset problem. Inter-batch noise is the bottleneck.
Next Ideas to Try: Focus on reducing inter-batch noise (data curriculum, class conditioning) rather than improving within-batch SPSA quality.
---

## ============================================================
## COMPLETED / TRIED IDEAS (archived, sorted by round)
## ============================================================

---
idea_id: `backprop_baseline`
Description: Establish backprop baseline with depth=1 AdamW training for 1 hour. Standard backpropagation with full gradients as the reference point.
Confidence: 10
Why: Need a reference. Backprop is the gold standard.
Time of idea generation: 2026-03-10T00:00:00
Status: Success
HPPs: `--depth 1 --lr 1e-4` (default config)
Time of run start and end: 2026-03-10 - 2026-03-10
Results vs. Baseline: 152.41 FID (IS the baseline)
wandb link:
Analysis: Depth=1 is a weak model. Deeper models with higher LR do much better.
Conclusion: Good reference point. Deeper models needed.
Next Ideas to Try: depth=2,4,6,8 + higher LR
---

---
idea_id: `backprop_depth8_lr7e4_b2_95`
Description: Backprop with depth=8, lr=7e-4, beta2=0.95. Best backprop config found through systematic grid search of depth (1-12), LR (1e-4 to 1e-3), and Adam beta2 (0.85-0.999).
Confidence: 8
Why: Depth=8 balances model capacity (more parameters) vs step count (fewer steps in 1hr due to slower fwd/bwd). LR=7e-4 is aggressive but stable. beta2=0.95 reduces gradient noise from Adam's second moment.
Time of idea generation: 2026-03-11T00:00:00
Status: Success
HPPs: `--depth 8 --lr 7e-4 --beta2 0.95`
Time of run start and end: 2026-03-11 - 2026-03-12 (rounds 5-7)
Results vs. Baseline: **49.03 FID** (best backprop, 3x better than baseline 152)
wandb link:
Analysis: Deeper is better up to depth=8 (diminishing returns from fewer steps). beta2=0.95 >> 0.999 for short training. Higher LR critical for 1hr runs.
Conclusion: Best backprop config. SPSA target is to beat 49 FID.
Next Ideas to Try: Transfer insights (depth, LR scaling) to SPSA.
---

---
idea_id: `spsa_baseline_random_batches`
Description: SPSA with default settings on full ImageNet random batches.
Confidence: 2
Why: Zero-order optimization is fundamentally harder than backprop. Need to establish SPSA baseline.
Time of idea generation: 2026-03-10T00:00:00
Status: Failed
HPPs: `--solver spsa` (defaults)
Time of run start and end: 2026-03-10
Results vs. Baseline: 317.10 FID (mean prediction — model outputs constant image)
wandb link:
Analysis: Complete failure. Model converges to dataset mean. SPSA gradient variance too high with d=563K parameters and random batches.
Conclusion: Random batches + default SPSA = mean prediction. Need variance reduction.
Next Ideas to Try: Fixed batch, curriculum T schedule, higher n_perts.
---

---
idea_id: `spsa_fixed_batch_48`
Description: Train SPSA on a fixed batch of 48 images instead of random batches from full ImageNet. Since both L+ and L- evaluations use the exact same data, batch noise is eliminated entirely. The model only needs to memorize 48 images.
Confidence: 7
Why: Eliminates inter-batch noise entirely. SPSA gradient is exact (up to perturbation noise). On 48 images, the model can learn detailed per-image features.
Time of idea generation: 2026-03-13T00:00:00
Status: Success
HPPs: `--solver spsa --fixed-batch-size 48 --fixed-batch-mode all`
Time of run start and end: 2026-03-13 - 2026-03-18
Results vs. Baseline: **193.87 FID** (best SPSA ever, R146) vs 317 FID (random batch baseline)
wandb link:
Analysis: Fixed batch works! Model learns meaningful features on 48 images that partially generalize to validation set. But generalization is limited — model memorizes specific images. Combined with loss-lr-scale + kalman-grad + GL denoising for 193.87 FID.
Conclusion: Fixed batch is the only working regime for SPSA. But limited by small training set. Need to bridge to full dataset.
Next Ideas to Try: Data curriculum (gradual expansion), class-conditional model, batch reuse.
---

---
idea_id: `curriculum_T_schedule`
Description: Curriculum schedule for denoising steps T: start training at T=1 (easy, one-step denoising) and gradually increase T to max (harder, multi-step). curriculum-frac controls what fraction of training is spent ramping up.
Confidence: 8
Why: T=1 is easy — one step of denoising. T=4+ requires accurate multi-step trajectories. Starting easy and getting harder lets the model build up capacity gradually.
Time of idea generation: 2026-03-14T00:00:00
Status: Success
HPPs: `--t-schedule curriculum --curriculum-frac 0.60 --t-min 1 --t-max 4`
Time of run start and end: 2026-03-14 - 2026-03-16 (R134-R141)
Results vs. Baseline: Improved from ~210→202.80 FID on fixed batch (1hr). frac=0.6-0.67 all competitive.
wandb link:
Analysis: Curriculum T is the single most impactful feature for SPSA. frac=0.6 (60% spent ramping, 40% at max T) is the sweet spot. Linear T and exponential T also work but curriculum is slightly better. T range 1→4 is optimal; 1→5 marginal, 1→6 catastrophic (285 FID).
Conclusion: **KEY FEATURE** — always use curriculum T frac=0.6 for SPSA.
Next Ideas to Try: Curriculum polish (fine-tune at max T in final X% of training).
---

---
idea_id: `gauss_legendre_loss`
Description: Replace endpoint MSE loss with Gauss-Legendre quadrature over the ODE trajectory. Instead of only measuring loss at the final point, evaluate denoising quality at GL quadrature nodes along the trajectory. This gives a richer signal about trajectory quality.
Confidence: 7
Why: Endpoint-only loss means all intermediate steps get no direct supervision. GL quadrature gives loss signal at each ODE step, weighted by GL weights. This should help SPSA because it provides more gradient signal per forward pass.
Time of idea generation: 2026-03-15T00:00:00
Status: Success
HPPs: `--spsa-loss-type denoising_gauss_legendre`
Time of run start and end: 2026-03-15 - 2026-03-16 (R137+)
Results vs. Baseline: Improved by ~2-4 FID on fixed batch. Part of the champion config.
wandb link:
Analysis: GL loss provides richer per-step signal. Combined with curriculum T, it helps the model learn accurate intermediate denoising steps. Standard component of all modern SPSA configs.
Conclusion: **KEY FEATURE** — always use GL loss for SPSA.
Next Ideas to Try: Combine with other loss components (diversity, classification).
---

---
idea_id: `loss_lr_scale`
Description: Scale learning rate by the current loss value: `lr = lr_base * warmdown(t) * clamp(loss/loss_init, 0.1, 3.0)`. When loss is high (model struggling), take larger steps. When loss is low (model doing well), take smaller steps. This provides natural learning rate adaptation.
Confidence: 7
Why: SPSA gradient noise is roughly constant regardless of loss magnitude. When loss is high, the true gradient is large relative to noise, so we should take bigger steps. When loss is low, signal-to-noise deteriorates, so smaller steps prevent overshoot.
Time of idea generation: 2026-03-17T00:00:00
Status: Success
HPPs: `--loss-lr-scale`
Time of run start and end: 2026-03-17 (R143)
Results vs. Baseline: **194.57 FID** (R143) — new record at the time! ~8 FID improvement over no loss-lr-scale.
wandb link:
Analysis: Dramatic improvement. Loss-lr-scale is the best single feature after curriculum T. BUT it hurts at 3h (198-202 FID vs 1hr's 194), suggesting it over-decays LR in long runs.
Conclusion: **KEY FEATURE for 1hr runs**. Do NOT use for 3hr runs.
Next Ideas to Try: Combine with kalman-grad, group-adaptive-lr, block-cyclic.
---

---
idea_id: `kalman_gradient_filter`
Description: Apply Kalman filtering to SPSA gradient estimates. Maintains a state estimate of the "true" gradient direction and updates it with each noisy SPSA observation. Provides smoother, more accurate gradient estimates.
Confidence: 6
Why: SPSA gradients are extremely noisy (d=563K). Kalman filtering is theoretically optimal for combining noisy observations of a slowly-changing state. On fixed batch, improved from 194.57→193.87 FID.
Time of idea generation: 2026-03-17T06:00:00
Status: Success
HPPs: `--kalman-grad`
Time of run start and end: 2026-03-17 (R146)
Results vs. Baseline: **193.87 FID** with loss-lr-scale + kalman-grad (R146), **193.19 FID** with + grpadapt (R147)
wandb link:
Analysis: Consistent ~0.7 FID improvement on fixed batch. Kalman filtering smooths gradient noise effectively. Combined with other features for best SPSA results.
Conclusion: Helpful but small improvement on fixed batch. Untested on full dataset.
Next Ideas to Try: Test on full dataset where gradient noise is much larger.
---

---
idea_id: `group_adaptive_lr`
Description: Per-parameter-group learning rate scaling. Scale each parameter group's LR by the inverse of its gradient norm, so groups with large gradients take smaller steps and groups with small gradients take larger steps.
Confidence: 4
Why: Different parameter groups may have very different gradient scales. Adaptive per-group LR could help underfitting groups learn faster. Improved from 195.55→193.19 FID when combined with kalman-grad (R147).
Time of idea generation: 2026-03-17T06:00:00
Status: Success
HPPs: `--group-adaptive-lr`
Time of run start and end: 2026-03-17 (R144-R147)
Results vs. Baseline: 195.55 FID alone (R144), 193.19 FID with kalman-grad (R147)
wandb link:
Analysis: Marginal improvement alone but good in combination. 3-seed average ~196.3 (very consistent).
Conclusion: Minor but consistent feature. Good in combination with kalman-grad.
Next Ideas to Try:
---

---
idea_id: `block_cyclic_perturbation`
Description: Instead of perturbing all parameters simultaneously, cycle through blocks of parameters. Step 1: perturb block A. Step 2: perturb block B. Etc. This reduces effective dimensionality per step.
Confidence: 3
Why: SPSA variance is O(d). Perturbing d/K parameters per step reduces variance by K^2 but needs K steps per full gradient. Net effect depends on whether reduced noise outweighs slower full-gradient updates.
Time of idea generation: 2026-03-16T00:00:00
Status: Unclear
HPPs: `--block-cyclic`
Time of run start and end: 2026-03-16 - 2026-03-17 (R139, R145, R147)
Results vs. Baseline: 203.00 FID alone (R139), 195.25 with loss-lr-scale (R145), 198.15 with kalman+grpadapt (R147)
wandb link:
Analysis: Results are mixed. Sometimes helps, sometimes hurts. When combined with loss-lr-scale it gave 195.25 (R145) but with kalman+grpadapt it gave 198.15 (worse than 193.19 without blockcyc). The overhead of cycling through blocks means fewer effective full-gradient steps.
Conclusion: Mixed results. Not clearly beneficial. Skip for now.
Next Ideas to Try:
---

---
idea_id: `full_dataset_baseline_modern`
Description: Test modern SPSA config (GL loss, curriculum T, zero-init, warmdown, winsorization, loss-lr-scale, kalman-grad) on full ImageNet random batches. Previous full-dataset failures used old config.
Confidence: 2
Why: Early full-dataset attempts (R142) used weaker config. The modern config might provide enough signal. However, the fundamental problem (batch noise >> SPSA signal at d=563K) likely persists.
Time of idea generation: 2026-03-17T12:00:00
Status: Failed
HPPs: Various: B=1k-10k, np=100-400, lr=eps=1e-2, ±curvature, ±batch-reuse
Time of run start and end: 2026-03-17T12:00 - 2026-03-17T18:00 (R142, R150, R151)
Results vs. Baseline: ALL 316-322 FID (mean prediction). No config worked.
wandb link:
Analysis: EVERY configuration on full ImageNet with random batches produces ~317 FID. B=1k, 5k, 10k all fail. np=100, 400 both fail. Curvature doesn't help. Kalman-grad doesn't help. Loss-lr-scale doesn't help. The problem is fundamental: SPSA gradient variance O(d=563K) makes it impossible to learn within-class variation when batches change every step.
Conclusion: **FUNDAMENTAL FAILURE** — modern config does not solve full-dataset SPSA. Need qualitatively different approach.
Next Ideas to Try: Data curriculum, class-conditional model, target-independent losses, dimensionality reduction.
---

---
idea_id: `batch_reuse`
Description: Reuse the same data batch for N consecutive SPSA steps before loading new data. With reuse=10, each batch gets 10 gradient steps (like fixed batch but rotating). Reduces effective batch noise by factor of N.
Confidence: 3
Why: If batch noise is the problem, reusing batches should help. With reuse=10 and ~560 steps/hr, model sees ~56 unique batches. More diverse than 48 fixed images.
Time of idea generation: 2026-03-17T14:00:00
Status: Failed
HPPs: `--batch-reuse-steps 5/10/20/50/100` with B=1k, np=100-200
Time of run start and end: 2026-03-17T14:00 - 2026-03-17T18:00 (R142)
Results vs. Baseline: ALL 316-318 FID (mean prediction). Reuse=5 through reuse=100 all failed.
wandb link:
Analysis: Even reusing the same batch for 100 consecutive steps doesn't help! The model never escapes the mean prediction basin. This suggests the problem isn't just batch noise — it's that the mean prediction IS the optimal solution when the model will eventually see diverse data.
Conclusion: **FAILED** — batch reuse does not solve mean prediction. The model "knows" it can't specialize to any batch because future batches will be different.
Next Ideas to Try: Target-independent losses (diversity, classification).
---

---
idea_id: `contrastive_loss`
Description: InfoNCE contrastive loss: L = -log(exp(sim(pred_i, target_i)/τ) / Σ_j exp(sim(pred_i, target_j)/τ)). Forces each prediction to be more similar to its own target than to other targets. Target-DEPENDENT but encourages output diversity.
Confidence: 0
Why: Contrastive loss should push outputs apart (each must be closest to its own target). However...
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type contrastive --n-perts 100/400 --total-batch-size 1000`
Time of run start and end: 2026-03-18T~06:00 - 2026-03-18T~09:00 (R152b)
Results vs. Baseline: 394.17 FID (np=100), 458.90 FID (np=400) — WORSE than mean prediction!
wandb link:
Analysis: **CRITICAL DISCOVERY**: Contrastive loss converges to ln(B) = ln(1000) = 6.908 = maximum InfoNCE entropy = MEAN PREDICTION. The mean image is equidistant from all targets, so it achieves maximum entropy in the contrastive distribution. Contrastive loss ACTIVELY ENCOURAGES mean prediction. np=400 is worse because more perturbations help the model converge to the bad optimum faster.
Conclusion: **FUNDAMENTAL FAILURE** — contrastive loss has mean prediction as its optimum on random batches. ANY target-dependent loss suffers from this.
Next Ideas to Try: Target-INDEPENDENT losses only (diversity, spectral, classification by class labels).
---

---
idea_id: `ssim_loss`
Description: Structural Similarity Index (SSIM) loss instead of MSE. SSIM measures perceptual similarity via luminance, contrast, and structure. More perceptually meaningful than pixel MSE.
Confidence: 2
Why: SSIM might provide richer gradient signal about structural quality. However, it's still target-dependent.
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type denoising_ssim --n-perts 100 --total-batch-size 1000`
Time of run start and end: 2026-03-18T~06:00 - 2026-03-18T~09:00 (R152b)
Results vs. Baseline: 307.78 FID (best of the alternative losses, still mean prediction territory)
wandb link:
Analysis: SSIM is slightly better than pure MSE (307 vs 317) because it captures some structural information. But it's still target-dependent and the class-conditional mean predictor is still near-optimal.
Conclusion: Minor improvement but doesn't solve the fundamental problem. Target-dependent losses all converge to mean prediction.
Next Ideas to Try:
---

---
idea_id: `trajectory_diversity_loss`
Description: Penalize similarity between ODE trajectories (not just endpoints). If two images have similar intermediate states, add a penalty. This encourages diverse generation paths.
Confidence: 2
Why: Trajectory-level diversity might be harder to game than endpoint diversity.
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type denoising_traj_div --n-perts 100 --total-batch-size 1000`
Time of run start and end: 2026-03-18T~06:00 - 2026-03-18T~09:00 (R152b)
Results vs. Baseline: 322.58 FID (mean prediction)
wandb link:
Analysis: Trajectory diversity didn't help. The model still converges to mean prediction with identical trajectories for all inputs.
Conclusion: Failed. Trajectory-level penalties insufficient.
Next Ideas to Try:
---

---
idea_id: `progressive_loss`
Description: Progressive loss: start with MSE loss at endpoint only, gradually add loss at intermediate ODE steps. Simpler curriculum than GL quadrature.
Confidence: 3
Why: Gradual introduction of intermediate loss might help SPSA handle multi-step evaluation.
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type denoising_progressive --n-perts 100 --total-batch-size 1000`
Time of run start and end: 2026-03-18T~06:00 - 2026-03-18T~09:00 (R152b)
Results vs. Baseline: 323.62 FID (mean prediction)
wandb link:
Analysis: On random batches, progressive loss behaves same as standard MSE — converges to mean.
Conclusion: Failed on random batches. Progressive loss helped slightly on fixed batch (205 FID in earlier rounds).
Next Ideas to Try:
---

---
idea_id: `ssim_mse_combo`
Description: Combined SSIM + MSE loss.
Confidence: 2
Why: Combining losses might capture both pixel and structural quality.
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type denoising_ssim_mse --n-perts 100 --total-batch-size 1000`
Time of run start and end: 2026-03-18T~06:00 - 2026-03-18T~09:00 (R152b)
Results vs. Baseline: 321.27 FID (mean prediction)
wandb link:
Analysis: SSIM+MSE combo is still target-dependent. Mean prediction remains near-optimal.
Conclusion: Failed.
Next Ideas to Try:
---

---
idea_id: `direct_fid_loss`
Description: Use FID score directly as the loss function for SPSA. FID is the evaluation metric, so optimizing it directly should be ideal. Zero-order doesn't need differentiability!
Confidence: 1
Why: FID is the true objective. However, computing FID requires generating many images and comparing statistics to a reference set — extremely expensive per step.
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type direct_fid --n-perts 100 --total-batch-size 128`
Time of run start and end: 2026-03-18T~06:00 (R152b)
Results vs. Baseline: ~348s/step = only 31 steps in 3h. Loss going UP (339→357). Fundamentally impractical.
wandb link:
Analysis: Each step takes ~6 minutes due to FID computation. Only ~31 steps possible in 3 hours. Not enough steps to learn anything. Also, FID is very noisy with small samples.
Conclusion: **IMPRACTICAL** — too slow. Need thousands of steps for SPSA, but FID takes minutes per evaluation.
Next Ideas to Try: Mini-FID with fewer samples? Still too slow.
---

---
idea_id: `cosine_loss`
Description: Cosine similarity loss between generated and target images.
Confidence: 1
Why: Cosine similarity is scale-invariant, might provide different gradient landscape.
Time of idea generation: 2026-03-17T20:00:00
Status: Failed
HPPs: `--spsa-loss-type denoising_cosine --n-perts 100 --total-batch-size 1000`
Time of run start and end: 2026-03-18T~06:00 (R152b)
Results vs. Baseline: CRASH — 20 rollbacks → abort. Loss oscillating around 0 with no gradient signal.
wandb link:
Analysis: Cosine similarity provides no usable gradient for SPSA. The loss landscape is too flat around the cosine optimum.
Conclusion: Failed completely. Cosine loss incompatible with SPSA.
Next Ideas to Try:
---

---
idea_id: `spsa_depth_increase`
Description: Increase model depth for SPSA. Deeper models perform dramatically better with backprop (depth 8 → 49 FID). Might also help SPSA.
Confidence: 2
Why: Deeper models have more capacity. But more parameters = higher d = worse SPSA gradient variance.
Time of idea generation: 2026-03-14T00:00:00
Status: Failed
HPPs: `--solver spsa --depth 3/4/8`
Time of run start and end: 2026-03-14 - 2026-03-16
Results vs. Baseline: depth=3: 221 FID. depth=4: 317 FID on full dataset. depth=8: 316 FID on full dataset. Only depth=1 works with SPSA.
wandb link:
Analysis: SPSA gradient variance is O(d). depth=1 has 563K params. depth=4 has ~1.5M params. The curse of dimensionality makes deeper models worse for SPSA, not better. This is the opposite of backprop where depth helps dramatically.
Conclusion: **Depth=1 is optimal for SPSA**. Cannot leverage depth like backprop.
Next Ideas to Try: Focus on loss/optimizer improvements rather than architecture.
---

---
idea_id: `warmdown_schedule`
Description: Linear warmdown at end of training: LR decays linearly to 0 over the last X% of training. Prevents late-training gradient noise from undoing early learning.
Confidence: 7
Why: SPSA gradients are noisy. Late in training when loss is low, noise dominates. Warming down LR reduces the damage from noisy late-training updates.
Time of idea generation: 2026-03-14T00:00:00
Status: Success
HPPs: `--warmdown-ratio 0.10` (10% warmdown) through 0.30 (30%)
Time of run start and end: 2026-03-14 - 2026-03-16
Results vs. Baseline: warmdown=0.30 (default) is best overall. warmdown<0.25 is high variance.
wandb link:
Analysis: warmdown=0.30 gives consistent results. Lower warmdown values are high variance (sometimes great ~200, usually bad ~244). warmdown=0.10 is fine for 1hr with loss-lr-scale (since llrs provides its own dampening).
Conclusion: **KEY FEATURE** — 30% warmdown for standard runs, 10% with loss-lr-scale.
Next Ideas to Try:
---

---
idea_id: `winsorize_pct`
Description: Winsorize loss values: clip top and bottom X% of loss values across perturbations before averaging. Removes outlier losses that could corrupt gradient estimate.
Confidence: 6
Why: SPSA with 100 perturbations produces 200 loss values. Some are outliers. Winsorization at 5% removes the 10 worst and 10 best, using the middle 180 for gradient estimation.
Time of idea generation: 2026-03-14T00:00:00
Status: Success
HPPs: `--winsorize-pct 0.05`
Time of run start and end: 2026-03-14
Results vs. Baseline: Part of the standard config. Removes ~2-3 FID of noise.
wandb link:
Analysis: 5% winsorization is the sweet spot. Higher values remove too much signal.
Conclusion: Standard feature, always use.
Next Ideas to Try:
---

---
idea_id: `checkpoint_rollback`
Description: After each step, if loss increased by more than a threshold, roll back to the previous checkpoint. Prevents catastrophic steps from ruining the model.
Confidence: 7
Why: SPSA occasionally takes catastrophic steps due to bad gradient estimates. Rollback limits downside.
Time of idea generation: 2026-03-14T00:00:00
Status: Success
HPPs: `--checkpoint-rollback --fail-threshold 99999` (with very high threshold = rollback only on divergence)
Time of run start and end: 2026-03-14
Results vs. Baseline: Essential safety feature. Prevents crashes from divergent steps.
wandb link:
Analysis: With fail-threshold=99999, only rolls back on extreme divergence. More aggressive thresholds cause too many rollbacks.
Conclusion: Standard feature, always use.
Next Ideas to Try:
---

---
idea_id: `curriculum_polish`
Description: After curriculum T reaches maximum, spend the final X% of training at max T only (no curriculum ramp). This "polishes" the model's multi-step denoising.
Confidence: 5
Why: Curriculum wastes some training time at low T. Polish phase focuses entirely on the hard case (max T).
Time of idea generation: 2026-03-16T00:00:00
Status: Success
HPPs: `--curriculum-polish 0.06` (6% polish phase)
Time of run start and end: 2026-03-16 (R139)
Results vs. Baseline: 202.88 FID with polish (R139) — marginal improvement.
wandb link:
Analysis: Small improvement. 6% polish is enough. Higher values waste too much time at hard T.
Conclusion: Minor but consistent improvement. Part of champion config.
Next Ideas to Try:
---

---
idea_id: `loss_lr_ema`
Description: Use EMA of loss values for loss-lr-scale instead of instantaneous loss. Smoother LR adaptation.
Confidence: 3
Why: Instantaneous loss is noisy. EMA should smooth it.
Time of idea generation: 2026-03-17T08:00:00
Status: Failed
HPPs: `--loss-lr-ema 0.99` and `--loss-lr-ema 0.995`
Time of run start and end: 2026-03-17 (R146)
Results vs. Baseline: ema=0.99: CRASH (EMA lag → LR stays high → divergence). ema=0.995 survives but no better than no-EMA.
wandb link:
Analysis: EMA=0.99 is too smooth — when loss drops quickly (e.g., during T ramp), EMA lags behind, keeping LR artificially high, causing divergence. EMA=0.995 is even smoother but happens to survive. Not worth the complexity.
Conclusion: Failed. Don't use loss-lr-ema.
Next Ideas to Try:
---

---
idea_id: `batch_growth`
Description: Start with small batch size and grow over training. Early training uses B=16 for fast steps, late training uses B=48+ for stable gradients.
Confidence: 2
Why: More steps early might bootstrap faster. Larger batches later for stability.
Time of idea generation: 2026-03-17T08:00:00
Status: Failed
HPPs: `--batch-growth 16`
Time of run start and end: 2026-03-17 (R146)
Results vs. Baseline: 208.69 FID (vs 193.87 baseline) — **HURTS by 15 FID**
wandb link:
Analysis: Small early batches = weak gradient signal = wasted steps. The model needs good gradient signal from the start.
Conclusion: Failed badly. Don't reduce batch size early.
Next Ideas to Try:
---

---
idea_id: `pert_recycle`
Description: Reuse perturbation vectors for N consecutive steps. Saves compute on random number generation.
Confidence: 1
Why: Might allow more steps per hour. But reusing perturbations means exploring the same directions repeatedly.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--pert-recycle 3/5`
Time of run start and end: 2026-03-16 (R140)
Results vs. Baseline: 297.70 FID (recycle=3), 308 FID (recycle=5) — TERRIBLE
wandb link:
Analysis: Reusing perturbations destroys gradient diversity. Each step explores the same random directions, creating a rank-deficient gradient that only moves in a small subspace.
Conclusion: Never reuse perturbations.
Next Ideas to Try:
---

---
idea_id: `layerwise_spsa`
Description: Perturb one layer at a time instead of all layers. Reduces effective dimensionality per step.
Confidence: 2
Why: O(d_layer) variance instead of O(d_total). But needs K steps per full gradient (K = num layers).
Time of idea generation: 2026-03-14T00:00:00
Status: Failed
HPPs: various configs over R75, R121, R140
Time of run start and end: 2026-03-14 - 2026-03-16
Results vs. Baseline: 314-324 FID — TERRIBLE across all attempts
wandb link:
Analysis: One layer per step is too slow. Model needs to coordinate across all layers simultaneously. Layerwise perturbation creates inconsistent per-layer updates that fight each other.
Conclusion: **FUNDAMENTAL FAILURE** — don't perturb subsets of layers.
Next Ideas to Try:
---

---
idea_id: `sparse_perturbation`
Description: Only perturb X% of parameters per step (random mask). Reduces effective d.
Confidence: 1
Why: O(X*d) variance. But missed parameters get no update.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--sparse-pert 0.3` (30% sparsity)
Time of run start and end: 2026-03-16 (R139b)
Results vs. Baseline: 315.66 FID — TERRIBLE
wandb link:
Analysis: 30% sparsity means 70% of parameters are stale. The sparse gradient is biased and incomplete.
Conclusion: Don't use sparse perturbation.
Next Ideas to Try:
---

---
idea_id: `elite_perturbations`
Description: Keep the top-K best perturbation directions and reuse them. Evolutionary strategy approach.
Confidence: 2
Why: Elite selection focuses compute on promising directions.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--elite-perts 10`
Time of run start and end: 2026-03-16 (R140)
Results vs. Baseline: 226.74 FID — HURTS by ~24 FID
wandb link:
Analysis: Elite reuse creates a biased gradient estimate. The "best" directions in one step may be irrelevant in the next. Degrades to a low-rank update that can't explore the full parameter space.
Conclusion: Failed. Don't use elite perturbation reuse.
Next Ideas to Try:
---

---
idea_id: `stochastic_weight_averaging`
Description: Average model weights over training (SWA). Produces smoother model.
Confidence: 2
Why: SWA can reduce overfitting and smooth out noise. Standard technique.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--swa-frac 0.01/0.05/0.1`
Time of run start and end: 2026-03-16 (R140, R140b)
Results vs. Baseline: 207.39 (swa=0.1), CRASH (swa=0.05) — worse or crashed
wandb link:
Analysis: SWA doesn't help with SPSA's short training. With only ~3000 steps, there aren't enough weight snapshots for meaningful averaging.
Conclusion: Failed. Not enough training steps for SWA to help.
Next Ideas to Try:
---

---
idea_id: `eps_adaptive`
Description: Adaptive epsilon scheduling: increase/decrease eps based on gradient variance.
Confidence: 1
Why: Better eps might give better gradient signal.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--eps-schedule adaptive_var`
Time of run start and end: 2026-03-16 (R140)
Results vs. Baseline: 310.24 FID — TERRIBLE
wandb link:
Analysis: Adaptive eps oscillates wildly and destabilizes training.
Conclusion: Failed. eps=1e-2 fixed is optimal.
Next Ideas to Try:
---

---
idea_id: `double_dip`
Description: Two-cycle curriculum: ramp T from 1→4, then reset to 1 and ramp again. Gives model two passes at learning.
Confidence: 2
Why: Second pass might refine what first pass learned.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--double-dip`
Time of run start and end: 2026-03-16 (R140)
Results vs. Baseline: 222.51 FID — HURTS by ~20 FID
wandb link:
Analysis: Resetting T undoes multi-step learning from first cycle. The model has to relearn T=1 denoising, wasting steps.
Conclusion: Failed. Single curriculum is better.
Next Ideas to Try:
---

---
idea_id: `velocity_matching`
Description: Auxiliary loss: match ODE velocity field at each step. Provides per-step gradient signal independent of final endpoint.
Confidence: 3
Why: Per-step loss gives richer signal than endpoint-only. Velocity matching is a standard flow matching technique.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--vel-match 0.3/0.5`
Time of run start and end: 2026-03-16 (R139b, R154)
Results vs. Baseline: vel-match 0.3: CRASH (diverged). vel-match 0.5: pending (R156 with data curriculum).
wandb link:
Analysis: vel-match alone caused divergence. May work as auxiliary loss combined with GL denoising.
Conclusion: Crashed alone. Testing with data curriculum in R156.
Next Ideas to Try:
---

---
idea_id: `lora_spsa`
Description: Low-rank adaptation: only train a LoRA component (rank r=4-16) while keeping base model frozen. Reduces effective d from 563K to ~10K.
Confidence: 2
Why: Dramatically reduces parameter count, should reduce SPSA gradient variance.
Time of idea generation: 2026-03-14T00:00:00
Status: Failed
HPPs: LoRA with various ranks
Time of run start and end: 2026-03-14 (R134)
Results vs. Baseline: 316-317 FID — CATASTROPHIC. LoRA rank constraint kills denoising ability.
wandb link:
Analysis: LoRA restricts the model to a low-rank subspace that can't represent the full denoising function. The model literally cannot generate diverse images because the rank is too low.
Conclusion: **FUNDAMENTAL FAILURE** — LoRA is too restrictive for generative models.
Next Ideas to Try:
---

---
idea_id: `mmd_loss`
Description: Maximum Mean Discrepancy (MMD) loss: distribution matching between generated and target images in kernel space. Measures distance between distributions rather than per-sample MSE.
Confidence: 2
Why: MMD is a distribution-level loss. Might avoid mean prediction by requiring the output distribution to match the target distribution. However, still target-dependent.
Time of idea generation: 2026-03-18T05:00:00
Status: Implemented, not tried
HPPs: `--spsa-loss-type mmd --n-perts 100 --total-batch-size 1000` (queued in R154)
Time of run start and end: pending
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `flow_match_loss`
Description: Per-step flow matching loss: at each ODE step, compare the model's predicted velocity to the ideal linear interpolation velocity. Provides per-step supervision.
Confidence: 2
Why: Standard flow matching loss. Per-step signal might help SPSA. But still target-dependent.
Time of idea generation: 2026-03-18T05:00:00
Status: Implemented, not tried
HPPs: `--spsa-loss-type denoising_flow_match --n-perts 100 --total-batch-size 1000` (queued in R154)
Time of run start and end: pending
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `edge_loss`
Description: MSE + spatial gradient matching (Sobel edge loss). Penalizes blur by requiring matching edge structure.
Confidence: 2
Why: Edge loss might penalize the blurry mean prediction. But still target-dependent.
Time of idea generation: 2026-03-18T05:00:00
Status: Implemented, not tried
HPPs: `--spsa-loss-type denoising_edge --n-perts 100 --total-batch-size 1000` (queued in R154)
Time of run start and end: pending
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `gl_class_diversity`
Description: GL denoising + inter-class pairwise distance: different class labels → outputs should be far apart. Uses class labels (stable across batches) to define diversity.
Confidence: 2
Why: Class labels are stable. Diversity between different classes should prevent mean prediction.
Time of idea generation: 2026-03-18T06:00:00
Status: Implemented, not tried
HPPs: `--spsa-loss-type gl_class_diversity --diversity-weight 5.0` (queued in R155)
Time of run start and end: pending
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `diversity_mse`
Description: Simple endpoint MSE + pairwise distance penalty. No GL quadrature, just endpoint loss + diversity.
Confidence: 2
Why: Simpler version of gl_diversity. MSE baseline + diversity penalty.
Time of idea generation: 2026-03-18T06:00:00
Status: Implemented, not tried
HPPs: `--spsa-loss-type diversity_mse --diversity-weight 10.0` (queued in R155)
Time of run start and end: pending
Results vs. Baseline: pending
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `ce_warmup_then_GL`
Description: Two-phase: class_ce warmup (InceptionV3 classification) for first 30% of training → GL denoising for remaining 70%. Class_ce breaks mean prediction using stable class labels, then GL refines image quality.
Confidence: 3
Why: Unlike contrastive warmup, class_ce doesn't have mean prediction as its optimum. The classifier FAILS on mean prediction (high cross-entropy). So class_ce warmup should push the model away from mean prediction, then GL takes over.
Time of idea generation: 2026-03-18T08:00:00
Status: Running
HPPs: `--warmup-loss-type class_ce --loss-warmup-frac 0.30` (R157)
Time of run start and end: 2026-03-18T~09:30 - running
Results vs. Baseline: pending
wandb link: pending
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `data_curriculum_plus_kalman`
Description: Data curriculum (start with 48 fixed images, grow to full dataset) combined with Kalman gradient filtering. Kalman smooths the increasing gradient noise as the pool grows.
Confidence: 4
Why: Data curriculum alone might struggle during the transition phase as pool grows and noise increases. Kalman filtering can smooth the gradient noise during this transition.
Time of idea generation: 2026-03-18T07:30:00
Status: Running
HPPs: `--data-curriculum 48 --kalman-grad --n-perts 100 --total-batch-size 48 --device-batch-size 48` (R156)
Time of run start and end: 2026-03-18T~09:30 - running
Results vs. Baseline: pending
wandb link: pending
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `3hr_training`
Description: Extend training from 1hr to 3hr. More steps = better optimization. 3hr showed consistent improvement on fixed batch (200.82 FID at 3hr vs 202.80 at 1hr).
Confidence: 6
Why: More steps allow better convergence. Diminishing returns beyond 3hr (4hr overfits). Sweet spot is 3hr.
Time of idea generation: 2026-03-15T00:00:00
Status: Success
HPPs: `--time-budget 10800`
Time of run start and end: 2026-03-15 - 2026-03-16 (R137-R138)
Results vs. Baseline: 3hr best: 199.66 FID (R137 s20). 2hr: 194.92 FID (R141 s2). 3hr sweet spot for stability.
wandb link:
Analysis: 3hr gives ~2-3 FID improvement over 1hr with much less variance. 4hr overfits. np=100 diverges at 3hr (need np=150+).
Conclusion: 3hr is optimal for research runs. 1hr for quick screening.
Next Ideas to Try:
---

---
idea_id: `sign_consensus`
Description: Only apply gradient updates where the sign is consistent across perturbations. If >K perturbations agree on direction, apply update; otherwise skip.
Confidence: 1
Why: Sign consensus might filter out noise.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--sign-consensus 5`
Time of run start and end: 2026-03-16 (R139)
Results vs. Baseline: 306.18 FID — TERRIBLE
wandb link:
Analysis: Sign consensus is too conservative — it filters out most of the gradient signal, leaving almost no update.
Conclusion: Failed completely.
Next Ideas to Try:
---

---
idea_id: `search_strategy_local`
Description: Local search strategy: periodically search for better epsilon by trying nearby values.
Confidence: 1
Why: Might find better operating point.
Time of idea generation: 2026-03-16T00:00:00
Status: Failed
HPPs: `--search-strategy local`
Time of run start and end: 2026-03-16 (R140)
Results vs. Baseline: 316.66 FID — TERRIBLE
wandb link:
Analysis: Local search wastes steps on exploration that doesn't help.
Conclusion: Failed. Fixed eps=1e-2 is optimal.
Next Ideas to Try:
---

---
idea_id: `autoreg_mse_ce_w05`
Description: UNIFIED AUTOREGRESSIVE TRAJECTORY — eliminate teacher forcing entirely. Run ONE autoregressive ODE from noise → image (T=20 steps). At GL quadrature nodes along the autoregressive path, compute velocity MSE against true velocity (x_b - noise). At endpoint, classify with InceptionV3 and compute CE loss. Combined: GL-weighted MSE + w*CE. This replaces gl_class_ce which wastefully runs TWO trajectories (4 teacher-forced GL + 20 autoregressive = 24 fwd passes). Unified does it in ~20 fwd passes total. Key insight: velocity MSE on the model's OWN trajectory is more honest than teacher-forced — it trains on the actual inference distribution. User explicitly requested: "DO NOT do teacher forcing anymore, noise schedules in diffusion suck, lets RNN our way to pdata."
Confidence: 10
Why: (1) gl_class_ce already got 249.16 FID (BEST on full dataset) but wastes half its compute on teacher-forced GL that measures quality on a different trajectory than inference. (2) Velocity MSE on autoregressive path is more aligned with actual inference quality. (3) Saves ~17% forward passes (20 vs 24). (4) Simpler code — one trajectory instead of two. (5) User's vision: treat diffusion as pure RNN with no arbitrary noise schedule.
Time of idea generation: 2026-03-19T06:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `autoreg_ce_pure`
Description: PURE AUTOREGRESSIVE CLASSIFICATION — run full ODE from noise → image, classify endpoint with InceptionV3, CE loss only. No velocity MSE at all. This tests whether the classification signal alone (without any denoising loss) can drive learning on full ImageNet. class_ce previously got 344.76 FID at B=128 and 357 at B=1k, but those were with the dual-trajectory gl_class_ce getting 249. The pure CE on its own may work better now with proper hyperparameters. If it works, this would be the simplest possible loss — zero-order optimizer with pure task loss.
Confidence: 5
Why: Pure class_ce previously failed (344-357 FID) but that was before gl_class_ce showed CE is the KEY signal (w=0.5 beat w=1.0). Without the denoising loss as an anchor, CE alone might be too noisy for SPSA. But worth testing as ablation — if CE alone works, the MSE component is unnecessary complexity.
Time of idea generation: 2026-03-19T06:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `autoreg_mse_ce_sweep`
Description: Sweep diversity_weight for autoreg_mse_ce: w=0.1, 0.3, 0.5, 1.0, 2.0. gl_class_ce showed w=0.5 > w=1.0 > w=0.1, w=2.0 diverges. The unified trajectory may shift the optimal weight since MSE is now computed on autoregressive path (harder, higher MSE values) while CE stays the same.
Confidence: 8
Why: Weight tuning was critical for gl_class_ce (249.16 at w=0.5 vs 252.70 at w=1.0 vs 270.74 at w=0.1). The unified trajectory changes the MSE magnitude so the optimal ratio will shift. Must sweep to find it.
Time of idea generation: 2026-03-19T06:00:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `autoreg_mse_ce_lowlr`
Description: autoreg_mse_ce with lower learning rate (lr=eps=5e-3 or 3e-3). gl_class_ce diverged in 5/8 experiments — the autoregressive ODE is inherently unstable because small parameter changes compound through T=20 steps. Lower LR should stabilize at cost of slower convergence. With 3h budget and ~60s/step at B=1k, we get ~180 steps. Even at lower LR, CE provides strong gradient signal.
Confidence: 7
Why: 5/8 gl_class_ce experiments DIVERGED. The autoregressive ODE magnifies perturbations — each SPSA step can push the model into a regime where ODE outputs explode. Lower lr=eps reduces perturbation size, making each step more conservative. The tradeoff is fewer effective steps, but stability may more than compensate.
Time of idea generation: 2026-03-19T06:01:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: `autoreg_mse_ce_fewer_T`
Description: autoreg_mse_ce with fewer ODE steps (T=10 instead of T=20). Halves forward passes per step (~30s/step instead of ~60s), doubling total steps in time budget. Fewer T = less compounding of errors = more stable. Also means GL nodes map to different ODE steps. Tradeoff: coarser ODE integration = lower image quality ceiling.
Confidence: 6
Why: More steps in time budget is always good for SPSA (more gradient updates). T=20→T=10 doubles throughput. The divergence issue may be partly due to error compounding through 20 steps. T=10 may be more stable while still giving decent images. InceptionV3 can classify even somewhat noisy images.
Time of idea generation: 2026-03-19T06:01:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---
