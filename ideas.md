# Ideas Log

## Completed Experiments (Baseline + Key Advances)

-----------------------------------------------------
idea_id: baseline_tbptt
Description: Baseline TBPTT training with RecursiveDiT (h_cycles=2, l_cycles=3, l_layers=1, n_embd=768). Huber+cosine(fp32)+EMA+symmetric recursion+noise-adaptive depth+x0 reconstruction loss. This is our current best configuration after 85+ experiments.
Confidence: 10
Why: This is the established baseline with all accumulated improvements.
Time of idea generation: 2026-03-17 19:00
Status: Success
HPPs: solver=tbptt, h_cycles=2, l_cycles=3, l_layers=1, n_embd=768, lr=1e-4, batch=256, EMA=0.999, loss=Huber+0.5*cos_fp32+0.1*x0_recon
Time of run start and end: 2026-03-17 19:05 - 2026-03-18 08:00
Results vs. Baseline: 314.38 FID (best), down from 359.63 original baseline (-12.6%)
wandb link: see wandb project trm-recursive-diffusion
Analysis: Key wins were Huber loss (-42 FID with EMA), warm-start z_H (-8), cosine fp32 (-3), symmetric recursion (-0.3), noise-adaptive depth (-0.3), x0 recon (-0.2). Loss function changes dominated. Architecture tweaks were mostly neutral. We're using only 2.5GB of 80GB VRAM and 0.28% MFU — massively underutilizing the GPU.
Conclusion: Solid baseline. Further incremental loss tweaks yield <0.5 FID improvement. Need fundamentally different approaches.
Next Ideas to Try: See below.
-----------------------------------------------------

## New Ideas (Priority Queue)

-----------------------------------------------------
idea_id: multi_step_velocity_loss
Description: Instead of computing loss only at the final output, compute the flow matching velocity loss at EACH H_cycle boundary (after z_H is updated by z_L), but ONLY on the z_H output (not z_L intermediates, which failed before in batch 1). Weight later cycles exponentially: w_i = 2^i / sum(2^j).
Confidence: 7
Why: z_H is the "answer" state and should be supervised at each cycle.
Time of idea generation: 2026-03-18 10:30
Status: Failed
HPPs: exp weighting 2^i, Huber+cos at each H_cycle output
Time of run start and end: 2026-03-18 12:00 - 2026-03-18 13:15
Results vs. Baseline: 316.78 FID vs 314.38 baseline. +2.40 worse.
wandb link: multi_step_velocity_loss
Analysis: The extra output head evaluations per H_cycle slow training (~5% fewer steps). The no-grad H_cycles produce detached z_H outputs — supervising those with gradients flowing through the output head but NOT through the recursion gives a confusing signal. The deep supervision approach doesn't map well to truncated BPTT where most cycles are in no-grad mode.
Conclusion: Deep supervision doesn't help when most H_cycles are gradient-truncated. The final H_cycle already gets good gradient signal.
Next Ideas to Try: N/A
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: learned_injection_gate
Description: Replace the additive injection `x = x + input_injection` in l_level with a learned gating mechanism: `x = x + sigmoid(linear(x)) * input_injection`. This lets the model learn HOW MUCH of the injection to accept at each position. Some patches may need more refinement (complex regions) while others are already good (uniform regions). The gate is a single Linear(n_embd, n_embd) layer + sigmoid, adding ~590K params (small relative to 13.6M).
Confidence: 7
Why: Currently all patches receive identical injection strength. But image complexity varies spatially — a sky patch is simple while a face patch is complex. A learned gate would let the model adaptively control information flow. This is related to highway networks (Srivastava et al. 2015) and LSTM gating. Gate should be zero-initialized so it starts as standard additive (no regression from baseline).
Time of idea generation: 2026-03-18 10:30
Status: Implemented, not tried
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: denoising_as_recursion
Description: Unify the diffusion ODE steps with the recursive cycles. During training, instead of predicting velocity at a random single timestep, run the model recursively where each L-cycle takes one denoising step. Start from noise x_0 (t=0), and at each L-cycle predict velocity at t=cycle/total_cycles, step forward, feed result back. Loss on final denoised image vs clean target. At eval, the standard ODE solver still works (model predicts velocity at any t). But training teaches the shared block to be a good single-step denoiser, which should improve the Euler ODE quality.
Confidence: 6
Why: The recursive architecture naturally maps to iterative denoising. Currently recursion and ODE are separate (recursion refines within a timestep, ODE integrates across timesteps). Unifying them means the model practices actual denoising during training, not just single-timestep velocity prediction. This is conceptually similar to consistency models but uses the existing recursive structure. Risk: the training signal may be too noisy since early cycles see poor predictions.
Time of idea generation: 2026-03-18 10:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: token_mixing_mlp
Description: Replace self-attention in RecursiveDiTBlock with a simple token-mixing MLP (MLP-Mixer style). Instead of Q/K/V attention over patches, transpose to (B, C, N) and apply a linear layer across the N (spatial) dimension. This is O(N) instead of O(N^2), much faster, and the shared-weight recursion provides the "depth" that attention normally gives. With 256 patches and 768 channels, the token-mixing linear is 256x256 = 65K params per block (vs ~2.4M for attention). Much cheaper, so we get more training steps per hour.
Confidence: 6
Why: The recursive architecture already provides iterative refinement — each pass through the shared block is like another "layer" of processing. Attention's main benefit is long-range spatial mixing, which the recursion also provides (each cycle mixes information globally via the z_H/z_L interaction). MLP-Mixer has shown competitive performance to attention in vision tasks. With 8 recursive passes, even simple per-pass mixing should work. The speed gain means 2-4x more training steps in the same 1-hour budget.
Time of idea generation: 2026-03-18 10:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: target_ema_teacher
Description: Use the EMA model as a teacher during training (not just for eval). At each step, compute the EMA model's velocity prediction (detached) and add a distillation loss: MSE(student_pred, teacher_pred). The EMA model is a smoothed version of the student and acts as a stable target, similar to BYOL/target networks in RL. This is different from our earlier self-distillation (which used deeper recursion from the same model). Weight: 0.05.
Confidence: 6
Why: EMA gave us -42 FID at eval time, meaning the EMA weights produce much better predictions. But currently we only USE the EMA at eval — we don't TRAIN toward it. By distilling from EMA during training, the student model gets a smoother, more stable learning signal. This is the same principle as Polyak-Ruppert averaging in optimization theory, but applied as a loss term. The risk is that it creates a circular dependency (teacher is a lagged copy of student), but this has been shown to work in BYOL, DINO, and other self-supervised methods.
Time of idea generation: 2026-03-18 10:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: per_patch_loss_weighting
Description: Compute the velocity loss per-patch (not per-pixel averaged), then weight each patch's loss by its prediction error magnitude from the EMA model. Patches where the EMA model has high error are "hard" patches — weight them 2x. Patches where the EMA model is accurate are "easy" — weight them 0.5x. This is curriculum learning at the spatial level, focusing compute on the hardest image regions.
Confidence: 5
Why: FID is sensitive to failure modes — a few badly generated patches can tank the whole score. By focusing training on hard patches (faces, textures, fine details), we fix the worst failure modes first. The EMA model provides a stable difficulty estimate. Similar to focal loss but spatially localized. Risk: could overfit to hard patches and neglect easy ones.
Time of idea generation: 2026-03-18 10:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: stochastic_recursion_depth
Description: During training, randomly sample the total number of recursive iterations from a distribution (e.g., uniform 4-12 instead of fixed 8). At eval, use the full 8. This forces the model to produce good outputs regardless of recursion depth — like stochastic depth for recursive networks. The model must learn to make each iteration count, since it doesn't know how many it'll get.
Confidence: 5
Why: The model currently always gets exactly 8 L-level passes. It may learn to "spread out" its computation across all 8, doing a little each step. With stochastic depth, it must be prepared to output a good prediction after any number of steps — encouraging each step to be maximally useful. This is related to Graves' ACT (Adaptive Computation Time) and regularization via randomized depth (Huang et al. 2016). Risk: may hurt early training when the model hasn't learned basic patterns yet.
Time of idea generation: 2026-03-18 10:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: contrastive_velocity
Description: Add a contrastive loss on velocity predictions: velocities for images of the SAME class should be more similar than velocities for DIFFERENT classes (at the same timestep t). Use InfoNCE with temperature 0.1. This encourages class-conditional structure in the velocity field. Implementation: within each batch, compute pairwise cosine similarity of velocity predictions grouped by class label.
Confidence: 4
Why: The model has a class conditioning mechanism via label embeddings, but the velocity loss (Huber) doesn't explicitly encourage class-dependent structure. A contrastive term would push the model to produce systematically different velocities for different classes, improving class-conditional generation quality which directly impacts FID (which measures per-class quality). Risk: may be too expensive (O(B^2) pairwise computation) and the signal may be noisy since velocity also depends heavily on t and noise.
Time of idea generation: 2026-03-18 10:30
Status: Failed
HPPs: InfoNCE temperature=0.1, contrastive weight=0.01
Time of run start and end: 2026-03-18 12:00 - 2026-03-18 12:01
Results vs. Baseline: NaN divergence at step 4
wandb link: N/A (crashed immediately)
Analysis: InfoNCE with temperature 0.1 caused NaN. The velocity vectors in a batch are from different timesteps t, making cosine similarity between same-class samples meaningless — the velocity magnitude and direction depend heavily on t, not just class. The contrastive signal was noise, not signal.
Conclusion: Contrastive loss on velocity is fundamentally flawed because velocity depends on timestep more than class. Would need to stratify by t-bucket to make this work, but that's too complex.
Next Ideas to Try: N/A — this direction is a dead end.
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: residual_prediction
Description: Instead of predicting the full velocity v, predict a RESIDUAL on top of a simple analytical estimate. The analytical estimate is: v_simple = (x_clean - x_noisy) / (1 - t) ≈ (x_t - noise_estimate) / t. Since we have x_t and t, we can compute a naive velocity as v_naive = -x_t / (1-t) (pointing toward origin). The model then predicts delta_v such that v = v_naive + delta_v. This makes the prediction task easier — the model only needs to predict the correction, not the full velocity.
Confidence: 4
Why: Residual learning (He et al. 2016) makes optimization easier by letting the model learn corrections rather than full mappings. The velocity field is complex and varies greatly across timesteps. A residual formulation factors out the "obvious" component, letting the model focus on the hard part. Risk: the naive estimate might not be meaningful, and the residual could have worse conditioning than the original.
Time of idea generation: 2026-03-18 10:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: condition_annealing_sampling
Description: From Disney Research's CADS paper — during ODE sampling at eval, anneal the class conditioning strength over the ODE steps. Start with strong conditioning (t=0, noisy), then reduce conditioning as we approach clean image (t=1). This improves diversity without hurting quality. Implementation: modify the model's forward to scale the class embedding by a factor that depends on t: `c_class = label_embed(y) * max(0, 1 - t/0.5)` (full conditioning at t<0.5, linear decay to zero at t=1). Only active at eval, training unchanged.
Confidence: 7
Why: CADS (Disney Research 2024) showed this simple trick significantly improves FID by boosting diversity. It prevents the model from "collapsing" to a single mode per class at the final denoising steps. It's free at training time (no changes needed) and only modifies the eval-time conditioning schedule. Very simple to implement — just add a t-dependent scaling to the class embedding in forward when not training.
Time of idea generation: 2026-03-18 14:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: progressive_growing_recursion
Description: Start training with h_cycles=1, l_cycles=1 (minimal recursion) and progressively increase both over the course of training. First 25%: h=1,l=1. Then 50%: h=1,l=2. Then 75%: h=2,l=2. Final 25%: h=2,l=3 (full). This is curriculum learning on the computation depth — teach the shared block to be a good single-pass processor first, then gradually ask it to refine iteratively. Similar to progressive growing of GANs (Karras 2018) but applied to recursion depth instead of resolution.
Confidence: 6
Why: Progressive training has been shown to help in many settings (ProGAN, curriculum learning). The model currently struggles to learn meaningful recursion in 1 hour because it needs to simultaneously learn (1) good features and (2) how to refine iteratively. By starting with shallow recursion, the model first learns good features, then learns to refine them. The shared weights benefit because early training teaches them to be good single-pass processors, and later training teaches them to compose well.
Time of idea generation: 2026-03-18 14:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: noise_prediction_target
Description: Instead of predicting velocity v = x0 - (1-sigma_min)*noise, predict the noise epsilon directly. Then convert to velocity for the output: v = x0 - (1-sigma_min)*epsilon. The key insight: noise prediction may be an easier learning target for the model because the noise distribution is fixed (standard Gaussian) while the velocity depends on both x0 and noise. The model architecture and eval are unchanged — we just reparameterize the training target. At eval, the model still outputs velocity (we convert internally).
Confidence: 5
Why: Many successful diffusion models (DDPM, ADM) use epsilon prediction rather than velocity. The argument is that predicting what was added (noise) is easier than predicting the direction of the flow (velocity). Since our model is capacity-limited (shared weights), making the prediction task easier could help it learn faster in the 1-hour budget. The risk is that flow matching was designed for velocity prediction, and the reparameterization might lose some of the flow matching benefits (straight ODE paths).
Time of idea generation: 2026-03-18 14:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: spatial_smoothing_conv
Description: Add a lightweight 3x3 depthwise convolution between recursive cycles to provide local spatial mixing between adjacent patches. After each l_level call on z_H, reshape z_H from (B, 256, 768) to (B, 768, 16, 16), apply a 3x3 depthwise conv with zero-initialized weights (starts as identity), reshape back. This addresses the visible blockiness in generated images where 4x4 patches are coherent internally but don't blend with neighbors. The depthwise conv adds only 768*9 = 6912 params.
Confidence: 8
Why: Visual inspection of generated samples shows clear 4x4 patch grid artifacts. The model's attention across 256 patches provides global mixing but misses LOCAL spatial relationships between adjacent patches. A 3x3 depthwise conv is the simplest possible fix — it provides nearest-neighbor blending without the O(N^2) cost of attention. Zero-init means it starts as a no-op (no regression risk). This is similar to how ConvNeXt adds local spatial mixing to vision transformers.
Time of idea generation: 2026-03-18 21:30
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: spatial_smooth_only_output
Description: Apply the spatial smoothing conv ONLY on the final z_H output (after all recursion), not at every l_level call. This should be much faster (1 conv instead of 8+) while still fixing the patch grid artifacts at the output stage. The hypothesis is that intermediate spatial mixing isn't needed — what matters is blending the patches before the output head.
Confidence: 8
Why: The full spatial_smoothing_conv improved FID (314.29 vs 314.38) but was 4x slower (4727 vs 18000 steps). If we only smooth at the output, we get ~17000 steps with the smoothing benefit. The question is whether intermediate smoothing helps the recursion or if output-stage smoothing is sufficient. Since the model's attention already provides global mixing, the conv mainly needs to fix the final patch boundaries.
Time of idea generation: 2026-03-18 22:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: spatial_smooth_every_other
Description: Apply spatial smoothing conv every OTHER l_level call instead of every call. This halves the conv overhead while still providing periodic local mixing during recursion. The model gets ~9000 steps (2x slower, vs 4x for every-call).
Confidence: 7
Why: Balances the speed/quality tradeoff of spatial smoothing. Every-call was 4x slower but helped. Every-other should be 2x slower but might capture most of the benefit. The hypothesis is that alternating between global attention and local conv mixing creates a good multi-scale processing pipeline.
Time of idea generation: 2026-03-18 22:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: bigger_smooth_kernel
Description: Increase the spatial smoothing kernel from 3x3 to 5x5 depthwise conv. This provides wider local mixing (2 patch radius instead of 1). With 16x16 patch grid, a 5x5 kernel covers ~10% of the grid in each pass. Over 8 recursive passes, information can propagate across the full grid. Still depthwise so param count is small: 768*25 = 19.2K params.
Confidence: 7
Why: The 3x3 kernel was the key breakthrough. A 5x5 kernel provides wider local mixing per step, potentially reducing the number of recursive passes needed for full spatial coherence. The tradeoff is slightly more compute per conv (25 vs 9 multiply-adds per position), but the conv is already a small fraction of the total compute.
Time of idea generation: 2026-03-18 23:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: spatial_smooth_plus_cond_anneal
Description: Combine spatial smoothing (best architectural change) with condition annealing at eval (CADS-style). The spatial smoothing addresses quality (patch coherence) while condition annealing addresses diversity (prevents mode collapse in late ODE steps). Together they may stack for a bigger FID improvement.
Confidence: 7
Why: These address orthogonal problems. Spatial smoothing fixes quality (blockiness), condition annealing fixes diversity. FID measures both quality AND diversity, so fixing both should be additive.
Time of idea generation: 2026-03-18 23:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: wide_output_head
Description: Replace the single Linear output projection with a "shallow yet wide" head: Conv1x1(n_embd -> 4*n_embd) -> GELU -> Conv1x1(4*n_embd -> patch_size^2*3). This gives the output head more capacity to map from latent z_H to pixel-space velocity, without changing the recursive architecture. Inspired by RAE (Representation Autoencoders for DiT, 2025). Current head is just a single Linear(768 -> 48). The wide head would be Linear(768 -> 3072) -> GELU -> Linear(3072 -> 48).
Confidence: 7
Why: The output head is a massive bottleneck — it maps from 768-dim latent to 48-dim (4x4x3 patch) with a single linear layer. This is an extreme compression. A wider head with nonlinearity gives the model more expressive power at the output stage, which directly impacts generation quality. The recursion produces good latent features but the output head can't decode them well.
Time of idea generation: 2026-03-19 01:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: patch_overlap_embed
Description: Use overlapping patches for the patch embedding: instead of non-overlapping 4x4 patches (stride=4), use 6x6 patches with stride=4 (2-pixel overlap on each side). This provides local context at each patch boundary, helping the model produce spatially coherent predictions. The Conv2d becomes (3, n_embd, 6, 6, stride=4, padding=1). Output is still 16x16 patches = 256 tokens. This naturally smooths patch boundaries without needing the spatial smoothing conv.
Confidence: 7
Why: The blocky patch artifacts we observed are caused by non-overlapping patches with hard boundaries. Overlapping patches naturally share information at boundaries, providing the same spatial mixing that our spatial smoothing conv does but at the input level. This could be faster than the spatial smoothing conv (which runs 8+ times) since it only runs once at input.
Time of idea generation: 2026-03-19 01:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: conv1d_neighbor_mixing
Description: Replace the 2D spatial smooth (which requires costly reshape from (B,N,C) to (B,C,H,W) and back) with a 1D convolution that mixes neighboring tokens in the flattened sequence. Since patches are in raster-scan order, a 1D conv with kernel_size=3 mixes each patch with its left and right neighbors. To also mix with vertical neighbors (stride=16 apart), add a second 1D conv on the permuted sequence. This avoids the expensive 2D reshape entirely while still providing spatial mixing. Total new params: 768*3*2 = 4608 (tiny).
Confidence: 8
Why: The 2D reshape (B,256,768) -> (B,768,16,16) and back accounts for most of the spatial smoothing overhead. A 1D conv on the token sequence avoids this entirely. Raster-scan order means kernel_size=3 naturally mixes horizontal neighbors. For vertical mixing, we can use dilated 1D conv with dilation=16 (the grid width). This gives us both horizontal and vertical local mixing without any reshape. Should be much faster, potentially recovering the 3.5x throughput loss.
Time of idea generation: 2026-03-19 06:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

---
idea_id: full_bptt_lr2e4
Description: Combine full BPTT (backprop through all H_cycles, no gradient truncation) with learning rate 2e-4 (2x the original 1e-4). Full BPTT provides complete gradient information, and the higher LR allows faster convergence within the 1-hour budget. The full gradient signal means the optimizer can make larger, more informed steps.
Confidence: 10
Why: Full BPTT alone gave 259.42 (from 314.09). LR 2e-4 with full BPTT gave 230.05. Both are validated. The combination is the current best.
Time of idea generation: 2026-03-19 08:00
Status: Success
HPPs: solver=tbptt, lr=2e-4, full_bptt=true, h_cycles=2, l_cycles=3, spatial_smooth=3x3_depthwise, ema=progressive_0.99_to_0.9999
Time of run start and end: 2026-03-19 06:00 - 2026-03-19 08:00
Results vs. Baseline: 230.05 FID vs 259.42 previous best vs 359.63 original baseline. -129.6 total improvement!
wandb link: full_bptt_lr_2e-4
Analysis: Full BPTT was the breakthrough — gradient truncation was catastrophically wrong for diffusion. With full gradients, the optimizer can use higher LR (2e-4 vs 1e-4) because it has complete information. The model now gets meaningful gradient signal through ALL recursive cycles. Spatial smooth is still important (no_smooth=284.87 vs with_smooth=230.05). h_cycles=3 (265.47) was worse than h_cycles=2 — more cycles increase compute per step, reducing total steps in the 1-hour budget.
Conclusion: Full BPTT + higher LR is the dominant configuration. The recursive shared-weight architecture benefits enormously from complete gradient flow.
Next Ideas to Try: See below.
---

---
idea_id: full_bptt_lr3e4
Description: Push learning rate even higher to 3e-4 with full BPTT. Since 2e-4 improved over 1e-4 with full gradients, 3e-4 may further improve. The risk is instability, but the Huber loss and progressive EMA provide robustness.
Confidence: 6
Why: 1e-4→2e-4 gave -29 FID improvement. The relationship between LR and FID may not be linear — there's likely an optimal LR. 3e-4 was tried without full BPTT and was catastrophic (347.17), but full BPTT changes the optimization landscape completely. With full gradient information, higher LR is better supported. However, there's a ceiling — too high will cause divergence.
Time of idea generation: 2026-03-19 09:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: full_bptt_lr5e4
Description: Try LR 5e-4 with full BPTT. More aggressive than 3e-4 but tests the upper bound of the LR-FID curve. If this diverges, we know the optimal LR is between 2e-4 and 5e-4.
Confidence: 4
Why: Bracketing the optimal LR is important. 2e-4 worked, 3e-4 might work, 5e-4 tests the boundary. The Huber loss provides some robustness against instability.
Time of idea generation: 2026-03-19 09:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: full_bptt_remove_spatial_smooth
Description: With full BPTT (which provides complete gradient information through all recursive cycles), the spatial smoothing conv may no longer be needed. Full BPTT lets the attention mechanism learn proper cross-patch relationships through gradient flow. Without spatial smooth, we get ~3.5x more training steps (18000 vs 4700). This is a simplification experiment — if FID is comparable, we get massive speed gains.
Confidence: 5
Why: no_smooth with full BPTT got 284.87, which is much better than the old truncated BPTT baseline (314). But still worse than full BPTT with smooth (259.42). However, no_smooth gets 3.5x more steps. The question is whether 3.5x more steps at lower quality-per-step can beat fewer high-quality steps. At LR 2e-4, no_smooth might be competitive since it sees much more data.
Time of idea generation: 2026-03-19 09:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: full_bptt_no_smooth_lr2e4
Description: Combine the two: remove spatial smooth AND use LR 2e-4 with full BPTT. If spatial smooth isn't needed with full gradients, this gives us ~18000 steps at 2e-4 LR with full BPTT — the maximum possible training throughput with the strongest optimization signal.
Confidence: 6
Why: This tests whether the speed advantage of no spatial smooth (3.5x more steps) compensates for the quality loss. At LR 2e-4 with full BPTT, the model learns much faster per step. 18000 steps of fast learning might beat 4700 steps of slow learning + spatial smoothing. This is the "simplicity win" we should test.
Time of idea generation: 2026-03-19 09:00
Status: Running
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

---
idea_id: full_bptt_lr8e4
Description: Fine-tune the LR between 7e-4 (FID 95.87) and 1e-3 (FID 97.90). Try 8e-4 to find the exact optimal. The LR curve shows 7e-4 is slightly better than 1e-3, suggesting the peak is near 7e-4 but could be 8e-4 or 9e-4.
Confidence: 7
Why: We have a clear LR→FID curve: 5e-4=111.5, 7e-4=95.9, 1e-3=97.9, 2e-3=268 (diverged). The peak appears near 7e-4 but could be slightly higher. 8e-4 splits the difference. High confidence because we know the range works.
Time of idea generation: 2026-03-19 14:00
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
idea_id: full_bptt_lr7e4_no_freeze_smooth
Description: At LR 7e-4 with full BPTT, test whether the "freeze spatial smooth for first 20%" curriculum is still optimal. With the much higher LR, the model converges faster and the smooth may need to activate sooner or later. Try without the freeze (smooth active from step 0).
Confidence: 5
Why: The freeze curriculum was optimized at LR 1e-4. At 7x the LR, training dynamics are completely different. The model reaches the equivalent of "20% progress" much faster in terms of feature quality. The smooth might benefit from being active earlier. Conversely, at high LR the early training is noisier and smoothing early might help stabilize it.
Time of idea generation: 2026-03-19 14:00
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
idea_id: full_bptt_lr7e4_weight_decay_001
Description: Add weight decay 0.01 at LR 7e-4 with full BPTT. High LR benefits from regularization to prevent overfitting. Weight decay was tested before at lower LRs and was neutral/slightly helpful.
Confidence: 5
Why: At LR 7e-4, the model updates are 7x larger than the original 1e-4. This increases risk of overfitting or noisy weights. Weight decay provides a regularizing force that may improve generalization (FID) even if training loss is slightly higher.
Time of idea generation: 2026-03-19 14:00
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
idea_id: full_bptt_larger_batch_512
Description: With full BPTT at LR 9e-4, try batch size 512 (grad_accum=8 instead of 4). Larger batch gives more stable gradients which may allow the model to converge better. At the current ~800ms/step, 4700 steps * 512 images = 2.4M images seen. Previous attempt at batch 512 without full BPTT was worse, but full BPTT changes everything.
Confidence: 5
Why: Larger batches reduce gradient noise. With full BPTT + high LR, the gradients are already high-quality but could be noisy. Batch 512 smooths this. The tradeoff is fewer optimizer steps (2350 vs 4700), but each step is more informative. Many diffusion models train with large batches (256-1024).
Time of idea generation: 2026-03-19 18:00
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
idea_id: full_bptt_remove_reverse_pass
Description: Test if the symmetric reverse pass is still needed with full BPTT. Without truncation, the forward L-cycles already get full gradient signal. The reverse pass doubles the L-level calls in the last H_cycle. Removing it would speed up each step.
Confidence: 5
Why: The reverse pass was added when we had truncated BPTT and needed extra refinement. With full gradients, the model learns better representations in the forward pass. Removing the reverse pass saves compute and may not hurt FID. This is a simplification experiment.
Time of idea generation: 2026-03-19 18:00
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
idea_id: full_bptt_color_jitter
Description: Add random color jitter augmentation (brightness, contrast, saturation) during training. This is orthogonal to hflip and provides more data diversity. Standard augmentation for vision models.
Confidence: 5
Why: Hflip gave -10 FID. Color jitter is another standard augmentation that's orthogonal (spatial vs color). ImageNet has diverse lighting conditions, and jitter forces the model to be robust to color variations. Simple to implement with torchvision transforms.
Time of idea generation: 2026-03-19 18:00
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
idea_id: full_bptt_random_crop_aug
Description: Instead of just hflip, add random crop augmentation. Currently prepare.py uses RandomCrop(64) on training data, but we can add an additional random crop+resize at training time: crop a random 48x48-64x64 region and resize to 64x64. This provides scale augmentation that's different from hflip's spatial augmentation. Must apply to both x and velocity targets consistently.
Confidence: 6
Why: Hflip gave -10 FID at the optimal LR. Random crops provide scale diversity — the model sees the same object at different scales. Color jitter failed (corrupted targets), but geometric augmentations like crop are safe because they apply equally to x_t and velocity. The key is to apply the crop BEFORE flow matching forward_sample so both noisy sample and velocity target are consistent.
Time of idea generation: 2026-03-20 01:00
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
idea_id: full_bptt_mixup_labels
Description: Mixup on class labels only (not images): for 20% of samples, replace the class label with a random different class. This is like label smoothing but more aggressive — it forces the model to not rely too heavily on class conditioning and instead focus on the image content. Should improve diversity.
Confidence: 4
Why: Class conditioning is important for FID (class-conditional generation quality). But over-reliance on labels can reduce diversity within each class. Random label replacement forces the model to generate plausible images even with "wrong" labels, which may improve the diversity component of FID. Risk: could hurt class-conditional quality.
Time of idea generation: 2026-03-20 01:00
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
idea_id: full_bptt_warm_restart_lr
Description: Instead of linear warmup + linear warmdown, use a warm restart (cosine annealing with restarts). Reset the LR back to peak every 25% of training. This is the SGDR schedule from Loshchilov & Hutter 2017. Each restart allows the optimizer to escape local minima and explore new regions of the loss landscape.
Confidence: 5
Why: With full BPTT and high LR, the model converges fast but may get stuck. Warm restarts provide periodic "jolts" that can escape local minima. The cosine annealing within each cycle ensures smooth convergence between restarts. SGDR has shown benefits in vision tasks. The risk is that restarts may undo good progress.
Time of idea generation: 2026-03-20 01:00
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
idea_id: beta2_095_discovery
Description: Changed AdamW beta2 from 0.999 to 0.95 (matching TRM's optimizer setting). This makes the second moment estimate more responsive to recent gradient magnitudes, which is critical at high LR (1.5e-3) where the loss landscape changes rapidly. The default 0.999 averages over ~1000 steps of gradient history, while 0.95 averages over ~20 steps — much more responsive.
Confidence: 10
Why: Validated — FID 66.22 vs 75.40 baseline. The TRM paper uses beta2=0.95 in its AdamATan2 optimizer for a reason: recursive models with shared weights have highly non-stationary gradient statistics because the same weights are used at different recursion depths.
Time of idea generation: 2026-03-20 10:00
Status: Success
HPPs: lr=1.5e-3, beta1=0.9, beta2=0.95, no EMA, Huber+0.5*cos, full BPTT
Time of run start and end: 2026-03-20 10:00 - 2026-03-20 11:30
Results vs. Baseline: 66.22 FID vs 75.40 baseline. -9.18 improvement!
wandb link: beta2_095_s42
Analysis: beta2=0.95 from TRM was the key missing ingredient. The recursive architecture reuses weights across cycles, creating non-stationary gradient statistics. A lower beta2 adapts faster to the changing gradient landscape at each recursion depth. This is why TRM uses beta2=0.95 — it's not arbitrary, it's essential for recursive architectures.
Conclusion: beta2=0.95 is optimal for recursive shared-weight architectures. This is a transferable insight from TRM.
Next Ideas to Try: Explore beta2 between 0.9-0.99 to find exact optimum. Also try beta1=0.95 (TRM uses this too).
---

---
idea_id: l1_cos_beta95_lr17e3
Description: Combine the three recent wins: L1 loss (replaced Huber, FID 61.9 vs 66.2), beta2=0.95 (from TRM), and LR 1.7e-3 (which scored 65.7 with Huber). The hypothesis is that L1's improved gradient signal allows the model to tolerate even higher LR with beta2=0.95. L1 provides constant gradient magnitude regardless of error size, which may interact well with beta2=0.95's fast second-moment adaptation.
Confidence: 7
Why: L1 gave -4.3 FID over Huber. LR 1.7e-3 gave -0.5 FID over 1.5e-3 with Huber. These are independent improvements that should stack. L1's uniform gradient magnitude means the optimizer's second moment estimate (controlled by beta2) is more predictable, potentially allowing even higher LR. All three components are validated independently.
Time of idea generation: 2026-03-20 16:00
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
idea_id: l1_cos_beta95_beta1_095
Description: TRM uses both beta1=0.95 and beta2=0.95 in its optimizer. We only changed beta2. Try beta1=0.95 too (from default 0.9). Higher beta1 means less momentum — the optimizer responds faster to gradient direction changes, which may help with the recursive architecture's non-stationary gradients.
Confidence: 6
Why: If beta2=0.95 helped because recursive architectures have non-stationary gradient statistics, then beta1=0.95 should help for the same reason. The first moment estimate (momentum) also benefits from faster adaptation when gradients change direction across recursion cycles. TRM uses this setting for good reason.
Time of idea generation: 2026-03-20 16:00
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
idea_id: l1_only_no_cosine
Description: Test if cosine loss is still needed with L1. L1 already encourages both direction and magnitude matching (unlike MSE which overweights large errors). The cosine term was critical with Huber/MSE, but L1 may subsume its benefit. Removing cosine simplifies the loss to just L1.
Confidence: 5
Why: L1 loss applies equal gradient pressure to all dimensions of the velocity vector, which naturally encourages directional alignment. Cosine was added to compensate for MSE/Huber's tendency to focus on large-error dimensions at the expense of directional accuracy. With L1, this compensation may be unnecessary. If confirmed, this is a major simplification.
Time of idea generation: 2026-03-20 16:00
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
idea_id: l1_cos1_lr17e3_b95
Description: Combine the three best findings: L1 loss, cosine weight 1.0, LR 1.7e-3, beta2=0.95. LR 1.7e-3 scored 61.1 with cos=0.5, and cos=1.0 scored 59.6 at LR 1.5e-3. Combining both should give the best result. This is a principled combination of validated improvements.
Confidence: 7
Why: Each component is independently validated. cos=1.0 gave -2.4 FID. LR 1.7e-3 gave -0.8 FID. If additive, expect ~57 FID. These should stack because they affect different aspects: LR controls convergence speed, cosine weight controls loss balance.
Time of idea generation: 2026-03-20 20:00
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
idea_id: l1_cos15_b95
Description: Try cosine weight 1.5 (even higher than 1.0). If cos=1.0 beat cos=0.5, maybe cos=1.5 is even better. The cosine term encourages directional alignment which is critical for ODE sampling quality. More weight = more directional pressure.
Confidence: 5
Why: cos=0.5→1.0 gave -2.4 FID. The trend might continue. However, there's a point where too much cosine pressure hurts magnitude prediction. L1 already provides magnitude gradient, so the optimal balance might favor higher cosine weight. Risk: cos too high could make the model ignore magnitude entirely.
Time of idea generation: 2026-03-20 20:00
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
idea_id: velocity_norm_loss
Description: Replace L1+cosine with a single loss that combines both: L1 on the unit-normalized velocity (direction) + L1 on the scalar magnitude. This is mathematically cleaner than L1+cosine because it explicitly decomposes the velocity into direction and magnitude components. loss = L1(pred/|pred|, vel/|vel|) + 0.1*L1(|pred|, |vel|). The direction component gets most weight since FID cares more about direction (confirmed by cosine loss importance).
Confidence: 5
Why: The current L1+cosine loss has redundant components — L1 already captures direction implicitly. By explicitly decomposing into direction and magnitude, we can control the balance more precisely. The 10:1 ratio favoring direction matches our finding that cosine (direction) is more important than magnitude. Novel loss formulation not found in literature.
Time of idea generation: 2026-03-20 22:00
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
idea_id: gradient_accumulation_1_devbatch256
Description: Remove gradient accumulation entirely: device_batch=256, total_batch=256, accum=1. Currently we use device_batch=64, accum=4. With accum=1, we do 1 forward+backward per optimizer step instead of 4. This means the optimizer updates more frequently with noisier gradients. At high LR with beta2=0.95, noisier but more frequent updates might be better — similar to how SGD with small batch can outperform large batch in some regimes.
Confidence: 5
Why: At LR 2e-3 with beta2=0.95, the optimizer already handles noisy gradients well. More frequent updates (4x more per hour) means the model sees more optimization steps. The tradeoff is noisier gradients, but beta2=0.95 adapts fast. This also increases VRAM usage (256 images at once instead of 64) which utilizes the A100 better.
Time of idea generation: 2026-03-20 22:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---
