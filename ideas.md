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

---
idea_id: multi_scale_patch_2_4
Description: Use TWO patch sizes simultaneously: patch_size=4 (16x16=256 tokens, current) AND patch_size=8 (8x8=64 tokens, coarse). Process the coarse patches first with the shared L-level block, then upsample and add as conditioning to the fine patch processing. This gives the model a multi-resolution view without the cost of patch_size=2 (which used 33GB VRAM). The coarse path is cheap (64 tokens) and provides global structure that guides fine detail generation. Implementation: add a second PatchEmbed with kernel=8, process coarse tokens through l_level once, bilinear upsample to 16x16, add to z_H before fine processing.
Confidence: 5
Why: Multi-scale processing is fundamental in vision (U-Net, FPN, etc). Our model processes at a single scale (4x4 patches). Coarse-to-fine is how humans perceive images. The coarse pass adds ~10% compute (64 tokens vs 256) but provides global context. patch_size=2 was too expensive (33GB), but patch_size=8 is cheap (64 tokens, 4x less attention compute than current). Risk: the shared block weights may not generalize well across scales.
Time of idea generation: 2026-03-21 01:00
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
idea_id: inject_noise_level_to_recursion
Description: Currently the model gets the timestep t via AdaLN conditioning on the transformer blocks. But the RECURSION STRUCTURE doesn't know about t — it always does the same number of cycles regardless of noise level. What if we inject t directly into the recursive loop? At high noise (t near 0), the model should do more refinement. At low noise (t near 1, almost clean), less refinement is needed. Implementation: at each L-cycle, multiply the input injection by (1 + (1-t)), so noisy inputs get 2x injection strength while clean inputs get 1x. This is like a noise-conditioned recursion depth without actually changing the number of cycles.
Confidence: 4
Why: The model currently treats all timesteps equally in terms of recursion computation. But denoising from heavy noise requires more iterative refinement than polishing an almost-clean image. By scaling the injection strength by noise level, we give the model adaptive processing depth without changing the architecture. The AdaLN already provides timestep conditioning to the blocks, but the INJECTION PATHWAY doesn't know about t. Risk: might destabilize the recursive dynamics.
Time of idea generation: 2026-03-21 01:00
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
idea_id: learned_timestep_weighting
Description: Instead of uniform timestep sampling t~U[0,1], learn which timesteps matter most for FID. Use a simple heuristic: weight timesteps by their loss magnitude. At each step, compute per-sample loss, then for the next batch, sample t proportional to the running average loss at each t. This focuses training on the timesteps the model struggles with most. Implementation: maintain a 100-bin histogram of average loss per t-bin, sample t from this distribution (softmax-normalized). Update the histogram with EMA.
Confidence: 5
Why: Uniform t sampling wastes compute on easy timesteps. FID is determined by the quality of the full ODE trajectory — if certain timesteps have high loss, those are the bottleneck. Importance sampling of t based on loss is a principled way to allocate training compute. This is related to P2 weighting (Choi et al. 2022) but learned adaptively rather than fixed. Risk: the loss landscape changes during training, so the weighting needs to adapt.
Time of idea generation: 2026-03-21 04:00
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
idea_id: residual_z_H_accumulation
Description: Instead of overwriting z_H at each H_cycle boundary (z_H = l_level(z_H, z_L, c)), use a residual connection: z_H = z_H + alpha * l_level(z_H, z_L, c) where alpha is a learnable scalar initialized at 0.1. This prevents catastrophic overwriting of z_H's accumulated information and allows each H_cycle to make incremental corrections. Similar to how ResNets learn residual corrections rather than full mappings.
Confidence: 5
Why: Currently z_H is fully replaced at each H_cycle. With full BPTT, the gradient flows through these replacements, but the signal may still be lossy. A residual connection preserves information from earlier cycles while allowing refinement. The learnable alpha controls the step size of each refinement. This is a fundamental architectural change that makes the recursion more like an ODE (continuous refinement) rather than a sequence of discrete rewrites. Risk: the model might not converge if alpha is too large.
Time of idea generation: 2026-03-21 04:00
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
idea_id: multi_crop_ensemble_training
Description: During each training step, process the SAME image at 3 different random crops (center 64x64, random 56x56→resize to 64, random 48x48→resize to 64) and average the velocity predictions before computing loss. This is like a "model ensemble during training" — the model sees each image at multiple scales simultaneously. The averaged prediction should be smoother and more accurate. This uses 3x VRAM (26GB, still well within 80GB) but gives a much richer training signal per image. Implementation: for each batch, create 3 augmented versions, forward all 3, average predictions, compute single loss against the original-scale velocity.
Confidence: 5
Why: Multi-crop evaluation is standard in image classification (improves accuracy by 1-2%). Applying this during TRAINING forces the model to produce scale-consistent velocity predictions, which should improve the ODE trajectory quality. The 3x compute cost is affordable (we're at 11% GPU utilization). The key insight is that FID measures the full distribution, and scale consistency helps coverage.
Time of idea generation: 2026-03-21 02:00
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
idea_id: laplace_noise_schedule
Description: Replace uniform timestep sampling t~U[0,1] with a Laplace (double-exponential) distribution centered at t=0.5. The Laplace distribution puts more probability mass near t=0.5 (medium noise) and less at the extremes (t~0 pure noise, t~1 clean). This was shown in "Improved Noise Schedule for Diffusion Training" (ICCV 2025) to give 25.9% FID improvement over uniform sampling. Implementation: sample from Laplace(loc=0.5, scale=0.2), clamp to [0,1]. Very simple change — just one line of code.
Confidence: 7
Why: Published result from ICCV 2025 showing 25.9% FID improvement. The intuition is that medium-noise timesteps (t~0.5) are the most informative for learning — at t~0 the input is pure noise (hard to learn from), at t~1 it's nearly clean (easy, nothing to learn). Laplace focuses training on the "goldilocks zone" where the model can make meaningful progress. This is backed by peer-reviewed results and is trivial to implement.
Time of idea generation: 2026-03-21 12:00
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
idea_id: skip_connection_dit
Description: Add a long skip connection from the input embedding directly to the output head, bypassing the entire recursive structure. This is inspired by "Skip-DiT" (ICCV 2025) which showed that adding skip connections to DiT achieves 4x faster convergence. Implementation: out = final_proj(z_H_processed) + 0.1 * final_proj(input_emb). The skip provides a "shortcut" that lets easy parts of the velocity be predicted directly while the recursion handles the hard parts.
Confidence: 5
Why: Skip-DiT achieved FID parity with DiT in 1.6M steps vs 7M — a 4x speedup. Our recursive model already has input injection at each cycle, but no DIRECT skip from input to output. Adding one gives the model a "path of least resistance" for easy velocity components. This could improve convergence speed, which matters in our 1-hour budget. Risk: the skip might dominate and the recursion becomes unused.
Time of idea generation: 2026-03-21 12:00
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
idea_id: curriculum_denoising_recursion
Description: Fundamentally change how the recursive model learns: instead of just predicting velocity at random timesteps, add a curriculum-based denoising objective. During training, corrupt the clean image with noise (starting very light, increasing as the model masters each level), then run the full recursive forward (all H_cycles and L_cycles), and penalize the OUTPUT z_H for not reconstructing the clean image. The noise level increases when training loss plateaus, creating a curriculum from easy (nearly clean) to hard (heavy noise). Each recursive cycle should improve the reconstruction — this is the "monotonic improvement" property. The model still predicts velocity for the ODE sampler (compatibility with prepare.py), but the auxiliary denoising curriculum teaches the shared weights to be effective iterative refiners at ALL noise levels. Implementation: (1) Add Gaussian noise with learnable sigma to clean images, (2) project z_H through output head at each H_cycle boundary, compute reconstruction loss, (3) increase sigma when loss plateaus. The velocity prediction loss (L1+cos) remains as the primary loss, with the denoising curriculum as auxiliary.
Confidence: 7
Why: This is inspired by the user's insight that the TRM recursion should naturally map to iterative denoising. The architecture was DESIGNED for iterative refinement (that's what TRM does for reasoning). The curriculum prevents catastrophic forgetting — the model first masters easy denoising, then progressively harder levels. The monotonic improvement property is achieved by training with multi-step loss (supervise each H_cycle's output). The key advantage: this teaches the shared weights to be good single-step denoisers, which directly improves the ODE sampling quality (each of the 50 Euler steps benefits from better single-step denoising). Data: our model currently achieves FID ~57 with recursive velocity prediction. Adding a denoising curriculum that teaches the RECURSION to refine progressively could unlock the architecture's full potential. Risk: the auxiliary loss might interfere with velocity prediction, and the curriculum tuning could be tricky.
Time of idea generation: 2026-03-22 08:00
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
idea_id: monotonic_multistep_denoising
Description: Simpler version of the curriculum idea: during training, corrupt clean images with noise at the CURRENT flow matching timestep t, run the model's recursive forward, and extract intermediate outputs at each H_cycle boundary. Compute reconstruction loss at each boundary with INCREASING weight (cycle 1 gets weight 0.1, cycle 2 gets weight 1.0). This directly enforces the "each cycle improves" property. The loss is: sum over cycles of w_i * L1(output_head(z_H_at_cycle_i), clean_image). Combined with the velocity loss. This is different from our failed multi-step experiment (batch 1, FID 365) because: (1) we now have full BPTT, (2) we compute loss on z_H (not z_L), (3) we weight later cycles more, and (4) the model is much better trained.
Confidence: 6
Why: Our original multi-step loss experiment failed because it used truncated BPTT and computed intermediates from z_L. With full BPTT and z_H outputs, the gradient signal reaches all shared weights for all cycles. The increasing weight schedule enforces monotonic improvement — later cycles MUST be better. This directly implements the "each iteration only improves FID" property. Risk: the intermediate output head evaluations add compute, reducing total training steps.
Time of idea generation: 2026-03-22 08:00
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
idea_id: imgaug_curriculum_denoise
Description: Use imgaug library for rich, realistic image degradation in the curriculum denoising approach. Instead of simple Gaussian noise, use imgaug's augmenters to create realistic corruptions: (1) Start with light Gaussian blur + slight brightness change, (2) progress to heavier blur + noise + contrast changes, (3) eventually reach salt-and-pepper noise + heavy Gaussian noise + color jitter. The curriculum uses imgaug.augmenters with increasing severity controlled by a single "difficulty" parameter that ramps up as the model masters each level. The model must learn to reconstruct the clean image from these corruptions through its recursive cycles. This teaches the shared weights to be robust iterative refiners. The key insight from the user: we want theta such that 1 iteration does OK, and each additional iteration ONLY improves — never degrades.
Confidence: 7
Why: imgaug provides realistic, diverse corruptions that are much harder than simple Gaussian noise. This forces the model to learn robust features that generalize well. The curriculum prevents the model from being overwhelmed — it first masters easy corruptions before progressing to harder ones. The monotonic improvement property is enforced by supervising intermediate outputs. This is a novel training approach that leverages TRM's recursive architecture in a way that standard DiTs can't — the recursion IS the denoising.
Time of idea generation: 2026-03-22 16:30
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
idea_id: curriculum_denoise_analysis
Description: Post-mortem analysis of the curriculum denoising experiments.
Confidence: N/A
Why: N/A
Time of idea generation: 2026-03-22 20:00
Status: Failed
HPPs: Gaussian curriculum (sigma curriculum 0.05→...) and torch-based curriculum (noise+salt&pepper+brightness)
Time of run start and end: 2026-03-22 16:00 - 2026-03-22 20:00
Results vs. Baseline: Gaussian curriculum: 64.02 FID (vs 57 baseline, WORSE). Torch curriculum: 58.50 FID (within noise but toward worse end).
wandb link: curriculum_denoising_recursion, torch_curriculum_denoise
Analysis: The auxiliary denoising loss doubled forward passes per step (model runs for velocity AND for denoising), halving effective training steps from ~9000 to ~4500 in the 1-hour budget. The denoising signal wasn't valuable enough to compensate for the step reduction. The curriculum mechanism (increase noise when loss plateaus) worked correctly but the noise levels never got very high because the model spent most of its time on the velocity prediction task. The fundamental issue is that auxiliary losses are expensive in a time-constrained setting.
Conclusion: Auxiliary denoising curriculum doesn't help within 1-hour budget. The idea would work better with longer training (where the step reduction matters less) or if the denoising REPLACED the velocity loss entirely. For the current setup, the vanilla L1+cosine+dual-t loss is optimal.
Next Ideas to Try: A version where the recursion IS the denoiser (not auxiliary) — each H_cycle produces a progressively less noisy image. This requires restructuring the model's forward pass to output intermediate images, not velocities. But this would break compatibility with prepare.py's evaluation unless the final output is converted to velocity.
---

---
idea_id: dart_style_spatial_denoising
Description: Inspired by Apple's DART (Denoising Autoregressive Transformer, 2025): instead of treating all 256 patches equally, process them in a spatial order (raster scan) with the recursion revealing patches progressively. Early recursive cycles focus on coarse global structure (low-frequency), later cycles add fine details (high-frequency). Implementation: at each L_cycle within an H_cycle, process patches in groups of 64 (4 groups of 8x8) rather than all 256 at once. Each group conditions on the previous group's output. This creates a coarse-to-fine spatial denoising within the recursion. The first group establishes global structure, subsequent groups refine local details.
Confidence: 4
Why: DART shows that spatial ordering matters for denoising — processing patches in a meaningful order improves coherence. Our model processes all 256 patches simultaneously with global attention, which may be suboptimal. A progressive revelation strategy lets each recursive cycle specialize: early cycles handle coarse structure, later cycles handle details. Risk: the implementation is complex, may break the existing attention pattern, and the overhead of group processing could reduce total training steps.
Time of idea generation: 2026-03-22 22:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
---

-----------------------------------------------------
idea_id: cfg_guidance_1p5
Description: Classifier-free guidance (CFG) — the most impactful inference technique for class-conditional diffusion. During training, randomly drop class labels with probability p_uncond=0.1 (replacing class embedding with zeros). During inference, run both conditional and unconditional forward passes, combine as: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond). Guidance scale 1.5. This is well-established (Ho & Salimans 2022) and routinely improves FID by 30-50%+.
Confidence: 9
Why: CFG is the single most reliable technique for improving FID in class-conditional diffusion models. We're leaving massive quality on the table by not using it. Every competitive diffusion model uses CFG. The training cost is minimal (just masking labels 10% of the time). Our model already supports class conditioning via label_embed — CFG is a natural extension.
Time of idea generation: 2026-03-23 08:00
Status: Success (improved from 55.0 baseline)
HPPs: cfg_scale=1.5, p_uncond=0.1
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 51.33 vs 55.0 baseline, -3.7 FID improvement
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: CFG scale 1.5 improves FID from 55.0 to 51.33, but scale 2.0 is superior.
Next Ideas to Try: Optimize around CFG scale 2.0 — try 1.75 and 2.25 for finer tuning.
-----------------------------------------------------

-----------------------------------------------------
idea_id: cfg_guidance_2p0
Description: CFG with guidance scale 2.0, p_uncond=0.1. Slightly stronger guidance than 1.5.
Confidence: 9
Why: Scale 2.0 is often the sweet spot for 64x64 ImageNet in the literature. Testing to find optimal scale.
Time of idea generation: 2026-03-23 08:00
Status: Success (NEW ALL-TIME BEST!)
HPPs: cfg_scale=2.0, p_uncond=0.1
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 48.34 vs 55.0 baseline, -6.7 FID improvement (12% improvement!)
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: BREAKTHROUGH — CFG scale 2.0 with p_uncond=0.1 gives FID 48.34, breaking the 22-batch plateau at FID 55. This is now the new optimal configuration.
Next Ideas to Try: Fine-tune CFG scale (try 1.75, 2.25). Combine with other orthogonal improvements. Scale up model/training.
-----------------------------------------------------

-----------------------------------------------------
idea_id: cfg_guidance_3p0
Description: CFG with guidance scale 3.0, p_uncond=0.1. Moderate guidance.
Confidence: 8
Why: Testing the guidance scale sweep. 3.0 is a common default in many implementations.
Time of idea generation: 2026-03-23 08:00
Status: Success (improved but not best)
HPPs: cfg_scale=3.0, p_uncond=0.1
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 49.43 vs 55.0 baseline, -5.6 FID improvement
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: CFG scale 3.0 gives FID 49.43 — strong but slightly worse than scale 2.0 (48.34). Oversaturation starting.
Next Ideas to Try: Scale 2.0 confirmed as optimal. Do not go higher.
-----------------------------------------------------

-----------------------------------------------------
idea_id: cfg_guidance_4p0
Description: CFG with guidance scale 4.0, p_uncond=0.1. Stronger guidance, may oversaturate at 64x64.
Confidence: 7
Why: Testing upper range. Higher scales trade diversity for quality. May hurt FID if diversity drops too much.
Time of idea generation: 2026-03-23 08:00
Status: Success (improved but oversaturation starting)
HPPs: cfg_scale=4.0, p_uncond=0.1
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 52.47 vs 55.0 baseline, -2.5 FID improvement
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: CFG scale 4.0 gives FID 52.47 — clear oversaturation degradation vs scale 2.0 (48.34). Too much guidance hurts diversity.
Next Ideas to Try: Do not increase scale beyond 3.0. Focus on scale 2.0.
-----------------------------------------------------

-----------------------------------------------------
idea_id: cfg_guidance_2p0_pu02
Description: CFG scale 2.0 with higher label dropout p_uncond=0.2. More unconditional training may help the model learn better unconditional denoising.
Confidence: 7
Why: p_uncond=0.2 was used in some implementations and can help when the unconditional model is weak. Trade-off: more label dropout means less conditional training signal.
Time of idea generation: 2026-03-23 08:00
Status: Success
HPPs: cfg_scale=2.0, p_uncond=0.2
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 48.93 vs 55.0 baseline, -6.1 FID improvement
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: p_uncond=0.2 gives FID 48.93 — worse than p_uncond=0.1 (48.34). Higher dropout reduces conditional training signal.
Next Ideas to Try: Stick with p_uncond=0.1.
-----------------------------------------------------

-----------------------------------------------------
idea_id: logit_normal_timestep
Description: Replace uniform t~U(0,1) sampling with logit-normal: t = sigmoid(N(0,1)). This concentrates samples around t=0.5 where the denoising signal is richest, while still covering t near 0 and 1. Used in Stable Diffusion 3 (Esser et al. 2024) to improve training efficiency.
Confidence: 7
Why: Uniform sampling wastes training signal on very clean (t~1) and very noisy (t~0) inputs where the model has little to learn. Logit-normal focuses on intermediate timesteps. SD3 showed this matters.
Time of idea generation: 2026-03-23 08:00
Status: Failed (neutral - no improvement)
HPPs: logit_normal_t=True, no CFG
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 54.99 vs 55.0 baseline, ~0 FID change
wandb link:
Analysis: Despite much lower training loss (0.234 vs 0.271), FID is identical to baseline. The lower loss is misleading — logit-normal just avoids sampling hard timesteps near t=0 and t=1, making the average loss lower without improving model quality.
Conclusion: Logit-normal timestep sampling does not improve FID despite lowering training loss. The loss reduction is an artifact of avoiding hard timesteps.
Next Ideas to Try: Logit-normal is not useful on its own. Only worth revisiting if combined with other changes.
-----------------------------------------------------

-----------------------------------------------------
idea_id: cfg_2p0_logit_normal
Description: Combine CFG (scale=2.0, p_uncond=0.1) with logit-normal timestep sampling. Both are orthogonal improvements — CFG improves inference quality, logit-normal improves training efficiency.
Confidence: 8
Why: Combining the two most promising orthogonal improvements. If both work individually, their combination should be even better.
Time of idea generation: 2026-03-23 08:00
Status: Success
HPPs: cfg_scale=2.0, p_uncond=0.1, logit_normal_t=True
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 48.38 vs 55.0 baseline, -6.6 FID improvement (logit-normal adds nothing to CFG)
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: CFG 2.0 + logit-normal gives FID 48.38 — virtually identical to CFG 2.0 alone (48.34). Logit-normal adds nothing on top of CFG.
Next Ideas to Try: Drop logit-normal. Focus on CFG scale 2.0 with p_uncond=0.1 as the new baseline.
-----------------------------------------------------

-----------------------------------------------------
idea_id: cfg_guidance_2p0_pu015
Description: CFG scale 2.0 with p_uncond=0.15. Middle ground between 0.1 and 0.2.
Confidence: 7
Why: Fine-tuning the label dropout rate. 0.15 may be optimal.
Time of idea generation: 2026-03-23 08:00
Status: Success
HPPs: cfg_scale=2.0, p_uncond=0.15
Time of run start and end: 2026-03-23 08:02 - 2026-03-23 09:30
Results vs. Baseline: 49.22 vs 55.0 baseline, -5.8 FID improvement
wandb link:
Analysis: CFG is a massive win — the biggest single improvement since the full BPTT discovery. Guidance scale 2.0 is optimal for our 13.6M recursive DiT on 64x64 ImageNet. Below 2.0 (1.5) under-guides, above 3.0 (4.0) starts oversaturating. p_uncond=0.1 is optimal — higher dropout (0.15, 0.2) hurts because it reduces conditional training signal. Training loss is identical across p_uncond values, so the difference is purely at inference.
Conclusion: p_uncond=0.15 gives FID 49.22 — worse than p_uncond=0.1 (48.34). Confirms p_uncond=0.1 is optimal.
Next Ideas to Try: Stick with p_uncond=0.1. Do not increase label dropout.
-----------------------------------------------------

-----------------------------------------------------
idea_id: gqa_4kv_heads
Description: Replace full multi-head attention (12 Q, 12 K, 12 V heads) with Grouped Query Attention (12 Q heads, 4 shared K/V head groups). This reduces redundant KV projections in the shared recursive block (which is applied 2*h_cycles*l_cycles times). Fewer KV parameters means less parameter saturation in the recursion, and the saved parameters can be repurposed or the model runs faster (more training steps in 1 hour). Based on GQA (Ainslie et al., 2023) which showed competitive quality with fewer parameters.
Confidence: 7
Why: The recursive shared-weight block is applied ~12 times per forward pass. Full KV heads wastes capacity since each recursion sees slightly different input but uses identical KV projections. GQA forces the model to share KV representations across heads, which acts as regularization for the recursive structure. Also makes each step faster = more training steps in 1 hour.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: mag_aware_cosine_loss
Description: Modify the cosine similarity loss to be magnitude-aware. Currently, cosine similarity treats all velocity predictions equally regardless of their magnitude. But high-magnitude velocities (at intermediate timesteps) are more important for sample quality. Reweight: cos_loss = (1 - cos_sim) * (velocity_magnitude / max_velocity_in_batch). This focuses the directional loss on the timesteps that matter most.
Confidence: 6
Why: The current cosine loss is magnitude-blind. A prediction with tiny magnitude but perfect direction contributes the same as a prediction with large magnitude and perfect direction. But the large-magnitude predictions are more consequential during ODE integration. Weighting by magnitude makes the model prioritize getting high-signal timesteps right.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: input_skip_to_output
Description: Add a gated skip connection from the input embedding directly to the final output projection, bypassing the full recursion. The idea is that in flow matching, the velocity prediction often closely resembles the input perturbation direction, especially at intermediate timesteps. A direct skip lets the model learn this "identity-like" component cheaply while the recursion handles the residual. Implementation: final_output = unpatchify(final_proj(z_H) + alpha * linear(input_emb)) where alpha is a learnable scalar initialized to 0.
Confidence: 6
Why: In diffusion models, the velocity field has strong correlation with the noisy input structure. Recursive processing can lose this direct signal through many iterations. A skip connection preserves it. DiT and other successful architectures use similar skip patterns. The gated initialization (alpha=0) ensures it starts as a no-op and only learns the skip if it helps.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: decoupled_mag_dir_head
Description: Split the output head into two separate predictions: a unit direction (normalized velocity direction) and a scalar magnitude. The model predicts direction via the existing output head (then normalizes), and magnitude via a separate small MLP head (global average pooled features → scalar per sample). Final velocity = direction * magnitude. Loss: L1 on magnitude + cosine on direction. This decouples the two learning signals which the model currently conflates.
Confidence: 5
Why: Direction and magnitude are fundamentally different aspects of velocity prediction. Currently the model must simultaneously learn both from a single output, which creates gradient interference. Separating them gives each component its own learning signal. The cosine loss already measures direction; adding explicit magnitude supervision could help. Risk: added complexity may not pay off for a small model.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: patch_norm_before_pos
Description: Add RMS normalization to patch embeddings before adding positional embedding. Currently, raw patch tokens have high variance from image content, which can dominate the positional signal. Normalizing patches first ensures the positional embedding has consistent relative magnitude, which helps attention patterns. Simple 1-line change in PatchEmbed.forward().
Confidence: 5
Why: Raw patch projections can have highly variable magnitudes depending on image content (bright vs dark patches). Normalization before positional embedding ensures position information is consistently injected, which helps with spatial coherence in generated images.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: rescaled_cfg_with_rescaled_uncond
Description: Current rescaled CFG normalizes the guided output to match the conditional output's std. But the unconditional prediction itself may have different magnitude characteristics. Instead of rescaling the combined output, rescale the unconditional prediction to match the conditional prediction's magnitude BEFORE combining, then apply standard CFG. Formula: uncond_rescaled = uncond * (cond.std() / uncond.std()), then guided = uncond_rescaled + w * (cond - uncond_rescaled).
Confidence: 5
Why: The conditional and unconditional models may learn different output magnitude distributions. Mismatched magnitudes create artifacts when combining. Pre-rescaling the unconditional output ensures both start from the same scale, potentially allowing the guidance direction to be cleaner.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: triple_t_loss
Description: Extend dual-t loss to triple-t: train with THREE different timesteps per image instead of two. Each forward pass samples t1, t2, t3 independently and the loss averages all three. This gives 50% more gradient information per training step at the cost of 50% more forward passes (slower steps, but richer learning signal per step). Whether the extra gradient information compensates for fewer total steps is the question.
Confidence: 5
Why: Dual-t was a clear win over single-t. Triple-t continues this trend — more timestep diversity per gradient step means the model sees a wider range of denoising conditions per update. The trade-off is fewer total steps in 1 hour. With the current 760ms per step (dual-t), triple-t would be ~1140ms per step. That's ~3160 steps vs ~4740 steps. The question is whether 3160 steps with 3x timestep diversity beats 4740 steps with 2x diversity.
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: learned_cfg_scale_per_t
Description: Instead of a fixed CFG scale across all ODE timesteps, learn the optimal scale per timestep. Add a small MLP that maps the current ODE timestep t to a guidance scale: cfg_scale(t) = MLP(t). Train this MLP with a simple objective: minimize FID-proxy (or just use a fixed annealing profile learned from the ablations). Implementation: at inference, the model's forward() computes dynamic_cfg = base_cfg * sigmoid(mlp(t)), giving per-step adaptive guidance.
Confidence: 4
Why: Different ODE steps benefit from different guidance strengths. Early steps (structure) may need strong guidance, mid steps (detail) need moderate, late steps (refinement) need less. We showed linear/cosine dynamic CFG hurts, but those were hand-designed schedules. A learned schedule could find the right shape. Risk: the MLP adds parameters and complexity, may not have enough training signal (only trained during the 1-hour run).
Time of idea generation: 2026-03-23 18:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: hebbian_z_update
Description: Replace the standard additive input injection in l_level with a Hebbian-inspired update rule. Instead of z = z + injection, use z = z + alpha * (injection * z.mean(dim=-1, keepdim=True)) where the update is proportional to the correlation between the current state and the injection. This is inspired by Oja's rule from computational neuroscience, where synaptic weights update proportionally to pre-post correlation. The recursive shared-weight block already looks like a recurrent neural circuit — making the update rule Hebbian could improve how information flows through the recursion.
Confidence: 5
Why: Bio-mimetic learning rules like Oja's/Hebb's rule have shown surprising effectiveness in simple neural circuits. The TRM recursive structure IS a neural circuit with shared weights. A Hebbian gating mechanism (only update when injection correlates with current state) could help the recursion converge faster and more stably. The risk is that it changes the gradient flow significantly and may need LR adjustment.
Time of idea generation: 2026-03-24 06:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: predictive_coding_loss
Description: Inspired by Karl Friston's predictive coding theory and Rao & Ballard (1999): instead of computing loss only on the final output, add auxiliary "prediction error" losses at each recursion level. At each L-cycle step, the model predicts what the next z_L state should be, and the error between prediction and actual next state becomes an auxiliary loss. This forces each recursive step to make a meaningful prediction, not just pass information through. Implementation: after each l_level call, compute prediction_error = MSE(z_L_predicted, z_L_actual.detach()) and add 0.01 * prediction_error to the total loss.
Confidence: 5
Why: Predictive coding is how the brain is hypothesized to process hierarchical information — each level predicts the next and only passes prediction errors upward. Our recursive model has a natural hierarchy (z_L updates → z_H consolidation). Adding prediction error objectives at each level could improve gradient signal through the deep recursion (6 L-cycles in first H). Currently, gradients must flow through all 6 cycles from the final loss — auxiliary losses at each level provide local learning signals, similar to deep supervision in U-Nets.
Time of idea generation: 2026-03-24 06:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: attention_to_recursion_history
Description: Instead of only attending to the current z_H + input_emb in each L-cycle, let the model attend to a compressed history of previous recursion states. Maintain a running "memory" of past z_L states (e.g., exponential moving average) and concatenate it as extra key-value context in attention. This gives the model temporal context within the recursion — it can see how the representation has evolved. Inspired by Alex Graves' work on neural Turing machines and external memory.
Confidence: 5
Why: Currently each L-cycle step only sees the current z_L and z_H+input. It has no memory of how z_L evolved through previous cycles. A simple EMA memory (z_mem = 0.9*z_mem + 0.1*z_L after each cycle) concatenated as extra KV tokens in attention would give the model access to the recursion trajectory. This is a lightweight form of "thinking about thinking" — the model can observe its own refinement process. Risk: adds complexity to the attention and may slow down per-step time.
Time of idea generation: 2026-03-24 06:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: stochastic_recursion_exit
Description: Instead of always running exactly 6 L-cycles in the first H-cycle, implement a learned "exit gate" that decides when to stop iterating. After each L-cycle, a small MLP (z_L → scalar sigmoid) predicts a halting probability. Training uses the adaptive computation time (ACT) framework from Alex Graves (2016). This lets the model dynamically allocate compute: easy images exit early (saving compute), hard images use all 6 cycles. The "ponder cost" penalty encourages efficiency. This is directly inspired by Graves' original ACT paper and Schmidhuber's work on self-delimiting programs.
Confidence: 4
Why: Different images and timesteps need different amounts of recursive refinement. Clean images near t=1 may only need 2-3 L-cycles, while noisy images near t=0 benefit from all 6. Currently we waste compute on easy cases. ACT gives the model agency over its own computation depth. Risk: ACT is notoriously tricky to train — the halting probability can collapse to always-halt or never-halt. The ponder cost coefficient needs careful tuning.
Time of idea generation: 2026-03-24 06:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: contrastive_cfg_training
Description: Instead of just dropping class labels for CFG training (replacing with zeros), use a contrastive approach: when dropping labels for a sample, replace with a RANDOM WRONG class label instead. This teaches the unconditional pathway to actively distinguish "no class" from "wrong class" — making the CFG direction (cond - uncond) more semantically meaningful. During inference, the "unconditional" path uses zero class embedding as before.
Confidence: 6
Why: Current CFG training just zeros out the class embedding. The unconditional model learns to denoise without class information. But contrastive CFG (using wrong class labels as negative examples) teaches the model what NOT to generate, making the guidance direction cond-uncond more aligned with class-specific features. This is inspired by contrastive learning literature (SimCLR, CLIP) where negative examples improve representation quality. The implementation is trivial: one line change in the label dropout code.
Time of idea generation: 2026-03-24 06:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------

-----------------------------------------------------
idea_id: multi_step_consistency_loss  
Description: Inspired by consistency models (Song et al., 2023): add an auxiliary loss that enforces the model's predictions at different ODE timesteps to be consistent. If the model predicts velocity v(x_t, t), then x_0_hat = x_t + (1-t)*v should be consistent regardless of t. Add loss: MSE(x0_hat_from_t1, x0_hat_from_t2.detach()) for the two timesteps already used in dual-t training. This is essentially free since we already compute two forward passes at different t.
Confidence: 6
Why: The dual-t loss already gives us two predictions at different timesteps for the same clean image. These predictions should reconstruct the same x0. Enforcing this consistency is a strong self-supervised signal that doesn't require any additional forward passes. The .detach() on one side makes it a teacher-student style loss. Consistency models have shown impressive results in distillation — applying the same principle during training could improve prediction quality.
Time of idea generation: 2026-03-24 06:00
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------
