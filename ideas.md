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
Description: Instead of computing loss only at the final output, compute the flow matching velocity loss at EACH H_cycle boundary (after z_H is updated by z_L), but ONLY on the z_H output (not z_L intermediates, which failed before in batch 1). Weight later cycles exponentially: w_i = 2^i / sum(2^j). This is "deep supervision for recursive networks" — each level of recursion gets direct gradient signal about its velocity prediction quality. Different from our failed exp1 which incorrectly used z_L intermediates.
Confidence: 7
Why: Our exp1 failed because we computed intermediates from z_L (the working state). z_H is the "answer" state and should be supervised. Each H_cycle produces a z_H that goes through the output head — supervising these gives the shared blocks gradient signal at multiple recursion depths. Literature on deep supervision (Lee et al. 2015) shows this consistently helps in deep/recursive networks. The key difference from our failed attempt is using z_H (output state) not z_L (working state).
Time of idea generation: 2026-03-18 10:30
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
idea_id: learned_injection_gate
Description: Replace the additive injection `x = x + input_injection` in l_level with a learned gating mechanism: `x = x + sigmoid(linear(x)) * input_injection`. This lets the model learn HOW MUCH of the injection to accept at each position. Some patches may need more refinement (complex regions) while others are already good (uniform regions). The gate is a single Linear(n_embd, n_embd) layer + sigmoid, adding ~590K params (small relative to 13.6M).
Confidence: 7
Why: Currently all patches receive identical injection strength. But image complexity varies spatially — a sky patch is simple while a face patch is complex. A learned gate would let the model adaptively control information flow. This is related to highway networks (Srivastava et al. 2015) and LSTM gating. Gate should be zero-initialized so it starts as standard additive (no regression from baseline).
Time of idea generation: 2026-03-18 10:30
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
idea_id: denoising_as_recursion
Description: Unify the diffusion ODE steps with the recursive cycles. During training, instead of predicting velocity at a random single timestep, run the model recursively where each L-cycle takes one denoising step. Start from noise x_0 (t=0), and at each L-cycle predict velocity at t=cycle/total_cycles, step forward, feed result back. Loss on final denoised image vs clean target. At eval, the standard ODE solver still works (model predicts velocity at any t). But training teaches the shared block to be a good single-step denoiser, which should improve the Euler ODE quality.
Confidence: 6
Why: The recursive architecture naturally maps to iterative denoising. Currently recursion and ODE are separate (recursion refines within a timestep, ODE integrates across timesteps). Unifying them means the model practices actual denoising during training, not just single-timestep velocity prediction. This is conceptually similar to consistency models but uses the existing recursive structure. Risk: the training signal may be too noisy since early cycles see poor predictions.
Time of idea generation: 2026-03-18 10:30
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
idea_id: token_mixing_mlp
Description: Replace self-attention in RecursiveDiTBlock with a simple token-mixing MLP (MLP-Mixer style). Instead of Q/K/V attention over patches, transpose to (B, C, N) and apply a linear layer across the N (spatial) dimension. This is O(N) instead of O(N^2), much faster, and the shared-weight recursion provides the "depth" that attention normally gives. With 256 patches and 768 channels, the token-mixing linear is 256x256 = 65K params per block (vs ~2.4M for attention). Much cheaper, so we get more training steps per hour.
Confidence: 6
Why: The recursive architecture already provides iterative refinement — each pass through the shared block is like another "layer" of processing. Attention's main benefit is long-range spatial mixing, which the recursion also provides (each cycle mixes information globally via the z_H/z_L interaction). MLP-Mixer has shown competitive performance to attention in vision tasks. With 8 recursive passes, even simple per-pass mixing should work. The speed gain means 2-4x more training steps in the same 1-hour budget.
Time of idea generation: 2026-03-18 10:30
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
idea_id: target_ema_teacher
Description: Use the EMA model as a teacher during training (not just for eval). At each step, compute the EMA model's velocity prediction (detached) and add a distillation loss: MSE(student_pred, teacher_pred). The EMA model is a smoothed version of the student and acts as a stable target, similar to BYOL/target networks in RL. This is different from our earlier self-distillation (which used deeper recursion from the same model). Weight: 0.05.
Confidence: 6
Why: EMA gave us -42 FID at eval time, meaning the EMA weights produce much better predictions. But currently we only USE the EMA at eval — we don't TRAIN toward it. By distilling from EMA during training, the student model gets a smoother, more stable learning signal. This is the same principle as Polyak-Ruppert averaging in optimization theory, but applied as a loss term. The risk is that it creates a circular dependency (teacher is a lagged copy of student), but this has been shown to work in BYOL, DINO, and other self-supervised methods.
Time of idea generation: 2026-03-18 10:30
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
idea_id: per_patch_loss_weighting
Description: Compute the velocity loss per-patch (not per-pixel averaged), then weight each patch's loss by its prediction error magnitude from the EMA model. Patches where the EMA model has high error are "hard" patches — weight them 2x. Patches where the EMA model is accurate are "easy" — weight them 0.5x. This is curriculum learning at the spatial level, focusing compute on the hardest image regions.
Confidence: 5
Why: FID is sensitive to failure modes — a few badly generated patches can tank the whole score. By focusing training on hard patches (faces, textures, fine details), we fix the worst failure modes first. The EMA model provides a stable difficulty estimate. Similar to focal loss but spatially localized. Risk: could overfit to hard patches and neglect easy ones.
Time of idea generation: 2026-03-18 10:30
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
idea_id: stochastic_recursion_depth
Description: During training, randomly sample the total number of recursive iterations from a distribution (e.g., uniform 4-12 instead of fixed 8). At eval, use the full 8. This forces the model to produce good outputs regardless of recursion depth — like stochastic depth for recursive networks. The model must learn to make each iteration count, since it doesn't know how many it'll get.
Confidence: 5
Why: The model currently always gets exactly 8 L-level passes. It may learn to "spread out" its computation across all 8, doing a little each step. With stochastic depth, it must be prepared to output a good prediction after any number of steps — encouraging each step to be maximally useful. This is related to Graves' ACT (Adaptive Computation Time) and regularization via randomized depth (Huang et al. 2016). Risk: may hurt early training when the model hasn't learned basic patterns yet.
Time of idea generation: 2026-03-18 10:30
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
idea_id: contrastive_velocity
Description: Add a contrastive loss on velocity predictions: velocities for images of the SAME class should be more similar than velocities for DIFFERENT classes (at the same timestep t). Use InfoNCE with temperature 0.1. This encourages class-conditional structure in the velocity field. Implementation: within each batch, compute pairwise cosine similarity of velocity predictions grouped by class label.
Confidence: 4
Why: The model has a class conditioning mechanism via label embeddings, but the velocity loss (Huber) doesn't explicitly encourage class-dependent structure. A contrastive term would push the model to produce systematically different velocities for different classes, improving class-conditional generation quality which directly impacts FID (which measures per-class quality). Risk: may be too expensive (O(B^2) pairwise computation) and the signal may be noisy since velocity also depends heavily on t and noise.
Time of idea generation: 2026-03-18 10:30
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
idea_id: residual_prediction
Description: Instead of predicting the full velocity v, predict a RESIDUAL on top of a simple analytical estimate. The analytical estimate is: v_simple = (x_clean - x_noisy) / (1 - t) ≈ (x_t - noise_estimate) / t. Since we have x_t and t, we can compute a naive velocity as v_naive = -x_t / (1-t) (pointing toward origin). The model then predicts delta_v such that v = v_naive + delta_v. This makes the prediction task easier — the model only needs to predict the correction, not the full velocity.
Confidence: 4
Why: Residual learning (He et al. 2016) makes optimization easier by letting the model learn corrections rather than full mappings. The velocity field is complex and varies greatly across timesteps. A residual formulation factors out the "obvious" component, letting the model focus on the hard part. Risk: the naive estimate might not be meaningful, and the residual could have worse conditioning than the original.
Time of idea generation: 2026-03-18 10:30
Status: Not Implemented
HPPs:
Time of run start and end:
Results vs. Baseline:
wandb link:
Analysis:
Conclusion:
Next Ideas to Try:
-----------------------------------------------------
