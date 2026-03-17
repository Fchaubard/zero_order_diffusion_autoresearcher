# autoresearch

This is an experiment to have an LLM do its own research on TRM-inspired recursive diffusion, using the TinyRecursiveModels architecture (shared-weight blocks, two-level latent hierarchy z_H/z_L, recursive refinement cycles) adapted for image generation with flow matching, trained with truncated BPTT or zero-order optimization.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. TRM-inspired recursive model architecture (shared L-level blocks, z_H/z_L latent hierarchy, H/L cycles, SwiGLU MLP), optimizer (must be zero order, no use of full gradients ever, can use truncated BPTT only for baseline), training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains ImageNet data. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation. Once the experiment is complete, do analysis on the run and determine what you would advise to do next. If you need to build any one off script to do analysis, you must always put it in ./garbage/ so it doesnt muck up my current directory. Then provide an idea for the next experiment based on the results of the previous experiments. If you do not have any more ideas and you are lost, just say so. 

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 1 hour** (wall clock training time, excluding startup/compilation). You launch it as: `uv run train.py` (truncated backprop through time baseline) or `uv run train.py --solver spsa` (zero-order SPSA) or otherwise if you want to implement your own solver. All hyperparameters are exposed as CLI flags — run `uv run train.py --help` for the full list.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: loss equation, per layer loss and global loss, sum of loss per timestep or loss calculated only at end, MSE vs. cross entropy vs. FID direct loss vs. something else, type of zero order, hyperparameters, training loop, batch size, model size, the TRM-style recursion pattern, input injection strategy, gradient truncation policy, etc. You can play with T during training (the number of iters to produce the image aka the number of refinement steps). you can play with loss function (use CE, or SSIM, or FID directly, or focal loss, or hinge loss, or some RL loss, or whatever else you can dream up. you can play with the solver, you can use a zero order solvers like SPSA or cross entropy method or random search or other Evolutionary Search procedures. I would love to make this bio-logically plausible but not at the expense of more complexity! You can lean into bio-memetic ideas all you want like Ojas rule or Sangers rule but make it simple. You can play w using the sum of the losses at each step until T (multi_step_loss) vs. use only the last loss at step T (last_step_loss). Use multi_step_loss but weight each loss to be more loss the longer in T we go (linearly or exponentially?). You can adjust T throughout training on a sinsoidal schedule or something so we are changing the number of refinement steps over time starting with small and then growing and shrinking? not sure what that will do but its interesting. really be creative here. you can ALWAYS go to literature and lookup ideas from the past in theoretical neuroscience or deep learning, especially the ideas of schmidhuber, hinton, or alex graves. BUT you must keep simplicity in mind. 

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, flow matching, and training constants (time budget, image resolution, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_fid` function in `prepare.py` is the ground truth metric. You also can not add more data. 
- You can not do boring dumb research. Simple stupid ideas like adding adam, playing w a lr warmup schedule, making it deeper or wider, play w the recursive architecture parameters (h_cycles, l_cycles, l_layers, n_embd), play with LR too much, adding dropout, etc. these are boring ideas and not research grade ideas. do NOT do boring research! I am not interested in hyperparam tuning. Dont do it! 
- 
**The goal is simple: get the lowest val_fid.** Since the time budget is fixed, you don't need to worry about training time — it's always 1 hour. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_fid gains, but it should not blow up dramatically. Zero order is much more memory efficient and you should keep that in mind. Its able to train 'in place' with no solver bloat. You should consider this as a plus and keep it if possible. Caching of the probe speeds things up which you can do if you must. The TRM-style shared-weight architecture is inherently parameter-efficient (few params, more compute via recursion), which pairs well with SPSA since SPSA cost scales with parameter count. Gradient truncation (only backprop through the last H_cycle) further reduces memory for TBPTT training.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_fid improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_fid improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
solver:           tbptt
val_fid:          125.4321
training_seconds: 3600.0
total_seconds:    3615.2
peak_vram_mb:     8042.1
mfu_percent:      12.50
total_images_M:   2.1
num_steps:        8192
num_params_M:     13.2
h_cycles:         2
l_cycles:         3
l_layers:         1
denoising_steps:  20
n_perts:          40
final_lr:         1.00e-04
final_epsilon:    1.00e-04
```

(Note: `denoising_steps` through `final_epsilon` only printed for SPSA solver)

Note that the script is configured to always stop after 1 hour wallclock, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_fid:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_fid	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_fid achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_fid	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_fid:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_fid improved (lower), you "advance" the branch, keeping the git commit
9. If val_fid is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take 1 hour total (+ a few seconds for startup and eval overhead). If a run exceeds 1 hour and 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — look at the data closely for inspiration, do mini tests if you need to really analyze something, read papers online, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period. If you need to build any one off script you must always put it in ./garbage/ so it doesnt muck up my current directory. 

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~1 hour then you can run approx 1/hour * # of GPUs available (which is 8 A100s on this node), for a total of about 64 over the duration of the average human sleep (8 hours). The user then wakes up to experimental results, all completed by you while they slept! You should NEVER STOP THOUGH! Keep making the solver, the architecture, and the diffusion better and better. 
