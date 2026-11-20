# IE 7615 — Generative Project, Milestone 2
## Model Integration & Baseline Results — Summary of Observations
**Group 10**

### Setup
Base model: Stable Diffusion v1.5 (frozen, inference only). Text encoder CLIP
ViT-L/14 is integrated with the UNet/VAE through a hand-written denoising loop
(`src/pipeline.py`) rather than the off-the-shelf `StableDiffusionPipeline`; the
CLIP hidden size (768) matches the UNet cross-attention dimension. The custom
loop reproduces the reference pipeline's output to a mean pixel difference of
0.31/255, confirming correct integration. All runs: 512×512, fixed seed 42,
fp16 on a Tesla T4. Five prompts span a simple object, a multi-object scene, a
spatial-relation prompt, a stylistic (watercolor) prompt, and an abstract prompt.

### Baseline conditional generation
The five prompts generated cleanly through our loop at 30 steps, guidance 7.5
(PNDM). Output is coherent and prompt-aligned across all five, establishing the
reference set the sweeps vary from.

### Classifier-free guidance sweep (6 values: 1.0–15.0)
Guidance scale trades diversity for prompt fidelity. At cfg ≈ 1.0 outputs are
washed-out and only loosely follow the prompt (little conditioning signal). As
cfg rises, adherence and contrast increase; by cfg 10–15 images oversaturate and
show harder edges/artifacts. The default 7.5 is the best fidelity/naturalness
balance, matching common practice for SD 1.5.

### Noise-schedule / sampler sweep (PNDM, DDIM, DPM-Solver)
Sampler varied with everything else pinned (same prompt, seed, 30 steps,
cfg 7.5), so only the solver trajectory differs. Findings:
- **PNDM ≈ DDIM** on most prompts (simple object, spatial, stylistic, abstract):
  near-identical results, as expected from two deterministic methods on the same
  noise schedule.
- **DPM-Solver diverges most**, even from identical noise — a visibly different
  but equally valid sample (e.g. recomposed spatial scene, larger apple).
- **Sampler can change content, not just quality:** DDIM rendered the multi-object
  prompt with a different cat colour. Sampler choice is therefore not a purely
  cosmetic knob at fixed seed.
- **No sampler is clearly higher-fidelity at 30 steps** — all are sharp and
  artifact-free; differences are compositional/chromatic.
- **Timing:** DDIM and DPM-Solver ~4.5 s/image, PNDM ~4.7 s (first call ~6.6 s,
  warm-up). DPM-Solver is the natural candidate for a low-step (20) follow-up.

### Takeaways for M3
Guidance 7.5 + DPM-Solver (or PNDM) is a solid default. Next: a step-count sweep
(20/30/50) and quantitative metrics (FID, IS, CLIPScore) to replace visual
judgment with numbers, plus the SD 1.5 vs 2.1 backbone comparison.

### Reproducibility
Seed 42 throughout. Environment: torch 2.14.1+cu130, diffusers 0.31.0,
transformers 4.46.3, Python 3.13. Per-image config and library versions are
recorded in each `outputs/milestone2_conditioning/*/run_log.json`.
