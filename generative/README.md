# From Words to Worlds: Text-to-Image Generation with Latent Diffusion

IE 7615 Deep Learning for AI — Generative Project (Group 10)

## Overview

A text-to-image pipeline built on a **pretrained** latent diffusion model
conditioned on CLIP text embeddings. We treat generation as a controlled
inference problem and study how sampling and conditioning choices affect
image fidelity and prompt alignment. Nothing is trained from scratch.

## Team

| Member | Role |
|---|---|
| Sathya Sai Rajan | Dataset curation, dedup, reproducibility logging |
| Kush Pulindra Patel | Pipeline & text-conditioning integration |
| Varun Ponugoti | Experimentation, repository owner |
| Vignesh R. Nambiar | Evaluation (FID/IS/CLIPScore), ethical analysis |

## Model & Dataset

- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
  (fallback: `sd-legacy/stable-diffusion-v1-5`)
- **Text encoder:** CLIP ViT-L/14 (SD 1.5); OpenCLIP ViT-H/14 (SD 2.1, for comparison)
- **Dataset:** curated COCO Captions 2017 subset (~2,000 image–caption pairs)

## Experiments

- Classifier-free guidance scale sweep (fidelity vs. diversity)
- Sampler/step comparison: DDIM, PNDM, DPM-Solver
- Backbone comparison: SD 1.5 vs. SD 2.1 (text-encoder effect)
- Metrics: FID, Inception Score, CLIPScore + qualitative gallery

## Repository Structure

```
text-to-image-diffusion/
├── README.md
├── requirements.txt               # pinned dependencies (torch installed separately)
├── .gitignore
├── data/                          # datasets (gitignored)
├── notebooks/                     # one notebook per milestone, run top to bottom
│   ├── 01_baseline_inference.ipynb
│   └── 02_conditional_generation.ipynb
├── src/                           # reusable code imported by the notebooks
│   ├── __init__.py
│   ├── utils.py                   # device, paths, seeding, run logs, gallery writer
│   ├── pipeline.py                # load model, encode prompts, generate
│   ├── schedulers.py              # build schedulers / noise schedules by name
│   ├── dataset_prep.py            # placeholder
│   └── evaluate.py                # placeholder
├── outputs/                       # committed results, one folder per milestone
│   ├── milestone1_baseline/
│   └── milestone2_conditioning/
│       ├── baseline/
│       ├── guidance_sweep/
│       ├── schedule_sweep/
│       ├── run_log.json
│       └── GALLERY.md
└── reports/                       # proposal, report, presentation figures
```

Notebooks are thin drivers: the logic lives in `src/`, so the same code
runs from any notebook and on any machine.

## Setup

The notebooks run on **Google Colab** or **locally**. A CUDA GPU is
recommended (tested on a Colab T4); Apple Silicon (MPS) and CPU work but
are much slower. Model weights (~4 GB) download automatically on first
run from public mirrors, so no Hugging Face token is needed.

### Google Colab

1. Open a notebook from GitHub in Colab
   (File → Open notebook → GitHub → `sathyaa23/text-to-image-diffusion`).
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all. The first cell clones the repo, installs `requirements.txt`,
   and makes `src/` importable. Colab's preinstalled torch is used as is.

### Local

```bash
git clone https://github.com/sathyaa23/text-to-image-diffusion.git
cd text-to-image-diffusion
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1. Install torch for your hardware (CUDA / MPS / CPU):
#    pick the command from https://pytorch.org/get-started/locally/
# 2. Install everything else:
pip install -r requirements.txt

jupyter lab
```

`torch` is deliberately not pinned in `requirements.txt`: the right build
depends on your GPU and CUDA version, and Colab ships its own.

## Workflow

- **Results:** each notebook writes to `outputs/<milestone>/`. Commit those
  folders; they are part of the deliverable.
- **On Colab:** files are lost when the session ends. Before closing, either
  download the outputs folder (each notebook's last cell zips it) and commit
  locally, or push from Colab with a GitHub personal access token.
- **Pull before you run** so you don't overwrite a teammate's outputs.

## Reproducibility

All runs use a fixed seed (42). Each `outputs/<milestone>/run_log.json`
records the model ID, scheduler, step count, guidance scale, device, and
the **actual** library versions used (including torch) for every
generated image.

## Milestones

- [x] **M1 — Setup & Dataset:** proposal, 5 baseline runs, repo init
- [ ] **M2 — Integration & Baseline:** conditional generation, guidance/noise tuning *(in progress)*
- [ ] **M3 — Evaluation & Analysis:** FID/IS, prompt-alignment, sensitivity study
- [ ] **M4 — Final Report & Presentation:** 6–8 page report, gallery, demo
