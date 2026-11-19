# Text-to-Image Generation with Latent Diffusion

## Overview

A text-to-image pipeline built on a **pretrained** latent diffusion model conditioned on
CLIP text embeddings. We treat generation as a controlled inference problem and study how
sampling and conditioning choices affect image fidelity and prompt alignment. Nothing is
trained from scratch.

## Model & Dataset

- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
  (fallback: `sd-legacy/stable-diffusion-v1-5`)
- **Text encoder:** CLIP ViT-L/14 (SD 1.5); OpenCLIP ViT-H/14 (SD 2.1, for comparison)
- **Dataset:** 2,000 image–caption pairs curated from COCO Captions 2017 (`val2017`):
  filtered for size, aspect ratio, colour mode, duplicates and CLIP's 77-token limit,
  one caption per image, images resized and center-cropped to 512 × 512

## Experiments

- Classifier-free guidance scale sweep (fidelity vs. diversity)
- Noise schedule and sampler/step comparison: DDIM, PNDM, DPM-Solver
- Backbone comparison: SD 1.5 vs. SD 2.1 (text-encoder effect)
- Metrics: FID, Inception Score, CLIPScore + qualitative gallery

## Folder Structure

```
generative/
├── README.md
├── requirements.txt               # pinned dependencies (torch installed separately)
├── data/                          # gitignored, except the subset's manifest and report
│   └── coco_subset/
│       ├── manifest.csv           # the 2,000 chosen image IDs + captions (committed)
│       └── cleaning_report.json   # filter rules and drop counts (committed)
├── notebooks/                     # one notebook per milestone, run top to bottom
│   ├── 01_baseline_inference.ipynb
│   └── 02_conditional_generation.ipynb
├── src/                           # reusable code imported by the notebooks
│   ├── __init__.py
│   ├── utils.py                   # device, paths, seeding, run logs, grids, gallery
│   ├── dataset_prep.py            # COCO download, filtering, subset building
│   ├── pipeline.py                # text encoder, UNet/VAE loading, denoising loop
│   ├── schedulers.py              # noise schedules / samplers by name
│   └── evaluate.py                # placeholder
├── outputs/                       # committed results, one folder per milestone
│   ├── milestone1_baseline/
│   └── milestone2_conditioning/
└── reports/                       # proposal, report, presentation figures
```

Notebooks are thin drivers: the logic lives in `src/`, so the same code runs from any
notebook and on any machine.

## Notebooks

| Notebook | Milestone | What it does |
|---|---|---|
| [`01_baseline_inference.ipynb`](https://colab.research.google.com/github/Varun-Ponugoti/DLAI/blob/main/generative/notebooks/01_baseline_inference.ipynb) | M1 | Environment check; downloads COCO `val2017` and builds the 2,000-pair subset; runs the CLIP text → embedding pipeline on the captions; 5 baseline prompt → image runs with `StableDiffusionPipeline` |
| [`02_conditional_generation.ipynb`](https://colab.research.google.com/github/Varun-Ponugoti/DLAI/blob/main/generative/notebooks/02_conditional_generation.ipynb) | M2 | Hand-written denoising loop (CLIP + UNet + VAE + scheduler); baseline conditional generation; guidance scale and noise schedule sweeps |

Run the M1 notebook first: it builds the dataset subset that later work relies on.

## Setup

The notebooks run on **Google Colab** or **locally**. A CUDA GPU is recommended (tested on
a Colab T4); Apple Silicon (MPS) and CPU work but are much slower. Model weights (~4 GB)
download automatically on first run from public mirrors, so no Hugging Face token is needed.

### Google Colab

1. Open a notebook with the links in the table above.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all. The first cell clones the repo, installs `generative/requirements.txt`, and
   makes `src/` importable. Colab's preinstalled torch is used as is.

### Local

```bash
git clone https://github.com/Varun-Ponugoti/DLAI.git
cd DLAI/generative
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1. Install torch for your hardware (CUDA / MPS / CPU):
#    pick the command from https://pytorch.org/get-started/locally/
# 2. Install everything else:
pip install -r requirements.txt

jupyter lab
```

`torch` is deliberately not pinned in `requirements.txt`: the right build depends on your
GPU and CUDA version, and Colab ships its own.

## Data

The COCO images are **not** in the repo. The M1 notebook downloads COCO `val2017`
(~1 GB) into `data/coco/` and builds the subset in `data/coco_subset/`. The build is
deterministic (seed 42), so anyone running it gets the same 2,000 pairs; the committed
`manifest.csv` records exactly which ones they are.

## Workflow

- **Results:** each notebook writes to `outputs/<milestone>/`. Commit those folders; they
  are part of the deliverable.
- **On Colab:** files are lost when the session ends. Each notebook's last cell zips its
  results with repo-relative paths: unzip at `generative/` in your local clone and commit.
  Alternatively, push from Colab with a GitHub personal access token.
- **Pull before you run** so you don't overwrite a teammate's outputs.

## Reproducibility

- All runs use a fixed seed (42).
- Each `outputs/<milestone>/run_log.json` records the model ID, scheduler, step count,
  guidance scale, device, timing, and the **actual** library versions used (including
  torch) for every generated image.
- `data/coco_subset/cleaning_report.json` records the filter rules and how many images and
  captions each rule removed.

## Milestones

- [x] **M1 — Setup & Dataset:** COCO subset curation (2,000 pairs), environment setup,
  text → embedding pipeline, 5 baseline runs
- [ ] **M2 — Integration & Baseline:** conditional generation, guidance/noise tuning *(in progress)*
- [ ] **M3 — Evaluation & Analysis:** FID/IS, prompt-alignment, sensitivity study
- [ ] **M4 — Final Report & Presentation:** 6–8 page report, gallery, demo
