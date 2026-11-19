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
│   ├── schedulers.py              # noise schedules / samplers by name (pending, M2)
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

Run them in order. See [How to Run](#how-to-run) for what each step does and produces.

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

## How to Run

Run the notebooks **in order, top to bottom** (Runtime → Run all on Colab, or Run All
locally). Every notebook starts with the same two steps:

- **Step 0 — Setup:** on Colab, clones the repo and installs requirements; locally, finds
  the project folder. Always run this first.
- **Step 1 — Environment:** detects the GPU and loads the shared config.

### 1. `01_baseline_inference.ipynb` — builds the dataset (run first)

| Step | What it does | Produces |
|---|---|---|
| 2 | Downloads COCO `val2017` images and captions (~1 GB) | `data/coco/` |
| 3 | Filters the images and captions, samples 2,000 pairs, resizes to 512 × 512 | `data/coco_subset/images/`, `manifest.csv`, `cleaning_report.json` |
| 4 | Runs the CLIP text → embedding pipeline on all 2,000 captions | `outputs/milestone1_baseline/text_embedding_report.json` |
| 5 | Generates the 5 baseline images with `StableDiffusionPipeline` | `outputs/milestone1_baseline/images/`, `grid.png`, `run_log.json`, `GALLERY.md` |
| 6 | *(Colab only)* Zips the results for download | `milestone1_results.zip` |

**Creating the dataset:** Steps 2–3 are all it takes. Nothing has to be downloaded by
hand: the first run fetches COCO, and the subset is built from it automatically. The
build is deterministic (seed 42), so it always produces the same 2,000 pairs listed in
the committed `manifest.csv`.

**Re-running:** the COCO download is skipped if the files are already on disk, and the
subset is rebuilt identically. Locally this happens once; on Colab, files are deleted
when the session ends, so each new session downloads COCO again.

**First-run downloads:** besides COCO, Step 4 downloads the CLIP text encoder and Step 5
the full Stable Diffusion pipeline (several GB, cached afterwards).

### 2. `02_conditional_generation.ipynb` — integration and tuning

| Step | What it does | Produces |
|---|---|---|
| 2 | Loads CLIP, UNet, VAE and scheduler separately; checks they connect | — |
| 3 | Runs our own denoising loop and compares it with M1's pipeline image | — |
| 4 | Baseline conditional generation: the 5 prompts through our loop | `outputs/milestone2_conditioning/baseline/`, `run_log.json`, `GALLERY.md` |

Further steps (guidance sweep, noise schedule sweep, recording) will be added as M2
progresses.

M2 does **not** need the COCO subset, so it runs without re-running M1. Step 3 only
needs M1's baseline image (`outputs/milestone1_baseline/images/p1_simple_object.png`),
which is committed to the repo.

## Data

The COCO images are **not** in the repo (they're gitignored); only `manifest.csv` and
`cleaning_report.json` are committed. To recreate the images, run Steps 2–3 of the M1
notebook (see [How to Run](#how-to-run)).

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

## Progress

### M1 — Setup & Dataset complete

- [x] **Dataset chosen:** COCO Captions 2017 (`val2017`, 5,000 images, ~25k captions)
- [x] **Cleaned & preprocessed:** 4,064 images passed the filters; 2,000 pairs sampled
  (seed 42), resized and center-cropped to 512 × 512. `manifest.csv` and
  `cleaning_report.json` committed
- [x] **Environment set up:** PyTorch + Diffusers + CLIP, with a package check in the notebook
- [x] **Text → embedding pipeline:** CLIP tokenizer + encoder run on all 2,000 captions
  (77 × 768 per prompt; median caption uses 12 of 77 tokens); nearest-neighbour captions
  score 0.54–0.84 vs. 0.24 for random pairs. Saved to
  `outputs/milestone1_baseline/text_embedding_report.json`
- [x] **5 baseline runs** with `StableDiffusionPipeline` (PNDM, 30 steps, guidance 7.5)

### M2 — Integration & Baseline in progress

Done:
- [x] **Text encoder integrated with the diffusion model:** CLIP + UNet + VAE + scheduler
  loaded separately and connected in our own denoising loop (`src/pipeline.py`).
  CLIP hidden size (768) matches the UNet's cross-attention dimension
- [x] **Integration verified:** our loop reproduces M1's off-the-shelf pipeline image
  (mean pixel difference 0.31 / 255) at ~4.3 s per image on GPU, vs. ~6 s for the pipeline

Pending:
- [ ] **Baseline conditional generation:** 5 prompts through our loop, with run logging
- [ ] **Classifier-free guidance sweep** (≥ 3 values)
- [ ] **Noise schedule sweep** (≥ 3 values), including `src/schedulers.py`
- [ ] **Record generated images:** combined `run_log.json`, `GALLERY.md`, M2 write-up

### Later milestones

- [ ] **M3 — Evaluation & Analysis:** FID/IS, prompt-alignment, sensitivity study
- [ ] **M4 — Final Report & Presentation:** 6–8 page report, gallery, demo
