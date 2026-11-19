# IE 7615 Deep Learning for AI — Group 10

Course projects for IE 7615 Deep Learning for AI. The repository holds two independent
projects, each in its own folder with its own README, requirements and run instructions.

## Team

- Varun Ponugoti
- Kush Pulindra Patel
- Sathya Sai Rajan
- Vignesh R. Nambiar

## Projects

| Project | Folder | Topic | Stack |
|---|---|---|---|
| 1 — Discriminative | [`discriminative/`](discriminative/) | Object classification and detection on a 73-class object dataset | TensorFlow / Keras, YOLOv8 |
| 2 — Generative | [`generative/`](generative/) | Text-to-image generation with latent diffusion | PyTorch, Diffusers, CLIP |

### Project 1 — Discriminative: object recognition

Recognizing objects from a 73-class dataset, in three stages:

- **Classification:** compares a baseline CNN with MobileNetV2, ResNet50 and EfficientNetB0
  (EfficientNetB0 selected as the final model).
- **Object detection:** bounding boxes auto-labelled with Grounding DINO, then a YOLOv8s
  detector trained on the labelled data.
- **Live demo:** runs the trained detector on new images.

See [`discriminative/README.md`](discriminative/README.md) for setup and run order.

### Project 2 — Generative: text-to-image diffusion

A text-to-image pipeline built on pretrained Stable Diffusion 1.5, conditioned on CLIP text
embeddings. Nothing is trained from scratch; the project studies how sampling and
conditioning choices (guidance scale, noise schedule, sampler, text encoder) affect image
quality and prompt alignment, using a curated subset of COCO Captions 2017.

See [`generative/README.md`](generative/README.md) for setup and run order.

## Repository Structure

```
DLAI/
├── README.md              # this file
├── .gitignore             # shared ignores (checkpoints, caches, OS files)
├── discriminative/        # Project 1: own README, requirements, notebooks
└── generative/            # Project 2: own README, requirements, notebooks, src/
```

## Running the Projects

Both projects run on **Google Colab** with a GPU runtime
(Runtime → Change runtime type → GPU); the generative project also runs locally.

The two projects use different frameworks and have **separate requirements files**.
Install only the one for the project you're working on; there is no shared environment.

```bash
git clone https://github.com/Varun-Ponugoti/DLAI.git
cd DLAI/<project folder>
pip install -r requirements.txt
```

Each project's README covers data setup, notebook order and where outputs are saved.