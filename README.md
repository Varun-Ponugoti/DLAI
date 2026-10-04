# Deep Learning Models


## Overview
This repository contains the code for all deep learning models we tested on the
[Dataset Name] dataset, along with the evaluation results. The best model was
**[Best Model]** with [metric = value] on the test set.

## Repository Structure
```
├── data/                  # Dataset or instructions to download it
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_model1_mlp.ipynb
│   ├── 03_model2_cnn.ipynb
│   ├── 04_model3_lstm.ipynb
│   └── 05_model4_transfer.ipynb
├── src/                   # Shared helper code (data loading, metrics)
├── results/               # Saved metrics, plots, confusion matrices
├── requirements.txt
└── README.md
```

## Models Tested
| Model | File | Test Accuracy | F1-score |
|-------|------|---------------|----------|
| [Model 1] | `notebooks/02_model1_mlp.ipynb` | [0.00] | [0.00] |
| [Model 2] | `notebooks/03_model2_cnn.ipynb` | [0.00] | [0.00] |
| [Model 3] | `notebooks/04_model3_lstm.ipynb` | [0.00] | [0.00] |
| [Model 4] | `notebooks/05_model4_transfer.ipynb` | [0.00] | [0.00] |

## How to Run

### Step 1: Clone the repository
```bash
git clone https://github.com/Varun-Ponugoti/DLAI.git
cd DLAI
```

### Step 2: Check and install the requirements
The code was tested with **Python 3.10**. Check your Python version first:
```bash
python --version
```

Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Then confirm that everything installed correctly and there are no version conflicts:
```bash
pip check
python -c "import tensorflow as tf, numpy, pandas, PIL, sklearn, matplotlib; print('TensorFlow', tf.__version__, '- all packages OK')"
```

### Step 3: Choose where to run the code
Training uses 224×224 images and fine-tunes large pretrained networks
(ResNet50, EfficientNetB0, MobileNetV2), so it needs a GPU to finish in a
reasonable time. Check the requirements below and pick one of the two options.
 
#### Compute requirements for running locally
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 6 GB VRAM (CUDA-capable) | NVIDIA GPU with 8 GB+ VRAM |
| RAM | 8 GB | 16 GB |
| Disk space | 10 GB free (packages + dataset + saved models) | 15 GB free |
| Operating system | Linux, or Windows with WSL2 | Linux |
 
> **Note:** TensorFlow 2.20 only supports GPUs on Linux and on Windows through
> WSL2. On native Windows or macOS it runs on the CPU only, which works but makes
> training very slow (many hours instead of minutes). If your machine doesn't meet
> the requirements above, use Option B.
 
#### Option A: Run locally
If your machine meets the requirements above, run the notebook locally. The
first cell downloads the dataset from Google Drive and extracts it into `data/`
automatically. If the dataset is already there, the download is skipped.
 
If the automatic download fails (for example, because Google Drive has hit its
daily download limit), download the zip manually from
**[Google Drive link to dataset]** and unzip it into `data/`. The notebook
will detect it and skip the download.
 
#### Option B: Run on Google Colab (no local GPU needed)
Google Colab provides a free GPU in the browser.
 
1. Open the notebook in Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[username]/[repository-name]/blob/main/notebooks/[notebook-name].ipynb)
2. Turn on the GPU: *Runtime → Change runtime type → T4 GPU*.
3. Install the requirements by running this in the first cell, then restart the
   runtime (*Runtime → Restart session*) so the pinned versions are loaded:
```python
   !pip install -r https://raw.githubusercontent.com/[username]/[repository-name]/main/requirements.txt
```
### Step 4: Match the dataset folder name (only if running locally)
The notebook expects the dataset at `data/IE7615_Deep_Learning_for_AI/`. If you
used the automatic download, this is already correct and you can skip this step.
If you downloaded the dataset manually and the folder has a different name (for
example, `dataset` or `IE7615_Deep_Learning_for_AI-20261004`), use one of these
two options:
 
**Option 1: Rename the folder.** Rename your dataset folder to
`IE7615_Deep_Learning_for_AI` so the structure looks like this:
```
data/
└── IE7615_Deep_Learning_for_AI/
    ├── class_1/
    ├── class_2/
    └── ...
```
On Linux, macOS or WSL:
```bash
mv data/your-folder-name data/IE7615_Deep_Learning_for_AI
```
On Windows (Command Prompt):
```bat
ren data\your-folder-name IE7615_Deep_Learning_for_AI
```
 
**Option 2: Change the path in the notebook.** Keep your folder name and update
`DATA_DIR` in the first cell of the notebook to match it:
```python
DATA_DIR = os.path.join(DATA_ROOT, 'your-folder-name')
```
If your dataset is stored outside the repository, you can also use the full path:
```python
DATA_DIR = '/path/to/your-folder-name'
```
 
Either way, `DATA_DIR` must point to the folder that **directly contains the
class subfolders**, not to a parent folder above it. If the notebook prints
`0 folders` when listing the classes, the path is one level too high or too low.
 

### Step 5: Run the notebooks
Run `01_preprocessing.ipynb`, then any model notebook. Each notebook trains,
evaluates, and saves its results to `results/`.

Pretrained ImageNet weights for the transfer learning models are downloaded
automatically the first time they are built, so an internet connection is needed.
A GPU is strongly recommended for training.

## Report
The full report (PDF) is available in this repository as `report.pdf`.
