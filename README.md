# IE 7615 Deep Learning for AI – Group 10 Project

| Folder | Milestone | Task |
|---|---|---|
| [`classification/`](classification/) | Milestone 1 | Image classification on a 73-class object dataset |
| [`objectdetection/`](objectdetection/) | Milestone 2 | Object detection with YOLOv8 |

**Original dataset:** [Google Drive link](https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS). Copy the dataset folder into your own Google Drive (**My Drive**) before running anything.

All notebooks run on Google Colab with a GPU runtime (**Runtime → Change runtime type → GPU**).

---

## Milestone 1 – Classification

Trains and compares BaselineCNN, MobileNetV2, ResNet50 and EfficientNetB0; EfficientNetB0 was selected as the final model. See the [Milestone 1 report](classification/IE7615_Group-10_Milestone-1_Report.pdf) for details.

### Run on Google Colab

1. Open [`classification/DLAI_test1.ipynb`](https://colab.research.google.com/github/Varun-Ponugoti/DLAI/blob/main/classification/DLAI_test1.ipynb) in Colab.
2. Select a GPU: **Runtime → Change runtime type → GPU**.
3. In the setup cell marked `# <-- change this`, set:
```python
   DATA_DIR = '/content/drive/MyDrive/<your dataset folder>'
```
4. **Runtime → Run all**, and allow Google Drive access when prompted.

Trained models are saved to `/content/drive/MyDrive/<ModelName>.keras`.

---

## Milestone 2 – Object Detection

Bounding-box labels are generated automatically with Grounding DINO, then a YOLOv8s detector is trained on grid images built from the dataset.

### Before you start

1. Put the original dataset in your **My Drive** as a folder named `IE7615 Deep Learning for AI`.
   If you use a different name, update `DATA_DIR` in **two** cells of `prepare_dataset.ipynb` (the first setup cell and the first cell under *Splitting the dataset*).
2. Upload [`objectdetection/class_prompts.json`](objectdetection/class_prompts.json) to `My Drive/Objectdetection/class_prompts.json`.

> **⚠️ The dataset is edited while the code runs.** `prepare_dataset.ipynb` works directly on the dataset folder in your Drive: images are resized, irrelevant images are excluded, and duplicate images are **permanently deleted**. If you want to keep the original dataset, make another copy of it before running the notebook.

### Run on Google Colab (in this order)

Both notebooks are saved with their outputs, so the results can be viewed without re-running them. Preparing the data and training take a long time.

| Step | Notebook | What it does |
|---|---|---|
| 1 | [`objectdetection/prepare_dataset.ipynb`](https://colab.research.google.com/github/Varun-Ponugoti/DLAI/blob/main/objectdetection/prepare_dataset.ipynb) | Cleans the dataset, splits it 70/15/15 into train/val/test, and auto-labels bounding boxes with Grounding DINO. Run all cells. |
| 2 | [`objectdetection/train_yolo_model.ipynb`](https://colab.research.google.com/github/Varun-Ponugoti/DLAI/blob/main/objectdetection/train_yolo_model.ipynb) | Builds the grid dataset and `data.yaml`, then trains YOLOv8s (up to 100 epochs, image size 1024). |
| 3 | Same notebook, section **Evaluate on test set** | Evaluates the best model on the test set and prints mAP50, mAP50-95, precision and recall. |

In `train_yolo_model.ipynb`, run the cells in order but **skip the *Resume training* cell**. Only run it if training was interrupted.

Outputs are saved to `My Drive/Objectdetection/`:
- `data/`: split images, labels, and annotation previews
- `runs/yolov8s_grid/`: trained weights (`best.pt`, `last.pt`) and training plots
- `runs/yolov8s_grid_test/`: test set results

**Skip steps 1–2:** the prepared dataset and trained model are available [here](https://drive.google.com/drive/folders/1NXn7oX20Aw8ERXLlPt4_IGABaGaJsOWN?usp=sharing).