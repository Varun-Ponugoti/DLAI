# IE 7615 Deep Learning for AI – Group 10 Project

| Folder | Milestone | Task |
|---|---|---|
| [`classification/`](classification/) | Milestone 1 | Image classification on a 73-class object dataset |
| [`objectdetection/`](objectdetection/) | Milestone 2 | Object detection (in progress) |

**Dataset:** [Google Drive link](https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS). Copy the dataset folder into your own Google Drive before running anything.

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

In progress. Run instructions will be added to [`objectdetection/`](objectdetection/) once the notebooks are uploaded.