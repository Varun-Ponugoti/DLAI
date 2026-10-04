# Model Comparison for Object Image Classification

This project trains and compares four image classification models on a 73-class object image dataset. All models, from the baseline CNN to the transfer-learning models, are trained and evaluated in **one notebook**, so the full experimentation process can be verified and reproduced end to end.

## Dataset

**Download the dataset from Google Drive:** [LINK](https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS)

The dataset contains 73 object classes, each in its own folder (`images_OBJ001` … `images_OBJ073`), with roughly 100 JPEG images per class .

Folder structure:

```
<dataset folder>/
├── images_OBJ001/
│   ├── img1.jpg
│   └── ...
├── images_OBJ002/
└── ... images_OBJ073/
```

### Data cleaning 

- Non-JPEG and corrupted files are excluded (2 files).
- Exact duplicate images are found with MD5 hashing and removed (28 files).
- Final dataset: **7,315 images**, split with stratification into train / validation / test = 70% / 15% / 15% (5,120 / 1,097 / 1,098).


## Models tested

All models share the same input size (224×224), data augmentation, data splits, and callbacks.

| Model | Approach |
|-------|----------|
| **BaselineCNN** | Custom 4-block CNN (32→64→128→256 filters) trained from scratch, up to 40 epochs |
| **MobileNetV2** | ImageNet-pretrained; frozen base + new head (15 epochs), then fine-tuning of the last 30 layers at LR 1e-5 (15 epochs) |
| **ResNet50** | Same two-stage transfer-learning procedure |
| **EfficientNetB0** | Same two-stage transfer-learning procedure |

## Results (test set, 1,098 images)

| Model | Test accuracy | Macro F1 | Parameters |
|-------|---------------|----------|------------|
| **EfficientNetB0** (selected) | **0.9417** | **0.9413** | 4.1M |
| ResNet50 | 0.9381 | 0.9392 | 23.7M |
| MobileNetV2 | 0.9098 | 0.9102 | 2.4M |
| BaselineCNN | 0.7450 | 0.7465 | 0.4M |

EfficientNetB0 was selected as the final model: it achieved the highest accuracy and macro F1 with about one-sixth the parameters of ResNet50. The notebook also shows a validation-accuracy plot for all models, plus a per-class classification report and confusion matrix for the best model.

## How to run

### Option 1: Google Colab (recommended, this is how the project was run)

1. Download the dataset from the Google Drive link above and place the folder in your own Google Drive.
2. Open `DLAI_test1.ipynb` in [Google Colab](https://colab.research.google.com/).
3. Select a GPU runtime: **Runtime → Change runtime type → GPU**.
4. Update `DATA_DIR` in the setup cell (the one marked `# <-- change this`) to point to your dataset folder, e.g.
   ```python
   DATA_DIR = '/content/drive/MyDrive/<your dataset folder>'
   ```
5. Run all cells (**Runtime → Run all**). Colab will ask for permission to mount Google Drive.

Trained models are saved to `/content/drive/MyDrive/<ModelName>.keras`.

### Option 2: Running locally

1. Download and unzip the dataset from the Google Drive link above.
2. Install the dependencies (>=Python 3.10 recommended):
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook DLAI_test1.ipynb
   ```
4. **Change the dataset location in the code** to the folder where you saved the dataset. Edit `DATA_DIR` in the setup cell (marked `# <-- change this`):
   ```python
   DATA_DIR = 'path/to/your/dataset'
   ```
5. Make the following Colab-specific edits:
   - **Skip the first four cells** (Google Drive mounting and Drive folder listing); they only work in Colab.
   - Change the output paths that point to `/content/...` to a local folder:
     - `'/content/removed_duplicates.csv'` and `'/content/excluded_images_log.csv'` (data cleaning cells)
     - `f'/content/drive/MyDrive/{name}.keras'` (model training loop)
6. Run the remaining cells in order.

A GPU is strongly recommended. Training all four models on CPU will take a long time.

