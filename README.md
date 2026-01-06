# Facial Emotion Recognition

A simple workflow for training and evaluating a facial emotion classifier. The project uses a ResNet‑34 backbone with a lightweight spatial attention module, trains first on AffectNet (balanced subset), then fine‑tunes on FER2013, and provides rich visualizations and single‑image inference.

---

## What this project does
- Classifies faces into seven emotions: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Two‑phase training: pretrain on AffectNet (8 classes), then adapt to FER2013 (7 classes)
- Strong augmentations and class‑balanced loss to handle data imbalance
- Clear evaluation: confusion matrix, per‑class accuracy, precision/recall/F1, misclassification heatmaps, and statistical significance checks
- Single‑image prediction helper for quick demos

## Model and training highlights
- Backbone: `torchvision.models.resnet34(pretrained=True)`
- Attention: simple spatial attention (conv over avg/max maps)
- Class balancing: `compute_class_weight` + `CrossEntropyLoss(weight=...)` with label smoothing
- Optimizers & schedulers: Adam + `ReduceLROnPlateau`
- Mixed precision: optional AMP for faster training on GPU
- Augmentations: flips, rotation, affine translate, color jitter, random erasing

## Data setup
The notebook expects these folder structures:
- AffectNet (balanced subset): `AFFECTNET_DIR/train` and `AFFECTNET_DIR/val`
- FER2013: `FER_TRAIN_DIR` and `FER_TEST_DIR`

Dataset sources:
- Balanced AffectNet (subset): https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet
- FER2013: https://www.kaggle.com/datasets/msambare/fer2013

By default the notebook uses Kaggle paths:
- `AFFECTNET_DIR = '/kaggle/input/balanced-affectnet'`
- `FER_TRAIN_DIR = '/kaggle/input/fer2013/train'`
- `FER_TEST_DIR = '/kaggle/input/fer2013/test'`

If you run locally, update those variables near the top of the notebook to point to your local directories.

## Quick start
### Option A: Run on Kaggle
1. Open the notebook in a Kaggle kernel.
2. Attach the AffectNet (balanced) and FER2013 datasets via “Add Data”.
   - Balanced AffectNet: https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet
   - FER2013: https://www.kaggle.com/datasets/msambare/fer2013
3. Run all cells.

### Option B: Run locally (macOS/Linux/Windows)
1. Create a Python environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision scikit-learn matplotlib seaborn pillow scipy gdown
```

2. (Optional) Download datasets with the Kaggle CLI.

First, install and configure the Kaggle CLI:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place your Kaggle API token at ~/.kaggle/kaggle.json
# You can get it from Kaggle: Account > API > Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

Then download and unzip the datasets:

```bash
# Balanced AffectNet (subset)
kaggle datasets download -d dollyprajapati182/balanced-affectnet -p data/affectnet --unzip

# FER2013
kaggle datasets download -d msambare/fer2013 -p data/fer2013 --unzip
```

3. Place your datasets (if not using Kaggle CLI) and update `AFFECTNET_DIR`, `FER_TRAIN_DIR`, `FER_TEST_DIR` in the notebook.
3. Open and run [Facial Emotion Recognition.ipynb](Facial Emotion Recognition.ipynb).

## Training flow
1. Phase 1 (AffectNet, 8 classes)
   - Build `ResNet34(num_classes=8)` with spatial attention
   - Train ~15 epochs with class weights and AMP (if GPU available)
2. Phase 2 (FER2013, 7 classes)
   - Replace the final head with a 7‑class classifier
   - Fine‑tune ~12 epochs with smaller LR for backbone, larger LR for the head

Key hyperparameters live near the top of the notebook: `BATCH_SIZE`, `IMAGE_SIZE`, class lists, and data transforms.

## Evaluation and visualization
The notebook produces:
- Accuracy/Loss curves
- `classification_report` (precision/recall/F1 per class)
- Confusion matrix and normalized misclassification heatmap
- Per‑class accuracy bar chart
- Statistical analysis (chi‑square test) of confusion patterns
- Gallery of misclassified samples with predicted probabilities

## Single‑image prediction
Use the helper to classify a single image:

```python
predict_single_image(
    image_path,
    model,
    transform=fer_transforms['val'],
    class_names=FER_CLASSES,
)
```

There’s also a small `gdown` example that downloads an image and runs prediction.

## Example results (from the provided runs)
- Average test accuracy on FER2013 is around ~67%
- Strong classes: Happy (~85%), Surprise (~80%)
- Tough classes: Fear (~43%)
- Common confusions: Fear→Sad (~22%), Disgust→Anger (~19%), Sad↔Neutral (~17%)

Results will vary with hardware, random seed, and exact dataset splits.

## Tips to improve
- Add channel attention or SE blocks alongside spatial attention
- Use stronger backbones (e.g., ResNet50, EfficientNet) or vision transformers
- Try focal loss for hard examples
- Balance classes via sampling or mixup/augment policies
- Add face alignment or cropping with a detector before classification

## Requirements
- Python 3.9+
- PyTorch and Torchvision
- NumPy, SciPy, scikit‑learn
- Matplotlib, Seaborn, Pillow
- gdown (optional, for quick image downloads)

## Dataset notes
- AffectNet and FER2013 are third‑party datasets. Please follow their licenses and terms of use.
- This repo does not redistribute dataset files.

## Contributors
- [B M Rauf](https://github.com/mebmrauf)
- [Azmari Sultana](https://github.com/azmarisultana)
- [Anupam Sen Sagor](https://github.com/sagorsenanupam)