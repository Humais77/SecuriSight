# SecuriSight ViT Compression Experiment

A small standalone experiment to fine-tune a pretrained Vision Transformer (ViT) for `REAL` vs `AI_GENERATED` image classification, then compare an original image with a compressed copy.

## What this does

- Fine-tunes `google/vit-base-patch16-224-in21k` using PyTorch and Hugging Face Transformers.
- Creates JPEG versions at quality 95, 90, 70, 40, and 20 while preserving the source authenticity label.
- Splits original source images into train/validation/test BEFORE making variants, so variants from one source never cross splits.
- Reports accuracy, precision, recall, F1-score, and confusion matrix.
- Compares a pair of images using file-size reduction, size ratio, PSNR, SSIM, and ViT predictions.

**This is an experimental classifier, not a forensic guarantee.** Compression alone cannot determine whether an image is real or fake.

## 1. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code
- 8 GB RAM can run a small experiment; 16 GB is more comfortable
- Internet for downloading the pretrained checkpoint the first time
- NVIDIA GPU is optional. AMD Radeon RX 590 does not normally provide CUDA support on native Windows PyTorch; use CPU locally or Google Colab/Kaggle GPU.

## 2. Create environment (PowerShell)

Open this folder in VS Code, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this in the same terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install PyTorch CPU build first (reliable on Windows):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If using Google Colab/Kaggle, install packages there and choose a GPU runtime. Follow the official PyTorch selector if using another supported GPU setup.

## 3. Prepare the dataset

Put images into these folders:

```text
data/
  raw/
    real/
      real_photo_001.jpg
      real_photo_002.jpg
    ai_generated/
      synthetic_001.jpg
      synthetic_002.jpg
```

Use authentic photographs for `real` and genuinely AI-generated images for `ai_generated`. Do not place the same image or near-duplicate in both classes. Start with 100-500 images per class for a smoke test; more diverse data is better for meaningful evaluation.

Dataset suggestions:
- CIFAKE dataset paper/access information: https://irep.ntu.ac.uk/id/eprint/50758/
- GenImage project: https://github.com/GenImage-Dataset/GenImage

Download data only after reviewing the dataset's terms and license. Organize files into the two folders above. The code does not automatically download datasets.

Prepare splits and JPEG variants:

```powershell
python src/prepare_dataset.py
```

The script uses 70% train, 15% validation, 15% test by source image, and writes variants to `data/processed/`. It recreates `data/processed/` every time, so do not store your only copy of data there.

## 4. Train ViT

Quick smoke test (uses at most 200 training images, 1 epoch):

```powershell
python src/train.py --epochs 1 --batch-size 2 --max-train-images 200
```

Small experiment:

```powershell
python src/train.py --epochs 3 --batch-size 2
```

If RAM/VRAM allows, try batch size 4. On CPU, training can take a long time. The best validation-loss model is saved to `outputs/vit-real-vs-ai/`, along with `metrics.json`.

The model is pretrained and fine-tuned, not trained from random initialization. The first run downloads the pretrained ViT weights.

## 5. Predict one image

```powershell
python src/predict.py "C:\path\to\your\image.jpg"
```

## 6. Compare original and compressed images

First make a compressed copy of the original (for example, using an image editor to save it as JPEG quality 40), then run:

```powershell
python src/compare_images.py "C:\images\original.jpg" "C:\images\compressed.jpg"
```

The report prints:
- original and compressed file sizes
- size reduction percentage
- original-to-compressed size ratio
- PSNR in dB
- SSIM
- each image's ViT prediction and confidence
- whether the predictions match

It also writes `outputs/comparison.csv`.

PSNR/SSIM are computed after resizing the compressed image to the original dimensions. This is useful for a rough comparison but means resizing also affects the pixel metrics. Prefer two versions derived from the same source image. If the two images are unrelated, similarity metrics cannot be interpreted as compression loss.

## 7. Understand the labels and metrics

- `REAL` means the classifier predicts the real class.
- `AI_GENERATED` means the classifier predicts the synthetic class.
- Confidence is the model's softmax probability, not a calibrated probability of truth.
- File-size reduction is not a direct measure of image quality loss.
- A real image remains real after compression; an AI-generated image remains AI-generated after compression.

## 8. Important experimental limitation

This starter script creates all JPEG variants for each image, then trains on those variants. Since every variant of a source stays in the same split, source leakage is avoided. However, training and evaluation both contain multiple compression qualities. For a stronger research experiment, create a second evaluation where the training set uses only original/high-quality images and the test set is evaluated separately at JPEG qualities 90, 70, 40, and 20. Compare that baseline against the compression-augmented model.

Do not claim production-level deepfake detection from a small dataset. Results depend heavily on source diversity, generator diversity, labels, and the train/test design.
