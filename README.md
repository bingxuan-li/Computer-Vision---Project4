# GeoGuessr StreetView — StreetCLIP

A **multi-view, multi-task vision model** for predicting **US state location** and **GPS coordinates** from StreetView images, built on top of **StreetCLIP (CLIP-based geolocation model)**.

---

## Overview

This project implements a **StreetCLIP-based multi-view model** that takes **four directional StreetView images** (north, east, south, west) and jointly predicts:

1. **State classification**

   * Full probability distribution over 50 US states
   * Top-1 and Top-5 accuracy used for evaluation
2. **GPS regression**

   * Latitude and longitude (continuous)

The model is trained with **multi-task learning** using a weighted loss:

* **70%** state classification loss
* **30%** GPS regression loss

---

## Model Architecture

### Backbone: StreetCLIP

* Uses the pretrained HuggingFace model
  **`geolocal/StreetCLIP`**
* CLIP image encoder pretrained for geolocation tasks
* Can be **frozen initially** and **unfrozen later** during training

### Multi-View Processing

* Each of the 4 views (N / E / S / W) is encoded **independently** using the same CLIP image encoder
* Image features are then **fused** using a lightweight fusion module

### Feature Fusion

Supported fusion strategies:

* Transformer-based fusion (default)
* Attention-based fusion
* Simple concatenation (configurable)

### Output Heads

* **State head**

  * Linear + softmax
  * Outputs probabilities over 50 states
* **GPS head**

  * MLP regression head
  * Outputs `(latitude, longitude)`

---

## Dataset Format

### Input Images

Each sample consists of **four images**, stored in a directory:

```
img_000000_north.jpg
img_000000_east.jpg
img_000000_south.jpg
img_000000_west.jpg
```

Where `000000` is the **sample ID**.

---

### Training Labels (CSV)

Training data requires a CSV file with:

| Column      | Description                              |
| ----------- | ---------------------------------------- |
| `sample_id` | Integer sample ID                        |
| `state`     | State name (string, e.g. `"California"`) |
| `latitude`  | GPS latitude                             |
| `longitude` | GPS longitude                            |

Example:

```csv
sample_id,state,latitude,longitude
0,California,37.7749,-122.4194
1,Texas,30.2672,-97.7431
```

Internally, state names are mapped to **state indices (0–49)**.

---

### Test Data (No CSV)

For the test set:

* **No CSV file is required**
* The dataset is built by **scanning the image directory**
* Samples are inferred automatically from filenames

Only samples with **all four views present** are used.

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:

* Python 3.8+
* PyTorch
* HuggingFace Transformers
* torchvision
* NumPy
* pandas
* PIL
* Weights & Biases (optional)

---

## Training

Train the model using the provided `train.py`:

```bash
python train.py
```

Key training features:

* Multi-GPU support via `nn.DataParallel`
* CLIP freezing/unfreezing by step count
* Cosine learning rate schedule with warmup
* Automatic checkpointing:

  * Best model
  * Per-epoch checkpoints
  * Final checkpoint
* Optional Weights & Biases logging

Important config options (defined inside `train.py`):

* `freeze_clip_steps`
* `fusion` type
* batch size
* learning rate
* number of epochs

---

## Inference / Prediction

Run inference on a **directory-only test set**:

```bash
python infer.py \
  --ckpt outputs_streetclip_freeze/checkpoints/best.pt \
  --images_dir data/test \
  --out_csv submission.csv \
  --batch_size 64 \
  --amp
```

---

## Output Submission Format

The generated CSV contains:

| Column                  | Description                 |
| ----------------------- | --------------------------- |
| `sample_id`             | Sample ID                   |
| `image_north`           | North-facing image filename |
| `image_east`            | East-facing image filename  |
| `image_south`           | South-facing image filename |
| `image_west`            | West-facing image filename  |
| `predicted_state_idx_1` | Top-1 predicted state       |
| `predicted_state_idx_2` | Top-2 predicted state       |
| `predicted_state_idx_3` | Top-3 predicted state       |
| `predicted_state_idx_4` | Top-4 predicted state       |
| `predicted_state_idx_5` | Top-5 predicted state       |
| `predicted_latitude`    | Predicted latitude          |
| `predicted_longitude`   | Predicted longitude         |

Example:

```csv
sample_id,image_north,image_east,image_south,image_west,predicted_state_idx_1,predicted_state_idx_2,predicted_state_idx_3,predicted_state_idx_4,predicted_state_idx_5,predicted_latitude,predicted_longitude
0,img_000000_north.jpg,img_000000_east.jpg,img_000000_south.jpg,img_000000_west.jpg,4,43,45,18,48,37.7749,-122.4194
```

---

## Project Structure

```
Computer-Vision---Project4/
├── dataset.py          # Train/test dataset loaders
├── model.py            # StreetCLIPMultiView model
├── train.py            # Training script
├── infer.py            # Inference + CSV generation
├── data/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images (directory only)
├── outputs_streetclip_freeze/
│   └── checkpoints/    # Saved checkpoints
├── requirements.txt
└── README.md
```

---

## Notes

* Images are processed using **CLIPProcessor** (no manual resizing needed)
* State indices are **0–49**
* GPS coordinates are predicted in **decimal degrees**
* AMP (`--amp`) is recommended for faster inference on GPU
* The model works on **CPU or GPU** automatically
