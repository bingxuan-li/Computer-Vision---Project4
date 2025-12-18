# GeoGuessr StreetView

A Computer Vision project for predicting US state location and GPS coordinates from StreetView images.

## Overview

This project implements a multi-task deep learning model that takes four directional StreetView images (north, east, south, west) and predicts:
1. **State Classification**: Top 1-5 most likely US states
2. **GPS Coordinates**: Latitude and longitude of the location

## Dataset Format

### Input Images
For each sample, the dataset should contain 4 images:
- `img_XXXXXX_north.jpg` - North-facing view
- `img_XXXXXX_east.jpg` - East-facing view
- `img_XXXXXX_south.jpg` - South-facing view
- `img_XXXXXX_west.jpg` - West-facing view

Where `XXXXXX` is the sample ID (e.g., `000001`, `000042`, etc.)

### Training Labels (CSV Format)
Training labels should be provided in a CSV file with the following columns:
- `sample_id`: Integer ID of the sample
- `state_idx_1`: Primary state prediction (1-50, representing US states)
- `latitude`: GPS latitude coordinate
- `longitude`: GPS longitude coordinate

Example:
```csv
sample_id,state_idx_1,latitude,longitude
1,5,34.0522,-118.2437
2,33,40.7128,-74.0060
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on your dataset:

```bash
python src/train.py \
    --data-dir data/train \
    --csv-file data/train_labels.csv \
    --val-dir data/val \
    --val-csv data/val_labels.csv \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.001 \
    --num-states 50 \
    --checkpoint-dir models
```

**Arguments:**
- `--data-dir`: Directory containing training images
- `--csv-file`: Path to training labels CSV
- `--val-dir`: Directory containing validation images
- `--val-csv`: Path to validation labels CSV
- `--batch-size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--num-states`: Number of states to classify (default: 50)
- `--checkpoint-dir`: Directory to save model checkpoints (default: models)

### Prediction

Generate predictions on test data:

```bash
python src/predict.py \
    --test-dir data/test \
    --model-path models/best_model.pth \
    --output-csv submission.csv \
    --batch-size 8 \
    --num-states 50 \
    --num-top-states 5
```

**Arguments:**
- `--test-dir`: Directory containing test images
- `--model-path`: Path to trained model checkpoint
- `--output-csv`: Path to output submission CSV file (default: submission.csv)
- `--batch-size`: Batch size for inference (default: 8)
- `--num-states`: Number of states in the model (default: 50)
- `--num-top-states`: Number of top state predictions to include (1-5, default: 5)

### Submission File Format

The generated submission CSV will contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | int | Test sample ID |
| `image_north` | str | Filename of north-facing image |
| `image_east` | str | Filename of east-facing image |
| `image_south` | str | Filename of south-facing image |
| `image_west` | str | Filename of west-facing image |
| `state_idx_1` | int | Primary state prediction (required) |
| `state_idx_2` | int | Second state prediction (optional) |
| `state_idx_3` | int | Third state prediction (optional) |
| `state_idx_4` | int | Fourth state prediction (optional) |
| `state_idx_5` | int | Fifth state prediction (optional) |
| `latitude` | float | Predicted latitude |
| `longitude` | float | Predicted longitude |

Example submission.csv:
```csv
sample_id,image_north,image_east,image_south,image_west,state_idx_1,state_idx_2,state_idx_3,state_idx_4,state_idx_5,latitude,longitude
1,img_000001_north.jpg,img_000001_east.jpg,img_000001_south.jpg,img_000001_west.jpg,5,33,48,6,36,34.0522,-118.2437
2,img_000002_north.jpg,img_000002_east.jpg,img_000002_south.jpg,img_000002_west.jpg,33,22,9,5,36,40.7128,-74.0060
```

## Model Architecture

The model uses a multi-task learning approach with:

1. **Feature Extraction**: ResNet-50 pretrained on ImageNet
2. **Multi-Image Processing**: Processes all 4 directional images independently
3. **Feature Fusion**: Concatenates and fuses features from all directions
4. **Multi-Task Heads**:
   - State Classification head: Predicts probability distribution over 50 states
   - GPS Regression head: Predicts latitude and longitude coordinates

## Project Structure

```
Computer-Vision---Project4/
├── src/
│   ├── dataset.py      # Dataset loader for 4-direction images
│   ├── model.py        # Model architecture
│   ├── train.py        # Training script
│   └── predict.py      # Inference and submission generation
├── data/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── models/             # Saved model checkpoints
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- PIL
- NumPy
- pandas
- scikit-learn
- tqdm

## Notes

- The model expects images in JPG format
- All images are resized to 224x224 for processing
- State indices are 1-indexed (1-50)
- GPS coordinates are in decimal degrees
- The model can run on CPU or GPU (automatically detected)