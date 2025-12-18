# Quick Start Guide

This guide will help you quickly get started with the GeoGuessr StreetView project.

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd Computer-Vision---Project4
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Create the `data` directory structure (see `data/DATA_FORMAT.md`)
   - Place your images in `data/train/`, `data/val/`, and `data/test/`
   - Create CSV label files: `data/train_labels.csv` and `data/val_labels.csv`

## Training

Train the model with default settings:

```bash
python src/train.py \
    --data-dir data/train \
    --csv-file data/train_labels.csv \
    --val-dir data/val \
    --val-csv data/val_labels.csv
```

The best model will be saved to `models/best_model.pth`.

## Making Predictions

Generate predictions on test data:

```bash
python src/predict.py \
    --test-dir data/test \
    --model-path models/best_model.pth \
    --output-csv submission.csv
```

This will create a `submission.csv` file with the required format.

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (assuming data is prepared)
python src/train.py \
    --data-dir data/train \
    --csv-file data/train_labels.csv \
    --val-dir data/val \
    --val-csv data/val_labels.csv \
    --epochs 30 \
    --batch-size 16

# 3. Generate predictions
python src/predict.py \
    --test-dir data/test \
    --model-path models/best_model.pth \
    --output-csv submission.csv \
    --num-top-states 5

# 4. Check the submission file
head submission.csv
```

## Tips

- **GPU Training**: The code automatically uses GPU if available. For faster training, use a machine with CUDA-enabled GPU.
- **Batch Size**: Adjust `--batch-size` based on your GPU memory. Smaller batch sizes (4-8) work on most GPUs.
- **Epochs**: Start with 20-30 epochs and increase if the validation loss is still decreasing.
- **Learning Rate**: The default learning rate is 0.001. If training is unstable, try reducing it to 0.0001.

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch-size 4`
- Use CPU instead of GPU (slower but uses system RAM)

### Missing Images
- Check that your image filenames follow the naming convention: `img_XXXXXX_direction.jpg`
- Ensure all 4 directions (north, east, south, west) are present for each sample

### Poor Model Performance
- Train for more epochs
- Check your data quality and labels
- Consider data augmentation (add to the transforms in train.py)
- Use a larger model or more training data

## Next Steps

After getting predictions, you can:
1. Evaluate model performance on validation data
2. Fine-tune hyperparameters
3. Experiment with different model architectures
4. Add data augmentation for better generalization
5. Ensemble multiple models for improved accuracy

For more details, see the main README.md file.
