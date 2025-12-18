# Data Structure Example

This document shows the expected data directory structure for the GeoGuessr StreetView project.

## Directory Structure

```
data/
├── train/
│   ├── img_000001_north.jpg
│   ├── img_000001_east.jpg
│   ├── img_000001_south.jpg
│   ├── img_000001_west.jpg
│   ├── img_000002_north.jpg
│   ├── img_000002_east.jpg
│   ├── img_000002_south.jpg
│   ├── img_000002_west.jpg
│   └── ...
├── val/
│   ├── img_000100_north.jpg
│   ├── img_000100_east.jpg
│   ├── img_000100_south.jpg
│   ├── img_000100_west.jpg
│   └── ...
├── test/
│   ├── img_001000_north.jpg
│   ├── img_001000_east.jpg
│   ├── img_001000_south.jpg
│   ├── img_001000_west.jpg
│   └── ...
├── train_labels.csv
└── val_labels.csv
```

## Label CSV Format

### Training/Validation Labels (train_labels.csv, val_labels.csv)

```csv
sample_id,state_idx_1,latitude,longitude
1,5,34.0522,-118.2437
2,33,40.7128,-74.0060
3,48,47.6062,-122.3321
```

**Columns:**
- `sample_id`: Integer ID matching the image filename (e.g., 1 for img_000001_*.jpg)
- `state_idx_1`: State index from 1-50 representing US states
- `latitude`: GPS latitude in decimal degrees
- `longitude`: GPS longitude in decimal degrees

## Image Naming Convention

Images must follow this naming pattern:
```
img_{SAMPLE_ID}_{DIRECTION}.jpg
```

Where:
- `SAMPLE_ID`: 6-digit zero-padded number (e.g., 000001, 000042, 123456)
- `DIRECTION`: One of `north`, `east`, `south`, `west`

Examples:
- `img_000001_north.jpg`
- `img_000042_east.jpg`
- `img_123456_south.jpg`

## State Index Mapping

State indices are 1-indexed integers from 1 to 50, representing the 50 US states in alphabetical order:

1. Alabama
2. Alaska
3. Arizona
4. Arkansas
5. California
6. Colorado
7. Connecticut
8. Delaware
9. Florida
10. Georgia
... (and so on)

Note: You should create your own mapping based on your specific dataset.
