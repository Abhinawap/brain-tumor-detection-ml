# Experiments

- `train_segmentation.py` - trains the U-Net with MLflow tracking on a seeded train/val/test split, then evaluates the best checkpoint once on the test split. Run `python experiments/train_segmentation.py --help` for options.
