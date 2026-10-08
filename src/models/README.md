# Models Module

- `unet.py` - U-Net encoder/decoder with sigmoid output
- `metrics.py` - Dice, IoU, pixel accuracy, sensitivity, specificity (`SegmentationMetrics` thresholds predictions at 0.5 for all of them)
- `losses.py` - Dice and BCE + Dice losses
