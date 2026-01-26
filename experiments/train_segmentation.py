"""
Training script for brain tumor segmentation with U-Net and MLflow tracking.

This script handles:
- Dataset loading and splitting
- Model initialization
- Training loop with metric logging
- MLflow experiment tracking
- Validation and checkpointing

Usage:
    python experiments/train_segmentation_mlflow.py --epochs 50 --batch-size 16
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import BrainTumorDatasetWithAugmentation
from src.models.unet import UNet
from src.models.metrics import SegmentationMetrics
from src.models.losses import BCEDiceLoss


class MLflowTrainer:
    """
    Trainer for U-Net segmentation model with MLflow tracking.

    Handles training loop, validation, metrics logging to MLflow, and checkpointing.

    Args:
        model: U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save model checkpoints
        log_interval: How often to log metrics (default: 10 batches)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        checkpoint_dir: str = 'models/checkpoints',
        log_interval: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.metrics = SegmentationMetrics()
        self.best_val_dice = 0.0

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        batch_losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            batch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log batch metrics to MLflow
            if batch_idx % self.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                mlflow.log_metric('train_batch_loss', loss.item(), step=step)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0
        }

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Calculate metrics
            batch_metrics = self.metrics(outputs, masks)

            # Accumulate
            total_loss += loss.item()
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

            pbar.set_postfix({'dice': f'{batch_metrics["dice"]:.4f}'})

        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        for key in all_metrics:
            all_metrics[key] /= len(self.val_loader)

        all_metrics['loss'] = avg_loss
        return all_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

            # Log best model to MLflow
            mlflow.pytorch.log_model(self.model, "best_model")
            print(f'✓ Saved best model (Dice: {metrics["dice"]:.4f})')

    def train(self, num_epochs: int):
        """
        Full training loop with MLflow logging.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])

            # Log epoch metrics to MLflow
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_metrics['loss'], step=epoch)
            mlflow.log_metric('val_dice', val_metrics['dice'], step=epoch)
            mlflow.log_metric('val_iou', val_metrics['iou'], step=epoch)
            mlflow.log_metric('val_accuracy', val_metrics['accuracy'], step=epoch)
            mlflow.log_metric('val_sensitivity', val_metrics['sensitivity'], step=epoch)
            mlflow.log_metric('val_specificity', val_metrics['specificity'], step=epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Dice:   {val_metrics['dice']:.4f}")
            print(f"  Val IoU:    {val_metrics['iou']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            # Save checkpoint
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']

            self.save_checkpoint(epoch, val_metrics, is_best)
            print("=" * 60)

        print(f"\nTraining complete!")
        print(f"Best Val Dice: {self.best_val_dice:.4f}")

        # Log final best metric
        mlflow.log_metric('best_val_dice', self.best_val_dice)


def main():
    """Main training function with MLflow tracking."""
    parser = argparse.ArgumentParser(description='Train U-Net for brain tumor segmentation with MLflow')

    # Data arguments
    parser.add_argument('--data-dir', type=str, 
                        default='data/raw/Brain-Tumor-Segmentation-Dataset',
                        help='Path to dataset')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image size (default: 128)')
    parser.add_argument('--classes', type=int, nargs='+', default=[1, 2, 3],
                        help='Classes to include (default: 1 2 3, excluding no-tumor)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split (default: 0.2)')

    # Model arguments
    parser.add_argument('--loss', type=str, default='bce_dice',
                        choices=['bce', 'dice', 'bce_dice', 'focal'],
                        help='Loss function (default: bce_dice)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for BCE+Dice loss (default: 0.5)')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Checkpoint directory (default: models/checkpoints)')

    # MLflow arguments
    parser.add_argument('--experiment-name', type=str, default='brain-tumor-segmentation',
                        help='MLflow experiment name (default: brain-tumor-segmentation)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='MLflow run name (default: auto-generated)')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"unet_{args.loss}_ep{args.epochs}_bs{args.batch_size}_{timestamp}"

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_param('data_dir', args.data_dir)
        mlflow.log_param('image_size', args.image_size)
        mlflow.log_param('classes', args.classes)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('batch_size', args.batch_size)
        mlflow.log_param('learning_rate', args.lr)
        mlflow.log_param('val_split', args.val_split)
        mlflow.log_param('loss_function', args.loss)
        mlflow.log_param('alpha', args.alpha)
        mlflow.log_param('device', device)
        mlflow.log_param('optimizer', 'Adam')

        # Load dataset
        print(f"\nLoading dataset from {args.data_dir}")
        dataset = BrainTumorDatasetWithAugmentation(
            data_dir=args.data_dir,
            image_size=args.image_size,
            classes=args.classes,
            augment=True
        )

        print(f"Total samples: {len(dataset)}")
        dist = dataset.get_class_distribution()
        print(f"Class distribution: {dist}")

        # Log dataset info
        mlflow.log_param('total_samples', len(dataset))
        mlflow.log_param('class_distribution', str(dist))

        # Split into train and validation
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Disable augmentation for validation
        val_dataset.dataset.augment = False

        mlflow.log_param('train_samples', train_size)
        mlflow.log_param('val_samples', val_size)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True if device == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True if device == 'cuda' else False
        )

        # Create model
        print(f"\nInitializing U-Net model")
        model = UNet(in_channels=3, out_channels=1, features=64)
        param_count = model.count_parameters()
        print(f"Parameters: {param_count:,}")
        mlflow.log_param('model_parameters', param_count)

        # Loss function
        if args.loss == 'bce_dice':
            criterion = BCEDiceLoss(alpha=args.alpha)
        else:
            from src.models.losses import get_loss_function
            criterion = get_loss_function(args.loss)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Create trainer
        trainer = MLflowTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )

        # Train
        trainer.train(args.epochs)

        # Log final model artifact
        mlflow.pytorch.log_model(model, "final_model")

        print(f"\n✓ MLflow run completed: {mlflow.active_run().info.run_id}")
        print(f"✓ View results: mlflow ui")


if __name__ == '__main__':
    main()