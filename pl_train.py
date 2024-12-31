import os
from datetime import datetime
from typing import Optional, Tuple
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from loguru import logger

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        logger.info(f"\n{'='*20} Epoch {trainer.current_epoch} {'='*20}")

class ImageNetModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        num_workers: int = 16,
        max_epochs: int = 90,
        train_path: str = "path/to/imagenet",
        val_path: str = "path/to/imagenet",
        checkpoint_dir: str = "checkpoints"
    ):
        super().__init__()
        # self.save_hyperparameters()
        
        # Model
        self.model = models.resnet50(weights=None)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.train_path = train_path
        self.val_path = val_path
        self.checkpoint_dir = checkpoint_dir
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.best_val_acc = 0.0
        
        # Set up transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = (correct / labels.size(0))*100
        
        # Log metrics for this step
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'acc': torch.tensor(accuracy)
        })
        
        return loss

    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            print("Warning: No training outputs available for this epoch")
            return
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()
        
        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        logger.info(f"Training metrics - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, LR: {current_lr:.6f}")
        
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = (correct / labels.size(0))*100
        
        # Log metrics for this step
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append({
            'val_loss': loss.detach(),
            'val_acc': torch.tensor(accuracy)
        })
        
        return {'val_loss': loss, 'val_acc': accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
        
        # Log final validation metrics
        self.log('val_loss_epoch', avg_loss)
        self.log('val_acc_epoch', avg_acc)
        
        # Save checkpoint if validation accuracy improves
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"resnet50-epoch{self.current_epoch:02d}-acc{avg_acc:.4f}.ckpt"
            )
            self.trainer.save_checkpoint(checkpoint_path)
            logger.info(f"New best validation accuracy: {avg_acc:.4f}. Saved checkpoint to {checkpoint_path}")
        
        logger.info(f"Validation metrics - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        self.validation_step_outputs.clear()

    def train_dataloader(self):
        train_dataset = datasets.ImageFolder(
            self.train_path,
            transform=self.train_transforms
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        val_dataset = datasets.ImageFolder(
            self.val_path,
            transform=self.val_transforms
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.max_epochs,
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
        colorize=True,
        level="INFO"
    )
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="30 days"
    )
    
    logger.info(f"Logging setup complete. Logs will be saved to: {log_file}")
    return log_file

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file using various possible naming patterns."""
    # Look for checkpoint files with different possible patterns
    patterns = [
        "*.ckpt",  # Generic checkpoint files
        "resnet50-epoch*.ckpt",  # Our custom format
        "*epoch=*.ckpt",  # PyTorch Lightning default format
        "checkpoint_epoch*.ckpt"  # Another common format
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        checkpoint_pattern = os.path.join(checkpoint_dir, pattern)
        all_checkpoints.extend(glob.glob(checkpoint_pattern))
    
    if not all_checkpoints:
        logger.info("No existing checkpoints found.")
        return None
    
    def extract_info(checkpoint_path: str) -> Tuple[int, float]:
        """Extract epoch and optional accuracy from checkpoint filename."""
        filename = os.path.basename(checkpoint_path)
        
        # Try different patterns to extract epoch number
        epoch_patterns = [
            r'epoch=(\d+)',  # matches epoch=X
            r'epoch(\d+)',   # matches epochX
            r'epoch[_-](\d+)',  # matches epoch_X or epoch-X
        ]
        
        epoch = None
        for pattern in epoch_patterns:
            match = re.search(pattern, filename)
            if match:
                epoch = int(match.group(1))
                break
        
        # If no epoch found, try to get from file modification time
        if epoch is None:
            epoch = int(os.path.getmtime(checkpoint_path))
        
        # Try to extract accuracy if present
        acc_match = re.search(r'acc[_-]?([\d.]+)', filename)
        acc = float(acc_match.group(1)) if acc_match else 0.0
        
        return epoch, acc
    
    try:
        latest_checkpoint = max(all_checkpoints, key=lambda x: extract_info(x)[0])
        epoch, acc = extract_info(latest_checkpoint)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        logger.info(f"Epoch: {epoch}" + (f", Accuracy: {acc:.4f}" if acc > 0 else ""))
        return latest_checkpoint
    except Exception as e:
        logger.error(f"Error processing checkpoints: {str(e)}")
        # If there's any error in parsing, return the most recently modified file
        latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
        logger.info(f"Falling back to most recently modified checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def main():
    checkpoint_dir = "/home/ec2-user/ebs/volumes/era_session9"
    log_file = setup_logging(log_dir=checkpoint_dir)
    
    logger.info("Starting training with configuration:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # Find latest checkpoint
    # latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    latest_checkpoint = "/home/ec2-user/ebs/volumes/era_session9/resnet50-epoch18-acc53.7369.ckpt"
    
    model = ImageNetModule(
        learning_rate=0.156,
        batch_size=256,
        num_workers=16,
        max_epochs=60,
        train_path="/home/ec2-user/ebs/volumes/imagenet/ILSVRC/Data/CLS-LOC/train",
        val_path="/home/ec2-user/ebs/volumes/imagenet/imagenet_validation",
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info(f"Model configuration:")
    logger.info(f"Learning rate: {model.learning_rate}")
    logger.info(f"Batch size: {model.batch_size}")
    logger.info(f"Number of workers: {model.num_workers}")
    logger.info(f"Max epochs: {model.max_epochs}")
    
    progress_bar = CustomProgressBar()
    
    trainer = Trainer(
        max_epochs=60,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision=16,
        callbacks=[progress_bar],
        enable_progress_bar=True,
    )
    
    logger.info("Starting training")
    
    try:
        if latest_checkpoint:
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            trainer.fit(model, ckpt_path=latest_checkpoint)
        else:
            logger.info("Starting training from scratch")
            trainer.fit(model)
            
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        logger.info(f"Training session ended. Log file: {log_file}")

if __name__ == "__main__":
    main()
    # pass