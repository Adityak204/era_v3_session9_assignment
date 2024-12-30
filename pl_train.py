import os
from datetime import datetime
from typing import Optional, Tuple

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
    ):
        super().__init__()
        self.save_hyperparameters()
        
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
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
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
        accuracy = correct / labels.size(0)
        
        # Log metrics for this step
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'acc': torch.tensor(accuracy)
        })
        
        return loss

    def on_train_epoch_end(self):
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
        accuracy = correct / labels.size(0)
        
        # Log metrics for this step
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True,)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True,)
        
        self.validation_step_outputs.append({
            'val_loss': loss.detach(),
            'val_acc': torch.tensor(accuracy)
        })
        
        return {'val_loss': loss, 'val_acc': accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
        
        # Log final validation metrics
        self.log('val_loss_epoch', avg_loss,)
        self.log('val_acc', avg_acc,)
        
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
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with colored output
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
        colorize=True,
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="30 days"  # Keep logs for 30 days
    )
    
    logger.info(f"Logging setup complete. Logs will be saved to: {log_file}")
    return log_file

def main():
    # Set up logging
    log_file = setup_logging(log_dir="/home/ec2-user/ebs/volumes/era_session9")
    
    # Log system information
    logger.info("Starting training with configuration:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # Initialize model and trainer
    model = ImageNetModule(
        learning_rate=0.156,
        batch_size=256,
        num_workers=16,
        max_epochs=40,
        train_path = "/home/ec2-user/ebs/volumes/imagenet/ILSVRC/Data/CLS-LOC/train",
        val_path = "/home/ec2-user/ebs/volumes/imagenet/imagenet_validation",
    )
    
    # Log model configuration
    logger.info(f"Model configuration:")
    logger.info(f"Learning rate: {model.learning_rate}")
    logger.info(f"Batch size: {model.batch_size}")
    logger.info(f"Number of workers: {model.num_workers}")
    logger.info(f"Max epochs: {model.max_epochs}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/ec2-user/ebs/volumes/era_session9",
        filename="resnet50-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        verbose=True
    )
    
    progress_bar = CustomProgressBar()
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=40,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision=16,
        callbacks=[checkpoint_callback, progress_bar],
        enable_progress_bar=True,
    )
    
    # Log training start
    logger.info("Starting training")
    
    try:
        # Start training
        trainer.fit(model)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        logger.info(f"Training session ended. Log file: {log_file}")

if __name__ == "__main__":
    main()