import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
import os
import time
import json
from loguru import logger

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.cuda.amp import autocast, GradScaler

# Import custom libraries
from src.classifier import ResNet
from src.utils import train, test, seed_everything, train_mp, test_mp


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        train_path,
        val_path,
        batch_size,
        num_workers,
        epochs,
        artifact_path,
        log_path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.artifact_path = artifact_path
        self.log_path = log_path

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logger.add(
            self.log_path,
            format="{time} {level} {message}",
            level="INFO",
            rotation="10 MB",
        )

    def data_loader(self):
        train_transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    224,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(0.5),
                # Normalize the pixel values (in R, G, and B channels)
                transforms.Normalize(
                    mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_path, transform=train_transformation
        )
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=256, antialias=True),
                transforms.CenterCrop(224),
                # Normalize the pixel values (in R, G, and B channels)
                transforms.Normalize(
                    mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=self.val_path, transform=val_transformation
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        return train_loader, val_loader

    def main(self):
        train_loader, val_loader = self.data_loader()
        model = self.model.to(self.device)
        model = nn.DataParallel(model)
        model = model.to(self.device)

        optimizer = self.optimizer
        scheduler = self.scheduler

        # Initialize the best accuracy tracker
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            logger.info(f"********* Epoch {epoch}/{self.epochs} *********")
            train_loss, train_acc = train(model, self.device, train_loader, optimizer, epoch)
            test_loss, test_acc = test(model, self.device, val_loader)
            scheduler.step()
            print("LR = ", scheduler.get_last_lr())

            # Log metrics
            logger.info(f"Epoch: {epoch}")
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
            )
            logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Save model checkpoint if the accuracy improves
            if test_acc > best_acc:
                print(
                    f"Test accuracy improved from {best_acc:.4f} to {acc:.4f}. Saving model..."
                )
                best_acc = test_acc

                # Save the model checkpoint with optimizer state, epoch, and learning rate
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "accuracy": test_acc,
                    "learning_rate": scheduler.get_last_lr()[
                        0
                    ],  # Assuming a single LR value for simplicity
                }

                # Create a file path to save the checkpoint
                checkpoint_path = (
                    f"{self.artifact_path}/best_model_epoch_{epoch}_acc_{acc:.4f}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")


class TrainerMP:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        train_path,
        val_path,
        batch_size,
        num_workers,
        epochs,
        artifact_path,
        log_path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.artifact_path = artifact_path
        self.log_path = log_path

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logger.add(
            self.log_path,
            format="{time} {level} {message}",
            level="INFO",
            rotation="10 MB",
        )

    def data_loader(self):
        train_transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    224,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(0.5),
                # Normalize the pixel values (in R, G, and B channels)
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_path, transform=train_transformation
        )
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=256, antialias=True),
                transforms.CenterCrop(224),
                # Normalize the pixel values (in R, G, and B channels)
                transforms.Normalize(
                    mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=self.val_path, transform=val_transformation
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        return train_loader, val_loader

    def main(self):
        train_loader, val_loader = self.data_loader()
        model = self.model.to(self.device)
        model = nn.DataParallel(model)
        model = model.to(self.device)

        # Before training loop
        torch.cuda.empty_cache()
        optimizer = self.optimizer
        scheduler = self.scheduler

        # Initialize the best accuracy tracker
        best_acc = 0.0

        # Initialize the mixed precision scaler
        scaler = GradScaler()

        for epoch in range(1, self.epochs + 1):
            logger.info(f"********* Epoch {epoch}/{self.epochs} *********")
            train_loss, train_acc = train_mp(
                model, self.device, train_loader, optimizer, epoch, scaler
            )
            test_loss, test_acc = test_mp(model, self.device, val_loader)
            # Learning rate scheduling
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0] if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else optimizer.param_groups[0]['lr']
            logger.info(f"Current Learning Rate: {current_lr:.6f}")

            # Log metrics
            logger.info(f"Epoch: {epoch}")
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
            )
            logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Save model checkpoint if the accuracy improves
            if test_acc > best_acc:
                print(
                    f"Test accuracy improved from {best_acc:.4f} to {test_acc:.4f}. Saving model..."
                )
                best_acc = test_acc

                # Save the model checkpoint with optimizer state, epoch, and learning rate
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'accuracy': test_acc,
                    'loss': test_loss,
                    'learning_rate': current_lr,
                }

                # Create a file path to save the checkpoint
                checkpoint_path = f"{self.artifact_path}/best_model_epoch_{epoch}_acc_{test_acc:.4f}.pth"
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
