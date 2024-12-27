import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
import os
import time

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
from src.utils import train, test, seed_everything


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
            print(f"********* Epoch = {epoch} *********")
            train(model, self.device, train_loader, optimizer, epoch)
            _, acc = test(model, self.device, val_loader)
            scheduler.step(acc)
            print("LR = ", scheduler.get_last_lr())
            # Save model checkpoint if the accuracy improves
            if acc > best_acc:
                print(
                    f"Test accuracy improved from {best_acc:.4f} to {acc:.4f}. Saving model..."
                )
                best_acc = acc

                # Save the model checkpoint with optimizer state, epoch, and learning rate
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "accuracy": acc,
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
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()

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

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Runs the forward pass with autocasting
            with autocast():
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)

            # Scales loss and calls backward() to create scaled gradients
            self.scaler.scale(loss).backward()

            # Unscales gradients and calls or skips optimizer.step()
            self.scaler.step(optimizer)

            # Updates the scale for next iteration
            self.scaler.update()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    def test(self, model, device, val_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                with autocast():
                    output = model(data)
                    test_loss += nn.CrossEntropyLoss()(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)
        accuracy = 100.0 * correct / len(val_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        return test_loss, accuracy

    def main(self):
        train_loader, val_loader = self.data_loader()
        model = self.model.to(self.device)
        model = nn.DataParallel(model)
        model = model.to(self.device)

        optimizer = self.optimizer
        scheduler = self.scheduler

        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            print(f"********* Epoch = {epoch} *********")
            self.train(model, self.device, train_loader, optimizer, epoch)
            _, acc = self.test(model, self.device, val_loader)
            scheduler.step(acc)
            print("LR = ", scheduler.get_last_lr())

            if acc > best_acc:
                print(
                    f"Test accuracy improved from {best_acc:.4f} to {acc:.4f}. Saving model..."
                )
                best_acc = acc

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),  # Save scaler state
                    "accuracy": acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                }

                checkpoint_path = (
                    f"{self.artifact_path}/best_model_epoch_{epoch}_acc_{acc:.4f}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
