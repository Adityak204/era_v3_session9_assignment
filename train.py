import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import copy
import multiprocessing

# Import custom libraries
from src.classifier import ResNet, Bottleneck
from src.utils import train, test, seed_everything
from src.trainer import Trainer

if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
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
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root="/home/ec2-user/ebs/volumes/imagenet/ILSVRC/Data/CLS-LOC/train",
        transform=train_transformation,
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    config = {
        "train_path": "/home/ec2-user/ebs/volumes/imagenet/ILSVRC/Data/CLS-LOC/train",
        "val_path": "/home/ec2-user/ebs/volumes/imagenet/imagenet_validation",
        "batch_size": 512,
        "num_workers": 4 * torch.cuda.device_count(),
        "epochs": 100,
        "artifact_path": "/home/ec2-user/ebs/volumes/era_session9",
    }

    optimizer = optim.SGD(
        resnet_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.351,
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0,
        three_phase=False,
        last_epoch=-1,
        verbose="deprecated",
    )
    training = Trainer(
        model=resnet_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_path=config["train_path"],
        val_path=config["val_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        epochs=config["epochs"],
        artifact_path=config["artifact_path"],
    )
    training.main()
