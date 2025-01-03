from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from torch.cuda.amp import autocast, GradScaler


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Compute accuracy
        train_loss += loss.item()  # accumulate batch loss
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_description(desc=f"loss={loss.item():.4f} batch_id={batch_idx}")

    # Compute average loss and accuracy for the epoch
    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total

    print(
        f"\nEpoch {epoch}: Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n"
    )
    return avg_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            test_loss += loss.item()

            # Compute accuracy
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()

    # Compute average loss and accuracy for the test set
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy


def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_image(image, label):
    image = image.permute(1, 2, 0)
    plt.imshow(image.squeeze())
    plt.title(f"Label: {label}")
    plt.show()


# Mixed precision training function
# Fix in train_mp function - adjust loss calculation
def train_mp(model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
    criterion = nn.CrossEntropyLoss()  # Move criterion outside the loop

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)  # More efficient way to zero gradients

        # Mixed precision training: Use autocast for forward pass
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy
        train_loss += loss.item() * target.size(0)  # Multiply by batch size for correct averaging
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_description(desc=f"loss={loss.item():.4f} batch_id={batch_idx}")

    # Compute average loss and accuracy for the epoch
    avg_loss = train_loss / total  # Corrected average loss calculation
    accuracy = 100.0 * correct / total

    print(
        f"\nEpoch {epoch}: Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n"
    )
    return avg_loss, accuracy

# Fix in test_mp function - adjust loss calculation and add autocast
def test_mp(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast():  # Add autocast for consistency with training
                output = model(data)
                loss = criterion(output, target)

            # Compute loss
            test_loss += loss.item() * target.size(0)  # Multiply by batch size for correct averaging
            total += target.size(0)

            # Compute accuracy
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()

    # Compute average loss and accuracy for the test set
    test_loss /= total  # Corrected average loss calculation
    accuracy = 100.0 * correct / total

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, total, accuracy
        )
    )
    return test_loss, accuracy