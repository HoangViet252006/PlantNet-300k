from collections import Counter
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from dataset import PlantNetDataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, ColorJitter, \
    RandomRotation, RandomCrop, CenterCrop
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import numpy as np
import os
import argparse
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from cli import add_all_parsers
from model import build_model, model_params
import shutil
import torch


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="plasma")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_size = model_params[args.model_name][2]


    # Ensure the checkpoint and TensorBoard directories exist
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    # Set up TensorBoard writer for logging
    if os.path.exists(args.tensorboard_dir):
        shutil.rmtree(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)


    # Augmentation for tail classes (low-frequency classes)
    tail_augmentations = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=30),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        Resize(size=args.image_size, antialias=True),
        RandomCrop(size=crop_size),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325]),
    ])

    transform_train = Compose([
        ToTensor(),
        Resize(size=args.image_size, antialias=True),
        RandomCrop(size=crop_size),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325]),
    ])

    transform_val = Compose([
        ToTensor(),
        Resize(size=args.image_size, antialias=True),
        CenterCrop(size=crop_size),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325])
    ])

    train_dataset = PlantNetDataset(args.root, "train", transform=transform_train,
                                    tail_augmentations=tail_augmentations)


    # Count class frequencies and calculate class weights for balancing
    class_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset)
    num_classes = len(class_counts)
    class_weights = {
        class_id: total_samples / (num_classes * max(count, 1))  # Use `max` to prevent division by zero
        for class_id, count in class_counts.items()
    }

    # Create a list of sample weights for each class in the dataset
    sample_weights = [class_weights[label] for label in train_dataset.labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Set tail classes for augmentation
    tail_classes = [class_id for class_id, count in class_counts.items() if count < 200]
    train_dataset.set_tail_classes(tail_classes)

    oversampled_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,  # shuffle is handled by the sampler
        num_workers=args.num_workers,
        sampler=sampler
    )

    val_dataset = PlantNetDataset(args.root, "val", transform=transform_val,
                                  tail_augmentations=None)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Create model
    model = build_model(args.model_name, args.num_classes)
    model = model.to(device)


    # Initialize MultiClassFocalLoss
    alpha = torch.tensor([class_weights.get(c, 1.0) for c in range(len(class_counts))]).to(device)
    criterion = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss',
                                alpha=alpha, gamma=2, reduction='mean', trust_repo=True, force_reload=False)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    num_iters = len(oversampled_dataloader)
    total_steps = args.num_epochs * num_iters

    scheduler = OneCycleLR(optimizer, max_lr= 5 * args.lr, total_steps=total_steps)

    # Load checkpoint if available, else initialize variables
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_pre = checkpoint["best_pre"]
        model.load_state_dict(checkpoint["model_params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        best_pre = -1
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        # Train mode
        model.train()
        progress_bar = tqdm(oversampled_dataloader, colour="cyan")
        all_losses = []

        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            # Forward pass for train
            images_pred = model(images)
            loss = criterion(images_pred, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            all_losses.append(loss.item())
            progress_bar.set_description(
                f"Epoch: {epoch + 1}/{args.num_epochs}. Loss: {loss:.4f}"
            )
            writer.add_scalar('Train/Loss', np.mean(all_losses), epoch * num_iters + iter)

        # Validation mode
        model.eval()
        all_labels, all_predictions = [], []
        all_losses = []
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="green")
            for iter, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)

                # Forward pass for validation
                images_pred = model(images)
                loss = criterion(images_pred, labels)

                all_losses.append(loss.item())

                # Get predictions
                prediction_images = torch.argmax(images_pred, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(prediction_images.tolist())

        # Calculate accuracy, precision and loss for validation
        pre_score = precision_score(all_labels, all_predictions, average='weighted',  zero_division=0)
        acc_score = accuracy_score(all_labels, all_predictions)
        loss_value = np.mean(all_losses)

        print(f"Epoch: {epoch + 1}/{args.num_epochs}. Precision: {pre_score:.4f}. Loss: {loss_value:.4f}")
        writer.add_scalar('Val/Loss', loss_value, epoch)
        writer.add_scalar('Val/Metrics', {"Precision": pre_score, "Accuracy": acc_score}, epoch)

        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)

        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "best_pre": pre_score,
            "acc_score": acc_score,
            "model_params": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if best_pre < pre_score:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            best_pre = pre_score

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    train(args)