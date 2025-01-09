import torch
from dataset import PlantNetDataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import os
import argparse
from tqdm.autonotebook import tqdm
from model import build_model, model_params
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(20, 20))
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

    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="Test CNN model")
    parser.add_argument("--image_size", "-i", type=int, default=320)
    parser.add_argument("--checkpoint", "-p", type=str, default="trained_models/best.pt")
    parser.add_argument('--model_name', type=str, default="model_4")
    parser.add_argument('--num_classes', type=int, default=1081)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args




def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_size = model_params[args.model_name][2]

    transform_test = Compose([
        ToTensor(),
        Resize(size=args.image_size, antialias=True),
        CenterCrop(size=crop_size),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325])
    ])

    test_dataset = PlantNetDataset(args.root, "test", transform=transform_test,
                                   tail_augmentations=None)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_name, args.num_classes)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_params"])
    else:
        print("No checkpoint provided")
        exit(0)


    progress_bar = tqdm(test_dataloader, colour="red")
    all_labels, all_predictions = [], []
    model.eval()
    with torch.no_grad():
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            images_pred = model(images)

            # Get predictions
            prediction_images = torch.argmax(images_pred, dim=1)
            all_labels.extend(labels.tolist())
            all_predictions.extend(prediction_images.tolist())

            # Calculate accuracy, precision and loss for validation
            pre_score = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            acc_score = accuracy_score(all_labels, all_predictions)

        print(f"Precision: {pre_score:.4f}")
        print(f"Accuracy: {acc_score:.4f}")

        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(conf_matrix, test_dataset.categories)


if __name__ == '__main__':
    args = get_args()
    test(args)
