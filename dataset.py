import os
import json
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop, Normalize, CenterCrop, RandomHorizontalFlip, \
    RandomRotation, ColorJitter
from collections import Counter
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

class PlantNetDataset(Dataset):
    def __init__(self, sub_path, type_data="train", transform=None, tail_augmentations=None):
        assert type_data in ["train", "test", "val"]

        # Load label dictionary
        label_dict_path = os.path.join(sub_path, "class_idx_to_species_id.json")
        with open(label_dict_path, "r") as f:
            self.label_to_species = json.load(f)
        self.species_to_label = {species_id: class_idx for class_idx, species_id in self.label_to_species.items()}
        # Load class name dictionary
        class_name_dict_path = os.path.join(sub_path, "plantnet300K_species_id_2_name.json")
        with open(class_name_dict_path, "r") as f:
            self.species_to_class = json.load(f)

        self.categories = [self.species_to_class[species_id] for _, species_id in self.label_to_species.items()]
        # Load JSON file plant metadata
        metadata_path = os.path.join(sub_path, "plantnet300K_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.data = [
            os.path.join(f"{sub_path}/images/{type_data}/{plant_dict['species_id']}", f"{image_name}.jpg")
            for image_name, plant_dict in metadata.items()
            if plant_dict["split"] == type_data
        ]
        self.labels = [
            int(self.species_to_label[plant_dict["species_id"]])
            for plant_dict in metadata.values()
            if plant_dict["split"] == type_data
        ]

        self.transform = transform
        self.tail_augmentations = tail_augmentations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = self.data[item]
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Failed to load image at {image_path}")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[item]

        if self.tail_augmentations is not None and label in self.tail_classes:
            image = self.tail_augmentations(image)
        elif self.transform:
            image = self.transform(image)

        return image, label

    def set_tail_classes(self, tail_classes):
        self.tail_classes = tail_classes

    def get_class(self, label):
        species_id = self.label_to_species[label]
        return self.species_to_class[species_id]


def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    progress_bar = tqdm(loader, colour="yellow")
    for images, _ in progress_bar:
        progress_bar.set_description()
        # Flatten the image dimensions
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten height x width for each channel

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std




if __name__ == '__main__':
    # caculate mean and std of dataset

    # transform = Compose([
    #     ToTensor(),
    #     Resize((300, 300), antialias=True),
    # ])
    #
    # dataset = PlantNetDataset("plantnet_300K", "train", transform=transform,
    #                           tail_augmentations=None)
    # dataset_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    #
    # mean, std = calculate_mean_std(dataset_loader)
    # print(f"Mean: {', '.join([f'{m:.4f}' for m in mean])}")
    # print(f"Std: {', '.join([f'{s:.4f}' for s in std])}")


    # Augmentation for tail classes
    tail_augmentations = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=30),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        Resize(size=320, antialias=True),
        RandomCrop(size=300),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325])
    ])

    transform_train = Compose([
        ToTensor(),
        Resize(size=320, antialias=True),
        RandomCrop(size=300),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325])
    ])

    dataset = PlantNetDataset("plantnet_300K", "test", transform=transform_train,
                              tail_augmentations=tail_augmentations)
    # Count class frequencies and calculate weights
    class_counts = Counter(dataset.labels)
    total_samples = len(dataset)
    num_classes = len(class_counts)
    class_weights = {
        class_id: total_samples / (num_classes * max(count, 1))  # Use `max` to prevent division by zero
        for class_id, count in class_counts.items()
    }

    sample_weights = [class_weights[label] for label in dataset.labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Set tail classes for augmentation
    tail_classes = [class_id for class_id, count in class_counts.items() if count < 200]
    dataset.set_tail_classes(tail_classes)

    # Create a DataLoader with the sampler to oversample tail classes
    oversampled_dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        drop_last=True,
        shuffle=False,  # shuffle is handled by the sampler
        num_workers=4,
        sampler=sampler
    )

    # Create a dictionary to store the sample count for each class after oversampling
    class_distribution_after_sampling = {}

    # Iterate through the entire dataloader with WeightedRandomSampler

    progress_bar = tqdm(oversampled_dataloader, colour="blue")
    for iter, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description(
            f"iter: {iter}/{len(oversampled_dataloader)}"
        )
        for label in labels:
            label = int(label)
            if label not in class_distribution_after_sampling:
                class_distribution_after_sampling[label] = 0
            class_distribution_after_sampling[label] += 1

    class_distribution_after_sampling = dict(sorted(class_distribution_after_sampling.items()))

    class_ids = list(class_distribution_after_sampling.keys())
    counts = list(class_distribution_after_sampling.values())

    # Create the line chart
    plt.figure(figsize=(20, 10))

    sampled_class_ids = class_ids[::10]
    sampled_counts = counts[::10]

    plt.plot(sampled_class_ids, sampled_counts, marker='o', linestyle='-', color='teal', linewidth=2)

    plt.xlabel("Class ID", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Class Distribution After Oversampling", fontsize=16)

    plt.xticks(sampled_class_ids, fontsize=10, rotation=45)

    plt.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()
