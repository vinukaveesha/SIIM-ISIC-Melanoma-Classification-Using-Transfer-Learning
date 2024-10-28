import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming `images` and `target_reshaped` are numpy arrays
# Split data into train and validation sets
images, x_val, target_reshaped, y_val = train_test_split(
    images, target_reshaped, test_size=0.2, random_state=42, stratify=target_reshaped
)

# Convert images and labels to PyTorch tensors
images_tensor = torch.tensor(images).float()
target_tensor = torch.tensor(target_reshaped).long()
x_val_tensor = torch.tensor(x_val).float()
y_val_tensor = torch.tensor(y_val).long()

# Data augmentation equivalent in PyTorch
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.7, 1.3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

val_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)


# Custom dataset to apply transforms
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target


# Create datasets
train_dataset = CustomDataset(images_tensor, target_tensor, transform=train_transform)
val_dataset = CustomDataset(x_val_tensor, y_val_tensor, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
