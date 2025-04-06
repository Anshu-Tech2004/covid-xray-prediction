# train_data_check.py (Updated)

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Define transformations (MUST MATCH TRAINING TRANSFORMATIONS EXACTLY)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training dataset
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Visualize a few images to check preprocessing
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()

    # Undo normalization for visualization (optional)
    images = images * np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    images = np.clip(images, 0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(f'Label: {labels[idx].item()}')
    plt.show()

    # Print min/max values of tensors to check normalization
    images_tensor = torch.from_numpy(images) # Convert to tensor.
    print(f"Min: {torch.min(images_tensor)}, Max: {torch.max(images_tensor)}")