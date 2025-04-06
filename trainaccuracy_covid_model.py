# train_covid_model.py (Updated for K-Fold Cross-Validation)

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np

if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='dataset/train', transform=transform)

    # K-fold cross-validation setup
    k_folds = 5  # Adjust the number of folds as needed.
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Lists to store accuracies
    fold_accuracies = []

    # K-fold loop
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold + 1}')

        # Create data loaders for train and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler, num_workers=4)

        # Load model
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Training loop
        epochs = 10  # Adjust the number of epochs as needed.
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Fold {fold + 1} Accuracy: {accuracy}%')
        fold_accuracies.append(accuracy)

    # Average accuracy
    print(f'Average Accuracy: {np.mean(fold_accuracies)}%')