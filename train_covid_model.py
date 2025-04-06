# train_covid_model.py (Retraining on Full Dataset)

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the entire dataset
    dataset = datasets.ImageFolder(root='dataset/train', transform=transform)

    # Create data loader for training
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

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
    epochs = 10  # Adjust as needed
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

    # Save the trained model
    torch.save(model.state_dict(), 'covid_model.pth')
    print("Model retrained and saved as covid_model.pth")