
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
dataset = ImageFolder("Spectrograms", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# Training loop (simplified)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in dataloader:
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
