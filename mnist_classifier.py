import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def evaluate(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train_classifier():
    os.makedirs("classifier", exist_ok=True)

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False
    )

    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc = evaluate(model, test_loader)

        print(
            f"Epoch [{epoch}/{epochs}], "
            f"Loss: {total_loss:.4f}, "
            f"Test Accuracy: {test_acc:.4f}"
        )

    torch.save(
        model.state_dict(),
        "classifier/mnist_classifier.pth"
    )

    print("Saved classifier to classifier/mnist_classifier.pth")


if __name__ == "__main__":
    train_classifier()