import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)

batch_size = 128
latent_dim = 20
num_classes = 10
epochs = 10
lr = 1e-3

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)


class CVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(784 + num_classes, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.fc2 = nn.Linear(latent_dim + num_classes, 400)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x, y):
        x = x.view(x.size(0), -1)
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        x = torch.cat([x, y_onehot], dim=1)

        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        z = torch.cat([z, y_onehot], dim=1)

        h = F.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(h))

        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)

        return x_hat, mu, logvar


def loss_function(x_hat, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(
        x_hat,
        x,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss


model = CVAE(latent_dim, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(x, y)
            loss = loss_function(x_hat, x, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

        generate_digits(epoch)

    torch.save(model.state_dict(), "cvae_mnist.pth")


def generate_digits(epoch):
    model.eval()

    with torch.no_grad():
        z = torch.randn(100, latent_dim).to(device)

        labels = torch.arange(0, 10).repeat(10).to(device)

        samples = model.decode(z, labels)
        save_image(
            samples,
            f"results/generated_epoch_{epoch}.png",
            nrow=10
        )


def interpolate_digits(start_digit=1, end_digit=8):
    model.eval()

    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        images = []

        for alpha in torch.linspace(0, 1, steps=10):
            z = (1 - alpha) * z1 + alpha * z2

            label = torch.tensor([start_digit]).to(device)
            img = model.decode(z, label)
            images.append(img)

        images = torch.cat(images, dim=0)

        save_image(
            images,
            f"results/interpolation_{start_digit}_to_{end_digit}.png",
            nrow=10
        )


if __name__ == "__main__":
    train()
    interpolate_digits(1, 8)