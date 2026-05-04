import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class CVAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, num_classes=10):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(784 + num_classes, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc2 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 784)

    def encode(self, x, y):
        x = x.view(x.size(0), -1)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
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
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
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
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss


def generate_digits(model, device, latent_dim, output_dir, epoch):
    model.eval()

    with torch.no_grad():
        z = torch.randn(100, latent_dim).to(device)
        labels = torch.arange(0, 10).repeat(10).to(device)

        samples = model.decode(z, labels)

        save_image(
            samples,
            os.path.join(output_dir, f"generated_epoch_{epoch}.png"),
            nrow=10
        )


def interpolate_digits(model, device, latent_dim, output_dir, start_digit=1, end_digit=8):
    model.eval()

    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        images = []

        for alpha in torch.linspace(0, 1, steps=10):
            z = (1 - alpha) * z1 + alpha * z2

            if alpha < 0.5:
                label = torch.tensor([start_digit]).to(device)
            else:
                label = torch.tensor([end_digit]).to(device)

            img = model.decode(z, label)
            images.append(img)

        images = torch.cat(images, dim=0)

        save_image(
            images,
            os.path.join(output_dir, f"interpolation_{start_digit}_to_{end_digit}.png"),
            nrow=10
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = (
        f"latent{args.latent_dim}_hidden{args.hidden_dim}_lr{args.lr}_batch{args.batch_size}"
    )

    output_dir = os.path.join("results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = CVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = os.path.join(output_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "recon_loss", "kl_loss"])

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss_sum = 0
        recon_loss_sum = 0
        kl_loss_sum = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            x_hat, mu, logvar = model(x, y)
            total_loss, recon_loss, kl_loss = loss_function(x_hat, x, mu, logvar)

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()

        avg_total_loss = total_loss_sum / len(train_loader.dataset)
        avg_recon_loss = recon_loss_sum / len(train_loader.dataset)
        avg_kl_loss = kl_loss_sum / len(train_loader.dataset)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Total: {avg_total_loss:.4f}, "
            f"Recon: {avg_recon_loss:.4f}, "
            f"KL: {avg_kl_loss:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_total_loss, avg_recon_loss, avg_kl_loss])

        generate_digits(model, device, args.latent_dim, output_dir, epoch)

    interpolate_digits(model, device, args.latent_dim, output_dir)

    model_path = os.path.join(output_dir, "cvae_mnist.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Experiment saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    train(args)