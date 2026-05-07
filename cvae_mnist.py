import os
import csv
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split


class CVAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, num_classes=10, dropout=0.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(784 + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        x = x.view(x.size(0), -1)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        x = torch.cat([x, y_onehot], dim=1)

        h = self.encoder(x)
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

        x_hat = self.decoder(z)

        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)

        return x_hat, mu, logvar


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_function(x_hat, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def save_config(args, output_dir):
    config = {
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "beta": args.beta,
        "dropout": args.dropout,
        "seed": args.seed
    }

    config_path = os.path.join(output_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def run_one_epoch(model, dataloader, optimizer, device, beta=1.0, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0
    recon_loss_sum = 0
    kl_loss_sum = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            x_hat, mu, logvar = model(x, y)
            total_loss, recon_loss, kl_loss = loss_function(
                x_hat,
                x,
                mu,
                logvar,
                beta=beta
            )

            if train:
                total_loss.backward()
                optimizer.step()

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()

    dataset_size = len(dataloader.dataset)

    return {
        "total_loss": total_loss_sum / dataset_size,
        "recon_loss": recon_loss_sum / dataset_size,
        "kl_loss": kl_loss_sum / dataset_size
    }


def generate_digit_grid(model, device, latent_dim, output_dir, epoch):
    model.eval()

    with torch.no_grad():
        images = []

        for digit in range(10):
            z = torch.randn(10, latent_dim).to(device)
            labels = torch.full((10,), digit, dtype=torch.long).to(device)
            samples = model.decode(z, labels)
            images.append(samples)

        images = torch.cat(images, dim=0)

        save_image(
            images,
            os.path.join(output_dir, f"digit_grid_epoch_{epoch}.png"),
            nrow=10
        )


def save_reconstruction_grid(model, dataloader, device, output_dir, epoch):
    model.eval()

    with torch.no_grad():
        x, y = next(iter(dataloader))

        x = x[:10].to(device)
        y = y[:10].to(device)

        x_hat, _, _ = model(x, y)

        comparison = torch.cat([x.cpu(), x_hat.cpu()], dim=0)

        save_image(
            comparison,
            os.path.join(output_dir, f"reconstruction_epoch_{epoch}.png"),
            nrow=10
        )


def latent_interpolation_same_label(model, device, latent_dim, output_dir, digit=3):
    model.eval()

    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        images = []

        for alpha in torch.linspace(0, 1, steps=10).to(device):
            z = (1 - alpha) * z1 + alpha * z2
            label = torch.tensor([digit], dtype=torch.long).to(device)

            img = model.decode(z, label)
            images.append(img)

        images = torch.cat(images, dim=0)

        save_image(
            images,
            os.path.join(output_dir, f"latent_interpolation_digit_{digit}.png"),
            nrow=10
        )


def label_interpolation_fixed_latent(model, device, latent_dim, output_dir):
    model.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)

        images = []

        for digit in range(10):
            label = torch.tensor([digit], dtype=torch.long).to(device)
            img = model.decode(z, label)
            images.append(img)

        images = torch.cat(images, dim=0)

        save_image(
            images,
            os.path.join(output_dir, "same_latent_different_labels.png"),
            nrow=10
        )


def visualize_latent_space(model, dataloader, device, output_dir):
    if model.latent_dim != 2:
        print("Skipping latent space visualization because latent_dim is not 2.")
        return

    model.eval()

    all_mu = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            mu, _ = model.encode(x, y)

            all_mu.append(mu.cpu())
            all_labels.append(y.cpu())

    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        all_mu[:, 0],
        all_mu[:, 1],
        c=all_labels,
        s=5,
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("2D Latent Space Visualization")
    plt.savefig(os.path.join(output_dir, "latent_space.png"))
    plt.close()


def plot_losses(log_path, output_dir):
    epochs = []
    train_total = []
    val_total = []
    train_recon = []
    val_recon = []
    train_kl = []
    val_kl = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_total.append(float(row["train_total_loss"]))
            val_total.append(float(row["val_total_loss"]))
            train_recon.append(float(row["train_recon_loss"]))
            val_recon.append(float(row["val_recon_loss"]))
            train_kl.append(float(row["train_kl_loss"]))
            val_kl.append(float(row["val_kl_loss"]))

    plt.figure()
    plt.plot(epochs, train_total, label="Train Total Loss")
    plt.plot(epochs, val_total, label="Validation Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "total_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_recon, label="Train Reconstruction Loss")
    plt.plot(epochs, val_recon, label="Validation Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "reconstruction_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_kl, label="Train KL Loss")
    plt.plot(epochs, val_kl, label="Validation KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KL Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "kl_loss_curve.png"))
    plt.close()


def append_experiment_summary(args, output_dir, final_train, final_val):
    summary_path = os.path.join("results", "experiment_summary.csv")
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment_name",
                "latent_dim",
                "hidden_dim",
                "lr",
                "batch_size",
                "epochs",
                "beta",
                "dropout",
                "seed",
                "final_train_total_loss",
                "final_train_recon_loss",
                "final_train_kl_loss",
                "final_val_total_loss",
                "final_val_recon_loss",
                "final_val_kl_loss"
            ])

        writer.writerow([
            os.path.basename(output_dir),
            args.latent_dim,
            args.hidden_dim,
            args.lr,
            args.batch_size,
            args.epochs,
            args.beta,
            args.dropout,
            args.seed,
            final_train["total_loss"],
            final_train["recon_loss"],
            final_train["kl_loss"],
            final_val["total_loss"],
            final_val["recon_loss"],
            final_val["kl_loss"]
        ])


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using random seed: {args.seed}")

    experiment_name = (
        f"latent{args.latent_dim}_hidden{args.hidden_dim}_"
        f"lr{args.lr}_beta{args.beta}_dropout{args.dropout}_seed{args.seed}"
    )

    output_dir = os.path.join("results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    save_config(args, output_dir)

    transform = transforms.ToTensor()

    full_train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(0.9 * len(full_train_data))
    val_size = len(full_train_data) - train_size

    train_data, val_data = random_split(
        full_train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = CVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = os.path.join(output_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total_loss",
            "train_recon_loss",
            "train_kl_loss",
            "val_total_loss",
            "val_recon_loss",
            "val_kl_loss"
        ])

    best_val_loss = float("inf")
    final_train_losses = None
    final_val_losses = None

    for epoch in range(1, args.epochs + 1):
        train_losses = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=args.beta,
            train=True
        )

        val_losses = run_one_epoch(
            model,
            val_loader,
            optimizer,
            device,
            beta=args.beta,
            train=False
        )

        final_train_losses = train_losses
        final_val_losses = val_losses

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Total: {train_losses['total_loss']:.4f}, "
            f"Val Total: {val_losses['total_loss']:.4f}, "
            f"Train Recon: {train_losses['recon_loss']:.4f}, "
            f"Val Recon: {val_losses['recon_loss']:.4f}, "
            f"Train KL: {train_losses['kl_loss']:.4f}, "
            f"Val KL: {val_losses['kl_loss']:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_losses["total_loss"],
                train_losses["recon_loss"],
                train_losses["kl_loss"],
                val_losses["total_loss"],
                val_losses["recon_loss"],
                val_losses["kl_loss"]
            ])

        generate_digit_grid(model, device, args.latent_dim, output_dir, epoch)
        save_reconstruction_grid(model, val_loader, device, output_dir, epoch)

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pth")
            )

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "last_model.pth")
    )

    latent_interpolation_same_label(
        model,
        device,
        args.latent_dim,
        output_dir,
        digit=3
    )

    label_interpolation_fixed_latent(
        model,
        device,
        args.latent_dim,
        output_dir
    )

    visualize_latent_space(
        model,
        val_loader,
        device,
        output_dir
    )

    plot_losses(log_path, output_dir)

    append_experiment_summary(
        args,
        output_dir,
        final_train_losses,
        final_val_losses
    )

    print(f"Experiment saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)