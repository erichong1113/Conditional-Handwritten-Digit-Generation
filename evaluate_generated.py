import os
import csv
import argparse

import torch
import torch.nn.functional as F

from cvae_mnist import CVAE
from mnist_classifier import MNISTClassifier


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


device = get_device()


def evaluate_generated_images(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")

    model = CVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(
        torch.load(args.cvae_path, map_location=device)
    )

    model.eval()

    classifier = MNISTClassifier().to(device)

    classifier.load_state_dict(
        torch.load(args.classifier_path, map_location=device)
    )

    classifier.eval()

    results = []

    with torch.no_grad():
        for digit in range(10):
            z = torch.randn(args.num_samples, args.latent_dim).to(device)
            labels = torch.full(
                (args.num_samples,),
                digit,
                dtype=torch.long
            ).to(device)

            generated = model.decode(z, labels)

            logits = classifier(generated)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            accuracy = (preds == labels).float().mean().item()
            avg_confidence = probs.max(dim=1)[0].mean().item()

            results.append({
                "digit": digit,
                "classifier_accuracy": accuracy,
                "avg_confidence": avg_confidence
            })

            print(
                f"Digit {digit}: "
                f"Classifier Accuracy = {accuracy:.4f}, "
                f"Average Confidence = {avg_confidence:.4f}"
            )

    output_path = os.path.join(args.output_dir, "generated_evaluation.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "digit",
                "classifier_accuracy",
                "avg_confidence"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cvae_path", type=str, required=True)
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="classifier/mnist_classifier.pth"
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()

    evaluate_generated_images(args)