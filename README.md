# Conditional Handwritten Digit Generation using CVAE

## Overview

This project implements a Conditional Variational Autoencoder (CVAE) for handwritten digit generation using the MNIST dataset.

The goal is to train a generative model from scratch that can generate new handwritten digits. Since this is a conditional model, the model can take a digit label from 0 to 9 and generate an image that looks like that digit.

For example, if the input label is 5, the model should generate something that looks like a handwritten 5.

The original idea was a simple CVAE, but I expanded it to include hyperparameter tuning, validation tracking, reconstruction comparison, latent space visualization, interpolation, and classifier-based evaluation.

## Dataset

This project uses the MNIST dataset.

MNIST contains grayscale images of handwritten digits from 0 to 9. Each image is 28 by 28 pixels. The dataset is small and clean, which makes it a good fit for training a lightweight generative model from scratch.

The dataset is downloaded automatically through `torchvision`.

## Model

The main model is a Conditional Variational Autoencoder.

A regular VAE learns to compress an image into a latent space and then reconstruct it from that latent representation. A conditional VAE does the same thing, but it also uses the digit label as extra input.

In this project, the encoder takes both the image and its label. It maps them into a latent representation using two vectors: `mu` and `logvar`. The model then samples from this latent distribution using the reparameterization trick.

The decoder takes the sampled latent vector and a digit label, then generates an image. This makes the generation controllable because the model is not just generating random digits. It is generating digits based on the label it receives.

## Loss Function

The loss function has two parts.

The first part is reconstruction loss. This measures how close the reconstructed image is to the original input image.

The second part is KL divergence. This pushes the latent space to be closer to a normal distribution, which makes it possible to sample from the latent space later.

The full loss is:

```text
total loss = reconstruction loss + beta * KL loss
```

I added the `beta` parameter so I can control how much the KL loss matters. This allows me to compare how different KL weights affect the generated images.

## Extra Criteria

The main extra criterion is hyperparameter tuning. I trained multiple CVAE models with different settings and compared their results.

The experiments compare:

```text
latent dimension
beta value
generation quality
reconstruction quality
validation loss
classifier-based accuracy
```

I also added several analysis outputs, including generated image grids, reconstruction images, interpolation results, loss curves, latent space visualization, and generated image evaluation.

## Files

### `cvae_mnist.py`

This is the main training file. It defines the CVAE model, trains it, saves generated images, saves reconstruction images, logs losses, saves model checkpoints, and creates loss curve plots.

### `run_experiments.py`

This runs multiple CVAE experiments with different hyperparameters.

### `analyze_results.py`

This reads the experiment summary file and ranks the experiments based on validation loss.

### `mnist_classifier.py`

This trains a simple MNIST classifier. The classifier is later used to evaluate whether the generated images match their intended labels.

### `evaluate_generated.py`

This loads a trained CVAE and the trained MNIST classifier, generates images for each digit label, and checks whether the classifier predicts the correct digit.

## How to Install

Create and activate a virtual environment first.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages.

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision matplotlib pandas
```

## How to Run

### 1. Run all CVAE experiments

```bash
python run_experiments.py
```

This trains several CVAE models with different hyperparameters. Each experiment saves its own results inside the `results/` folder.

The current experiments include:

```text
latent_dim = 2, beta = 1.0
latent_dim = 20, beta = 1.0
latent_dim = 20, beta = 0.5
latent_dim = 10, beta = 1.0
latent_dim = 50, beta = 1.0
```

Each experiment runs for 50 epochs.

### 2. Analyze experiment results

```bash
python analyze_results.py
```

This creates a ranked results file:

```text
results/ranked_experiments.csv
```

This file ranks the experiments based on validation loss.

### 3. Train the MNIST classifier

```bash
python mnist_classifier.py
```

This trains a small classifier on real MNIST images and saves it to:

```text
classifier/mnist_classifier.pth
```

### 4. Evaluate generated images

Evaluate the baseline model:

```bash
python evaluate_generated.py \
  --cvae_path results/latent20_hidden400_lr0.001_beta1.0_dropout0.0_seed42/best_model.pth \
  --output_dir results/latent20_hidden400_lr0.001_beta1.0_dropout0.0_seed42 \
  --latent_dim 20 \
  --hidden_dim 400 \
  --dropout 0.0
```

Evaluate the beta 0.5 model:

```bash
python evaluate_generated.py \
  --cvae_path results/latent20_hidden400_lr0.001_beta0.5_dropout0.0_seed42/best_model.pth \
  --output_dir results/latent20_hidden400_lr0.001_beta0.5_dropout0.0_seed42 \
  --latent_dim 20 \
  --hidden_dim 400 \
  --dropout 0.0
```

The evaluation result is saved as `generated_evaluation.csv` inside the corresponding experiment folder.

## Output

Each experiment creates a folder inside `results/`.

Example:

```text
results/latent20_hidden400_lr0.001_beta1.0_dropout0.0_seed42/
```

Inside each experiment folder, the code saves files such as:

```text
config.json
training_log.csv
best_model.pth
last_model.pth
digit_grid_epoch_50.png
reconstruction_epoch_50.png
latent_interpolation_digit_3.png
same_latent_different_labels.png
total_loss_curve.png
reconstruction_loss_curve.png
kl_loss_curve.png
generated_evaluation.csv
```

For the `latent_dim = 2` experiment, the code also saves:

```text
latent_space.png
```

## Results Explanation

The generated digit grids show that the CVAE is able to create digits based on the labels I give it. The images are not perfectly sharp, but that is normal for a VAE. The main point is that the outputs look like handwritten digits and mostly match the target labels.

The reconstruction images compare the original MNIST digits with the model’s reconstructed versions. This helps show whether the encoder and decoder are actually learning useful information from the images.

The interpolation image shows how the output changes when the model moves between two points in the latent space. If the change looks smooth, it suggests that the model learned a somewhat smooth latent space.

The `same_latent_different_labels.png` image is also useful because it keeps the same latent vector but changes the label. This shows that the label is actually affecting the output, not just being ignored by the model.

The loss curves show how the training and validation losses changed over time. I used them to check whether the model was still improving and whether the validation loss followed a reasonable pattern.

The classifier-based evaluation gives a simple numerical way to check the generated images. If the CVAE is asked to generate a 5, and the classifier also predicts it as a 5, then that is counted as a successful generation.

## Hyperparameter Tuning

Instead of training only one model, I tested a few different settings to see how they affected the results.

The main things I changed were the latent dimension and the beta value. The latent dimension controls how much information the model can store in the latent space. A smaller latent dimension is easier to visualize, but it may not have enough space to capture all the details. A larger latent dimension gives the model more room, but it can also make the latent space harder to understand.

The beta value controls how much weight is placed on the KL divergence part of the loss. A lower beta makes the model focus more on reconstruction, which can sometimes make the digits look clearer. A higher beta puts more pressure on the latent space to follow a normal distribution.

By comparing these experiments, I could see how different settings affected the loss values, reconstruction quality, and generated images.

## Difficulties

One difficulty I faced was managing all the experiment outputs after adding hyperparameter tuning. Since each experiment saves generated images, reconstruction images, loss curves, config files, and CSV logs, the `results/` folder can quickly become messy. At first, it was hard to keep track of which result came from which setting.

To solve this, I made each experiment save into its own folder named with the main hyperparameters, such as latent dimension, beta value, dropout, and seed. I also saved a `config.json` file inside each experiment folder, so I could easily check what settings were used. This made it much easier to compare experiments later and avoid mixing up results.

Another challenge was deciding how to evaluate the generated images. Looking at the images by eye is useful, but it is also subjective. To make the evaluation more concrete, I trained a small MNIST classifier and used it to check whether the generated images matched their intended labels. This gave me a simple numerical way to compare how well the conditional generation worked.
