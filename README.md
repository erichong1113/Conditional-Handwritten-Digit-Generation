# Conditional VAE on MNIST

## Overview

This project implements a Conditional Variational Autoencoder (CVAE) on the MNIST dataset. The goal is to generate handwritten digits while controlling which digit is generated using labels from 0 to 9.

Instead of generating completely random digits, the model can take a label as input and generate a specific digit, like generating a 3 when given label 3.

## Dataset

I used the MNIST dataset, which contains grayscale images of handwritten digits from 0 to 9. Each image is 28x28 pixels.

## Model

The model is a Conditional Variational Autoencoder.

The encoder takes an image and its digit label, then maps them into a latent space. The decoder takes a latent vector and a digit label, then generates an image.

The loss function combines reconstruction loss and KL divergence.

## How to Run

Install the required packages:

```bash
pip install torch torchvision
