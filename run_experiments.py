import subprocess


experiments = [
    {
        "latent_dim": 2,
        "hidden_dim": 400,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "beta": 1.0,
        "dropout": 0.0
    },
    {
        "latent_dim": 10,
        "hidden_dim": 256,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "beta": 1.0,
        "dropout": 0.0
    },
    {
        "latent_dim": 20,
        "hidden_dim": 400,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "beta": 1.0,
        "dropout": 0.0
    },
    {
        "latent_dim": 50,
        "hidden_dim": 400,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "beta": 1.0,
        "dropout": 0.0
    },
    {
        "latent_dim": 20,
        "hidden_dim": 400,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "beta": 0.5,
        "dropout": 0.0
    },
    {
        "latent_dim": 20,
        "hidden_dim": 400,
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "beta": 2.0,
        "dropout": 0.0
    },
    {
        "latent_dim": 20,
        "hidden_dim": 512,
        "lr": 0.0005,
        "batch_size": 128,
        "epochs": 10,
        "beta": 1.0,
        "dropout": 0.1
    }
]


for exp in experiments:
    command = [
        "python",
        "cvae_mnist.py",
        "--latent_dim", str(exp["latent_dim"]),
        "--hidden_dim", str(exp["hidden_dim"]),
        "--lr", str(exp["lr"]),
        "--batch_size", str(exp["batch_size"]),
        "--epochs", str(exp["epochs"]),
        "--beta", str(exp["beta"]),
        "--dropout", str(exp["dropout"])
    ]

    print("\nRunning experiment:")
    print(exp)

    subprocess.run(command)