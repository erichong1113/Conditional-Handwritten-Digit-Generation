import os
import pandas as pd

summary_path = os.path.join("results", "experiment_summary.csv")

if not os.path.exists(summary_path):
    print("No experiment_summary.csv found. Run some experiments first.")
else:
    df = pd.read_csv(summary_path)

    df_sorted = df.sort_values(by="final_val_total_loss")

    print("\nExperiment Ranking by Validation Loss:\n")
    print(df_sorted[[
        "experiment_name",
        "latent_dim",
        "hidden_dim",
        "lr",
        "beta",
        "dropout",
        "final_val_total_loss",
        "final_val_recon_loss",
        "final_val_kl_loss"
    ]])

    output_path = os.path.join("results", "ranked_experiments.csv")
    df_sorted.to_csv(output_path, index=False)

    print(f"\nSaved ranked results to {output_path}")