import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python visualize.py <csv_path>")
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].hist(df["popularity"], bins=20, edgecolor="black")
    axes[0].set_title("Popularity Distribution")
    axes[0].set_xlabel("Popularity")
    axes[0].set_ylabel("Number of Tracks")

    axes[1].scatter(df["tempo"], df["energy"], alpha=0.2, s=8)
    axes[1].set_title("Tempo vs Energy")
    axes[1].set_xlabel("Tempo")
    axes[1].set_ylabel("Energy")

    corr_cols = ["popularity", "danceability", "energy", "loudness", "speechiness",
                 "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    corr = df[corr_cols].corr(numeric_only=True)
    sns.heatmap(corr, ax=axes[2], cmap="viridis", cbar=True)
    axes[2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("summary_plot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("[visualize] Saved summary_plot.png")
    subprocess.run([sys.executable, "cluster.py", input_path], check=True)

if __name__ == "__main__":
    main()
