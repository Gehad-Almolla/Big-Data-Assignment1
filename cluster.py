import sys
import pandas as pd
from sklearn.cluster import KMeans

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python cluster.py <csv_path>")
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)
    feature_cols = [
        "danceability_scaled", "energy_scaled", "acousticness_scaled",
        "valence_scaled", "tempo_scaled", "popularity_scaled"
    ]
    sample = df[feature_cols].sample(n=min(5000, len(df)), random_state=42)
    model = KMeans(n_clusters=3, random_state=42, n_init=3)
    model.fit(sample)
    labels = model.predict(df[feature_cols])
    counts = pd.Series(labels).value_counts().sort_index()
    with open("clusters.txt", "w", encoding="utf-8") as f:
        for cluster_id, count in counts.items():
            f.write(f"Cluster {cluster_id}: {count} samples\n")
    print("[cluster] Saved clusters.txt")

if __name__ == "__main__":
    main()
