import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python preprocess.py <csv_path>")

    input_path = sys.argv[1]
    df = pd.read_csv(input_path)

    # Data Cleaning
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"]).copy()
    for col in ["artists", "album_name", "track_name", "track_genre"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in ["duration_ms", "tempo"]:
        if col in df.columns:
            df = df[df[col] > 0].copy()
    df = df.reset_index(drop=True)

    # Feature Transformation
    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(int)
    if "track_genre" in df.columns:
        top_genres = df["track_genre"].value_counts().head(15).index.tolist()
        genre_series = df["track_genre"].where(df["track_genre"].isin(top_genres), "other")
        genre_dummies = pd.get_dummies(genre_series, prefix="genre", dtype=int)
        df = pd.concat([df, genre_dummies], axis=1)
    feature_inputs = [c for c in ["danceability", "energy", "valence"] if c in df.columns]
    if feature_inputs:
        df["mood_score"] = df[feature_inputs].mean(axis=1)
    scale_cols = [c for c in [
        "popularity", "duration_ms", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "mood_score"
    ] if c in df.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in scale_cols], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)

    # Dimensionality Reduction
    reduced = df.drop(columns=[c for c in ["track_id", "artists", "album_name", "track_name"] if c in df.columns]).copy()
    pca_source_cols = [c for c in [
        "danceability_scaled", "energy_scaled", "speechiness_scaled",
        "acousticness_scaled", "instrumentalness_scaled", "liveness_scaled",
        "valence_scaled", "tempo_scaled", "loudness_scaled", "popularity_scaled"
    ] if c in reduced.columns]
    if len(pca_source_cols) >= 2:
        X = reduced[pca_source_cols].to_numpy(dtype=float)
        # fast randomized-size reduction via covariance eig on reduced feature space
        X_centered = X - X.mean(axis=0, keepdims=True)
        cov = (X_centered.T @ X_centered) / max(len(X_centered) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        components = eigvecs[:, order[:2]]
        projected = X_centered @ components
        reduced["pca1"] = projected[:, 0]
        reduced["pca2"] = projected[:, 1]

    # Discretization
    if "popularity" in reduced.columns:
        reduced["popularity_band"] = pd.cut(
            reduced["popularity"], bins=[-1, 33, 66, 100], labels=["low", "medium", "high"]
        )
    if "duration_ms" in reduced.columns:
        q1 = reduced["duration_ms"].quantile(0.33)
        q2 = reduced["duration_ms"].quantile(0.66)
        reduced["duration_band"] = pd.cut(
            reduced["duration_ms"], bins=[-1, q1, q2, reduced["duration_ms"].max()],
            labels=["short", "medium", "long"], include_lowest=True
        )
    if "tempo" in reduced.columns:
        reduced["tempo_band"] = pd.cut(
            reduced["tempo"], bins=[0, 90, 130, reduced["tempo"].max() + 1],
            labels=["slow", "medium", "fast"], include_lowest=True
        )

    output_path = "data_preprocessed.csv"
    reduced.to_csv(output_path, index=False)
    print(f"[preprocess] Output saved to {output_path} with shape {reduced.shape}")
    subprocess.run([sys.executable, "analytics.py", output_path], check=True)

if __name__ == "__main__":
    main()
