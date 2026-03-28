import sys
import subprocess
import pandas as pd

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python analytics.py <csv_path>")
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)

    avg_popularity = df["popularity"].mean()
    top_genre = df["track_genre"].mode().iat[0]
    avg_tempo = df["tempo"].mean()
    explicit_ratio = df["explicit"].mean() * 100
    corr = df["energy"].corr(df["loudness"])

    insights = {
        "insight1.txt": f"The dataset contains {len(df):,} unique tracks after preprocessing. The average popularity is {avg_popularity:.2f}, and the most frequent genre is '{top_genre}'.",
        "insight2.txt": f"The average tempo is {avg_tempo:.2f} BPM. Explicit tracks represent {explicit_ratio:.2f}% of the dataset, which means most songs are non-explicit.",
        "insight3.txt": f"Energy and loudness have a correlation of {corr:.3f}, indicating that louder songs also tend to be more energetic."
    }

    for filename, content in insights.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    print("[analytics] Generated insights.")
    subprocess.run([sys.executable, "visualize.py", input_path], check=True)

if __name__ == "__main__":
    main()
