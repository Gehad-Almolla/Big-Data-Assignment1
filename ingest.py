import sys
import subprocess
import pandas as pd

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python ingest.py <dataset_path>")
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)
    raw_path = "data_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"[ingest] Loaded {input_path} with shape {df.shape}")
    print(f"[ingest] Saved raw copy to {raw_path}")
    subprocess.run([sys.executable, "preprocess.py", raw_path], check=True)

if __name__ == "__main__":
    main()
