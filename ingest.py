import pandas as pd
import sys
import os

df = pd.read_csv(sys.argv[1])
df.to_csv("data_raw.csv", index = False)

os.system("python3 preprocess.py data_raw.csv")