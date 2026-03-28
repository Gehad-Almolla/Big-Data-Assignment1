import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

def main():
    # Load transformed data
    df = pd.read_csv('transformed_data.csv')
    print(f"Loaded transformed data: {df.shape}")
    
    X = df.values
    
    n_clusters = 3 
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    unique, counts = np.unique(clusters, return_counts=True)
    
    with open('clusters.txt', 'w') as f:
        f.write("K-Means Clustering Results\n")
        f.write("=" * 30 + "\n")
        for cluster_id, count in zip(unique, counts):
            f.write(f"Cluster {cluster_id}: {count} samples ({count/len(clusters)*100:.1f}%)\n")
    
    print("Saved clusters.txt")

if __name__ == "__main__":
    main()