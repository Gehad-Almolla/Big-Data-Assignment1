import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

def main():
    df = pd.read_csv('data_processed.csv')
    df.shape
    df.columns.tolist()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    transformers = []
    
    #Standard Scaling for all numeric columns
    if numeric_cols:
        transformers.append(('standard', StandardScaler(), numeric_cols))
        
        #Min-Max Scaling specifically for popularity (0-100 range)
        if 'popularity' in numeric_cols:
            transformers.append(('minmax_pop', MinMaxScaler(), ['popularity']))
    
    #One-Hot Encoding for categorical columns
    if categorical_cols:
        transformers.append(('onehot', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), categorical_cols))
    
    #apply transformations
    print("Applying transformations...")
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_transformed = preprocessor.fit_transform(df)
    
    
    feature_names = []
    for name, transformer, cols in transformers:
        if name == 'onehot':
            encoded_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(cols)
            feature_names.extend(encoded_names)
        else:
            feature_names.extend(cols)
    
    print(f"Transformed to {len(feature_names)} features")
    
    # Dimensionality Reduction with PCA
    print("Applying PCA...")
    n_components = min(5, X_transformed.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_transformed)
    
    print(f"Reduced from {X_transformed.shape[1]} to {n_components} components")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df.to_csv('transformed_data.csv', index=False)
    print("Saved transformed_data.csv")

if __name__ == "__main__":
    main()
    