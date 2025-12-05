import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(df, target_col, apply_ohe, normalization, categorical_cols, numerical_cols):

    clean_df = df.dropna().copy()

    y = clean_df[target_col]
    X = clean_df.drop(columns=[target_col])

    cat_cols = [c for c in categorical_cols if c in X.columns]
    num_cols = [c for c in numerical_cols if c in X.columns]

    if apply_ohe and cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if normalization != "None" and num_cols:
        scaler = StandardScaler() if normalization == "StandardScaler" else MinMaxScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y

