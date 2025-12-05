import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocessing import process_data
from model_building import build_single_model, default_compare_model
from model_training import train_single_model, train_comparison
from ui_components import show_metrics_table, show_confusion, show_debug_attributes, show_comparison_table
from utils import parse_hidden_layer_sizes

def main():

    st.set_page_config(page_title="ML Toolkit", layout="wide")
    st.title("ML Toolkit: Classification & Evaluation")

    st.sidebar.title("Configuration")

    # 1. Load Dta
    df = load_data()

    if df is None:
        st.info("Upload or select a dataset to begin.")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.text(f"{df.shape[0]} rows × {df.shape[1]} columns")

    # Target col selection
    target = st.sidebar.selectbox("Target Column", df.columns)

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    cat_cols = [c for c in cat_cols if c != target]
    num_cols = [c for c in num_cols if c != target]

    # Preprocessing options
    st.sidebar.subheader("Preprocessing")
    apply_ohe = st.sidebar.checkbox("Apply One-Hot Encoding", True)
    normalization = st.sidebar.selectbox("Normalization", ["None", "StandardScaler", "Min-Max Scaler"], index=1)

    X_processed, y_processed = process_data(df, target, apply_ohe, normalization, cat_cols, num_cols)

    st.write(f"Processed shape: {X_processed.shape}, Target: {y_processed.shape}")

    # Train-test split
    seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=seed, stratify=y_processed
    )

    st.sidebar.subheader("Model Configuration")

    mode = st.sidebar.radio("Execution Mode", ["Single Model Training", "Compare Models"])

    model_options = ["Perceptron", "Multilayer Perceptron (MLP)", "Decision Tree"]

    if mode == "Single Model Training":

        model_choice = st.sidebar.selectbox("Choose Model", model_options)
        params = {"random_state": seed}

        if model_choice == "Decision Tree":
            params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
            params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

        elif model_choice == "Multilayer Perceptron (MLP)":
            params["hidden_layer_sizes"] = parse_hidden_layer_sizes(
                st.sidebar.text_input("Hidden Layers", "(100,)")
            )
            params["max_iter"] = st.sidebar.slider("Epochs", 100, 1000, 300, step=50)
            params["activation"] = st.sidebar.selectbox("Activation", ["relu", "tanh", "logistic"])

        elif model_choice == "Perceptron":
            params["eta0"] = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.1)
            params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 1000, 300, step=50)

        if st.button("🚀 Train Model", use_container_width=True):

            model = build_single_model(model_choice, params)
            y_pred, metrics = train_single_model(model, X_train, y_train, X_test, y_test)

            show_metrics_table(metrics)
            show_confusion(y_test, y_pred, labels=np.unique(y_processed))

            st.subheader("Model Internals")
            st.write(model.get_params())

            with st.expander("🔍 Debug Info"):
                show_debug_attributes(model)

    else:  # Comparison Mode
        selected = st.sidebar.multiselect("Select Models", model_options, default=model_options)

        if not selected:
            st.sidebar.warning("Select at least one model.")
            return

        if st.button("🚀 Train Models", use_container_width=True):
            models = {name: default_compare_model(name, seed) for name in selected}
            comparison_df = train_comparison(models, X_train, y_train, X_test, y_test)
            show_comparison_table(comparison_df)


if __name__ == "__main__":
    main()
