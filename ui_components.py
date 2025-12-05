import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix

def show_metrics_table(metrics):
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.3f}")

def show_confusion(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    st.pyplot(fig)

def show_debug_attributes(model):
    st.write("### Model Internals")

    attributes = {}
    for attr in ["n_iter_", "coef_", "intercepts_", "feature_importances_", "loss_curve_"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, np.ndarray):
                val = val.round(4).tolist()
            attributes[attr] = val

    if attributes:
        st.json(attributes)
    else:
        st.write("No debug attributes available.")

def show_comparison_table(df):
    st.subheader("Model Comparison")
    st.dataframe(df.style.format("{:.3f}"))
    st.bar_chart(df["Accuracy"])

