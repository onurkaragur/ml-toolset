import os 
import pandas as pd
import streamlit as st
from config import DEMO_DATASETS

def load_data():
    st.sidebar.subheader("Data Source")

    choice = st.sidebar.radio("How to load data:", ["Upload Your Own CSV", "Use Demo Dataset"])

    if choice == "Upload Your Own CSV":
        file = st.sidebar.file_uploader("Upload Your CSV", type=["csv"])
        if file:
            try:
                return pd.read_csv(file)
            except Exception as exc:
                st.error(f"Could not read the file {exc}")
        return None
    
    demo_choice = st.sidebar.selectbox("Select dataset: ", DEMO_DATASETS.keys())
    path = DEMO_DATASETS.get(demo_choice)

    if not os.path.exists(path):
        st.error(f"Dataset not found {path}")
        return None
    
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return None