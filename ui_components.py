import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix

def show_metrics_table(metrics):
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.3f}") 
