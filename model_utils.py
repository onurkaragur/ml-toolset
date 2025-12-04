import ast
import streamlit as st

def parse_hidden_layer_sizes(value: str):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, tuple):
            return parsed
        if isinstance(parsed, list):
            return tuple(parsed)
        if isinstance(parsed, int):
            return (parsed,)
    except:
        pass
    st.sidebar.warning("Invalid hidden layer format. Using default (100,).")
    return (100,)