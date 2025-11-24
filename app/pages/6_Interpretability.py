"""
Page: Model Interpretability
Streamlit will auto-route this to /Interpretability
"""

import streamlit as st
from components import render_header, render_sidebar
from views import interpretability

st.set_page_config(
    page_title="Model Interpretability",
    layout="wide",
)

render_header()
render_sidebar()

interpretability.render()
