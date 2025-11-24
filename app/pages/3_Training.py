"""
Page: Training Configuration
Streamlit will auto-route this to /Training
"""

import streamlit as st
from components import render_header, render_sidebar
from views import training

st.set_page_config(
    page_title="Training Configuration",
    layout="wide",
)

render_header()
render_sidebar()

training.render()
