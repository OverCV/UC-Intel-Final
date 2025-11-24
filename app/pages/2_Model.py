"""
Page: Model Configuration
Streamlit will auto-route this to /Model
"""

import streamlit as st
from components import render_header, render_sidebar
from views import model

st.set_page_config(
    page_title="Model Configuration",
    layout="wide",
)

render_header()
render_sidebar()

model.render()
