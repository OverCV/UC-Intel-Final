"""
Page: Dataset Configuration
Streamlit will auto-route this to /Dataset
"""

import streamlit as st
from components import render_header, render_sidebar
from views import dataset

# Page config
st.set_page_config(
    page_title="Dataset Configuration",
    layout="wide",
)

# Render persistent components
render_header()
render_sidebar()

# Render page content
dataset.render()
