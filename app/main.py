"""
Main Entry Point - Home & Setup Page
This is the landing page for the Streamlit multi-page app

Streamlit will automatically create navigation from pages/ directory
"""

import streamlit as st
from components import render_header, render_sidebar
from views import home

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Malware Classification - Home",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render persistent header and sidebar
render_header()
render_sidebar()

# Render home page content
home.render()
