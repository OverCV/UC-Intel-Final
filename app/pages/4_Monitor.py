"""
Page: Training Monitor
Streamlit will auto-route this to /Monitor
"""

import streamlit as st
from components import render_header, render_sidebar
from views import monitor

st.set_page_config(
    page_title="Training Monitor",
    layout="wide",
)

render_header()
render_sidebar()

monitor.render()
