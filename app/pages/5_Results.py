"""
Page: Results & Evaluation
Streamlit will auto-route this to /Results
"""

import streamlit as st
from components import render_header, render_sidebar
from views import results

st.set_page_config(
    page_title="Results & Evaluation",
    layout="wide",
)

render_header()
render_sidebar()

results.render()
