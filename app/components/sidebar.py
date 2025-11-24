"""
Sidebar Component
Persistent sidebar shown on all pages
"""

import streamlit as st

from components.theme import render_theme_settings
from components.utils import check_gpu_available, get_memory_info
from state.workflow import (
    has_dataset_config,
    has_model_config,
    has_training_config,
)


def render_sidebar():
    """
    Persistent sidebar shown on all pages
    Shows only state information, not navigation
    """
    st.sidebar.header("Session State")

    st.sidebar.divider()

    st.sidebar.header("Configuration Status")

    dataset_done = has_dataset_config()
    model_done = has_model_config()
    training_done = has_training_config()

    st.sidebar.markdown(f"{'✅' if dataset_done else '⬜'} Dataset configured")
    st.sidebar.markdown(f"{'✅' if model_done else '⬜'} Model configured")
    st.sidebar.markdown(f"{'✅' if training_done else '⬜'} Training configured")

    st.sidebar.divider()

    st.sidebar.header("System Resources")

    gpu_available = check_gpu_available()
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if gpu_available:
            st.metric("GPU", "Available")
        else:
            st.metric("GPU", "None")

    with col2:
        memory_info = get_memory_info()
        st.metric("Memory", memory_info)

    st.sidebar.divider()

    render_theme_settings()
