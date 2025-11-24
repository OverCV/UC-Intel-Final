"""
Sidebar Component
Persistent sidebar showing session state, system resources, and theme settings
"""

from components.theme import render_theme_settings
from components.utils import check_gpu_available, get_memory_info
from state.workflow import (
    get_dataset_config,
    get_model_config,
    has_dataset_config,
    has_model_config,
    has_results,
    is_training_active,
)
import streamlit as st


def render_sidebar():
    """
    Persistent sidebar shown on all pages
    Shows session state details, system resources, and theme settings
    """
    st.sidebar.header("Session State")

    # Show configuration summaries
    if has_dataset_config():
        config = get_dataset_config()
        dataset_name = config.get("dataset_path", "Configured")
        st.sidebar.caption(f"📊 Dataset: {dataset_name}")
    else:
        st.sidebar.caption("📊 Dataset: Not configured")

    if has_model_config():
        config = get_model_config()
        model_name = config.get("architecture", "Configured")
        st.sidebar.caption(f"🧠 Model: {model_name}")
    else:
        st.sidebar.caption("🧠 Model: Not configured")

    # Show training status
    if is_training_active():
        st.sidebar.caption("🔴 Training in progress")
    elif has_results():
        st.sidebar.caption("✅ Training complete")
    else:
        st.sidebar.caption("⚙️ Training: Not started")

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
