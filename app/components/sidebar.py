"""
Sidebar Component
Persistent sidebar showing system resources, theme settings, and session actions
"""

import streamlit as st

from components.theme import render_theme_settings
from components.utils import check_gpu_available, get_memory_info


def render_sidebar():
    """
    Persistent sidebar shown on all pages
    Shows system resources, theme settings, and delete session button
    """
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

    st.sidebar.divider()

    # Delete Session button at bottom
    if st.sidebar.button(
        "Delete Session",
        type="secondary",
        use_container_width=True,
        help="Delete current session and start fresh",
    ):
        _handle_delete_session()


def _handle_delete_session():
    """Handle delete session button click"""
    from state.persistence import delete_session
    from state.workflow import clear_workflow_state, get_session_id

    session_id = get_session_id()
    if session_id:
        delete_session(session_id)
        clear_workflow_state()
        st.rerun()
