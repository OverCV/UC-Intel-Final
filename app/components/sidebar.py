"""
Sidebar Component
Persistent sidebar showing system resources, theme settings, and session actions
"""

from components.theme import render_theme_settings
from components.tooltips import SIDEBAR_TOOLTIPS
from components.utils import (
    get_compute_device,
    get_cpu_info,
    get_gpu_memory,
    get_platform_info,
    get_system_memory,
)
import streamlit as st


def render_sidebar():
    """
    Persistent sidebar shown on all pages
    Shows system resources, theme settings, and delete session button
    """
    st.sidebar.header("System Resources", help=SIDEBAR_TOOLTIPS["system_resources"])

    device = get_compute_device()
    cpu_info = get_cpu_info()

    # Row 1: Compute device
    st.sidebar.caption(f"**{device['type']}** · {device['name']}")

    # Row 2: Memory metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        gpu_mem = get_gpu_memory()
        st.metric("GPU Mem", gpu_mem, help=SIDEBAR_TOOLTIPS["gpu_memory"])
    with col2:
        sys_mem = get_system_memory()
        st.metric("RAM", sys_mem, help=SIDEBAR_TOOLTIPS["system_ram"])

    # Row 3: CPU and platform
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("CPU", f"{cpu_info['cores']}C/{cpu_info['threads']}T", help=SIDEBAR_TOOLTIPS["cpu_info"])
    with col2:
        platform_str = get_platform_info()
        st.metric("Platform", platform_str, help=SIDEBAR_TOOLTIPS["platform"])

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
