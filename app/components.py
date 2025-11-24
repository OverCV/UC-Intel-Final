"""
Shared Components - Header, Navigation, and Sidebar
Persistent across all pages
"""

from constants import GET_CSS
import streamlit as st
import torch


def render_header():
    """
    Persistent header section shown on all pages
    Contains app title, session info, quick actions
    """
    # Apply custom CSS
    inject_custom_css()

    # Main header container
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

    with header_col1:
        st.markdown("### Malware Classification")
        if 'session_id' in st.session_state:
            st.caption(f"Session: {st.session_state.session_id}")
        else:
            st.caption("No active session")

    with header_col2:
        if 'past_sessions' not in st.session_state:
            st.session_state.past_sessions = []

        if st.session_state.past_sessions:
            selected = st.selectbox(
                "Past Sessions",
                options=["Current"] + st.session_state.past_sessions,
                label_visibility="visible",
                help="Load a previous training session to review results"
            )
        else:
            st.caption("No past sessions")

    with header_col3:
        # Quick action buttons in header
        if st.button("New Session", use_container_width=True):
            clear_session()
            st.rerun()

    st.divider()


def render_sidebar():
    """
    Persistent sidebar shown on all pages
    Shows only state information, not navigation
    """
    st.sidebar.header("Session State")

    if 'session_id' in st.session_state and st.session_state.session_id:
        st.sidebar.success(f"Session: {st.session_state.session_id}")
    else:
        st.sidebar.warning("No active session")

    st.sidebar.divider()

    st.sidebar.header("Configuration Status")

    dataset_done = bool(st.session_state.get('dataset_config'))
    model_done = bool(st.session_state.get('model_config'))
    training_done = bool(st.session_state.get('training_config'))

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

def render_theme_settings():
    """
    Theme customization settings
    """
    with st.sidebar.expander("Theme Settings"):
        st.markdown("**Color Palette**")

        # Initialize default colors if not set
        if "theme_primary" not in st.session_state:
            st.session_state.theme_primary = "#bdd373"
        if 'theme_secondary' not in st.session_state:
            st.session_state.theme_secondary = "#8fd7d7"
        if 'theme_background' not in st.session_state:
            st.session_state.theme_background = "#0e1117"

        # Color pickers (no key, manual state management)
        primary = st.color_picker(
            "Primary (buttons, links)",
            value=st.session_state.theme_primary
        )
        if primary != st.session_state.theme_primary:
            st.session_state.theme_primary = primary
            st.rerun()

        secondary = st.color_picker(
            "Secondary (headers)",
            value=st.session_state.theme_secondary
        )
        if secondary != st.session_state.theme_secondary:
            st.session_state.theme_secondary = secondary
            st.rerun()

        background = st.color_picker(
            "Background",
            value=st.session_state.theme_background
        )
        if background != st.session_state.theme_background:
            st.session_state.theme_background = background
            st.rerun()

        # Presets (softer colors from image)
        st.markdown("**Presets**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Soft Green", use_container_width=True):
                st.session_state.theme_primary = "#98c127"
                st.session_state.theme_secondary = "#bdd373"
                st.session_state.theme_background = "#0e1117"
                st.rerun()

        with col2:
            if st.button("Soft Blue", use_container_width=True):
                st.session_state.theme_primary = "#8fd7d7"
                st.session_state.theme_secondary = "#00b0be"
                st.session_state.theme_background = "#0e1117"
                st.rerun()

        col3, col4 = st.columns(2)
        with col3:
            if st.button("Soft Pink", use_container_width=True):
                st.session_state.theme_primary = "#f45f74"
                st.session_state.theme_secondary = "#ff8ca1"
                st.session_state.theme_background = "#0e1117"
                st.rerun()

        with col4:
            if st.button("Soft Orange", use_container_width=True):
                st.session_state.theme_primary = "#ffb255"
                st.session_state.theme_secondary = "#ffcd8e"
                st.session_state.theme_background = "#0e1117"
                st.rerun()


def inject_custom_css():
    """
    Inject custom CSS for professional styling
    """
    # Get theme colors from session state
    primary = st.session_state.get('theme_primary', '#bdd373')
    secondary = st.session_state.get('theme_secondary', '#8fd7d7')
    background = st.session_state.get('theme_background', '#0e1117')

    st.markdown(GET_CSS(primary, secondary, background), unsafe_allow_html=True)


def check_gpu_available() -> bool:
    """Check if GPU is available"""
    try:
        return torch.cuda.is_available()
    except:
        return False


def get_memory_info() -> str:
    """Get memory info string"""
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            return f"{allocated:.1f}/{total:.1f}GB"
        else:
            return "N/A"
    except:
        return "N/A"


def clear_session():
    """Clear current session state"""
    keys_to_clear = [
        "session_id",
        "dataset_config",
        "model_config",
        "training_config",
        "training_active",
        "results",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
