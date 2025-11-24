"""
Header Component
Persistent header section shown on all pages
"""

import streamlit as st

from components.styling import inject_custom_css
from components.utils import clear_session
from state.ui import get_past_sessions
from state.workflow import get_session_id


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
        session_id = get_session_id()
        if session_id:
            st.caption(f"Session: {session_id}")
        else:
            st.caption("No active session")

    with header_col2:
        past_sessions = get_past_sessions()

        if past_sessions:
            selected = st.selectbox(
                "Past Sessions",
                options=["Current"] + past_sessions,
                label_visibility="visible",
                help="Load a previous training session to review results"
            )
        else:
            st.caption("No past sessions")

    with header_col3:
        # Quick action buttons in header
        if st.button("New Session", width='stretch'):
            clear_session()
            st.rerun()

    st.divider()
