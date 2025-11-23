"""
Page 1: Home & Setup
BFS Level: Interface & Section Structure Only
"""

import streamlit as st


def render():
    """Main render function for Home page"""
    st.title("Malware Classification with Deep Learning")

    # Section 1: Project Overview
    render_project_overview()

    # Section 2: Quick Start
    render_quick_start()

    # Section 3: Load Previous Session
    render_load_session()


def render_project_overview():
    """Section 1: What this app does"""
    st.header("Project Overview")

    with st.expander("What does this app do?"):
        st.markdown("""
        - Load combined malware dataset
        - Configure CNN architectures
        - Train models with different hyperparameters
        - Evaluate and visualize results
        """)

    # TODO: Add Mermaid workflow diagram


def render_quick_start():
    """Section 2: System status and new session"""
    st.header("Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "Combined")
    with col2:
        st.metric("Pre-trained Models", "5")
    with col3:
        st.metric("GPU Status", "Checking...")

    if st.button("Start New Training Session", type="primary"):
        # TODO: Create new session, navigate to dataset page
        st.info("Navigation to Dataset page - TO BE IMPLEMENTED")


def render_load_session():
    """Section 3: Resume previous experiments"""
    st.header("Load Previous Session")

    # TODO: Load actual sessions from storage
    sessions = ["No previous sessions"]

    selected_session = st.selectbox("Select Session", sessions)

    if st.button("Load Session"):
        st.info("Session loading - TO BE IMPLEMENTED")

    # TODO: Add session history table
