"""
Main Streamlit App Entry Point
Malware Classification with Deep Learning

BFS Level: Navigation & Page Routing
"""

import streamlit as st
# Import pages
from pages import (
    home,
    dataset,
    model,
    training,
    monitor,
    results,
    interpretability,
)


# Page configuration
st.set_page_config(
    page_title="Malware Classification",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Page routing dictionary
PAGES = {
    "Home & Setup": home,
    "Dataset Configuration": dataset,
    "Model Configuration": model,
    "Training Configuration": training,
    "Training Monitor": monitor,
    "Results & Evaluation": results,
    "Model Interpretability": interpretability,
}


def main():
    """Main app entry point with navigation"""

    # Sidebar navigation
    st.sidebar.title("Navigation")
    # page_selection = st.sidebar.radio(
    #     "Go to",
    #     list(PAGES.keys()),
    #     key="main_navigation"
    # )

    # Resource monitor in sidebar
    render_sidebar_resources()

    # Render selected page
    # page = PAGES[page_selection]
    # page.render()


def render_sidebar_resources():
    """Render resource monitoring in sidebar"""
    st.sidebar.divider()
    st.sidebar.header("System Resources")

    # Placeholder metrics - will implement actual detection later
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("GPU", "Checking...")
    with col2:
        st.metric("Memory", "Checking...")

    st.sidebar.divider()
    st.sidebar.header("Current Session")
    st.sidebar.text("Session: None")


if __name__ == "__main__":
    main()
