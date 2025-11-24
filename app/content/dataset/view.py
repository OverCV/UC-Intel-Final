"""
Dataset Configuration - Main View Coordinator
Organizes content into tabs and delegates to tab-specific renderers
"""

from content.dataset.tabs import augmentation, distribution, overview, samples
from state.cache import get_dataset_info, set_dataset_info
import streamlit as st
from utils.dataset_utils import scan_dataset


def render():
    """Main render function for Dataset page"""
    st.title("Dataset Configuration")

    # Initialize dataset cache
    dataset_info = get_dataset_info()
    if not dataset_info:
        with st.spinner("Scanning dataset..."):
            dataset_info = scan_dataset()
            set_dataset_info(dataset_info)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 Overview & Split",
            "📊 Class Distribution",
            "🖼️ Samples & Preprocessing",
            "🔄 Augmentation",
        ]
    )

    with tab1:
        overview.render(dataset_info)

    with tab2:
        distribution.render(dataset_info)

    with tab3:
        samples.render(dataset_info)

    with tab4:
        augmentation.render(dataset_info)
