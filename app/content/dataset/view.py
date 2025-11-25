"""
Dataset Configuration - Main View Coordinator
Organizes content into tabs and delegates to tab-specific renderers
"""

from content.dataset.tabs import augmentation, distribution, overview, samples
from state.cache import get_dataset_info, set_dataset_info
from state.workflow import get_dataset_config, has_dataset_config
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

    # Auto-load saved configuration if it exists
    load_saved_configuration()

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


def load_saved_configuration():
    """Auto-load saved dataset configuration into session state"""
    if has_dataset_config():
        config = get_dataset_config()

        # Load selected classes
        if "selected_families" in config and "selected_classes" not in st.session_state:
            st.session_state.selected_classes = config["selected_families"]

        # Load split configuration
        if "split" in config:
            split_config = config["split"]
            if "train" in split_config and "train_split" not in st.session_state:
                # Calculate original slider values from percentages
                train_pct = split_config["train"]
                st.session_state.train_split = int(train_pct)

            if "stratified" in split_config and "stratified_split" not in st.session_state:
                st.session_state.stratified_split = split_config["stratified"]

            if "random_seed" in split_config and "random_seed" not in st.session_state:
                st.session_state.random_seed = split_config["random_seed"]

        # Load augmentation configuration
        if "augmentation" in config:
            aug_config = config["augmentation"]
            if "preset" in aug_config and "augmentation_preset" not in st.session_state:
                st.session_state.augmentation_preset = aug_config["preset"]

            # Load custom augmentation settings
            if "custom" in aug_config and aug_config["preset"] == "Custom":
                custom = aug_config["custom"]
                for key, value in custom.items():
                    session_key = f"aug_{key}"
                    if session_key not in st.session_state:
                        st.session_state[session_key] = value

        # Load imbalance handling
        if "imbalance_handling" in config:
            imb_config = config["imbalance_handling"]
            if "strategy" in imb_config and "imbalance_strategy" not in st.session_state:
                st.session_state.imbalance_strategy = imb_config["strategy"]

            if "class_weights" in imb_config and imb_config["class_weights"]:
                st.session_state.class_weights = imb_config["class_weights"]

            if "smote_ratio" in imb_config and imb_config["smote_ratio"] is not None:
                st.session_state.smote_ratio = imb_config["smote_ratio"]
