"""
Dataset Tab 4: Augmentation & Configuration
Data augmentation settings and final configuration save
"""

from pathlib import Path

from state.cache import get_dataset_info, get_train_split, get_val_split
from state.workflow import has_dataset_config, save_dataset_config
import streamlit as st
from utils.dataset_utils import DATASET_ROOT, calculate_split_percentages


def render(dataset_info):
    """Render augmentation settings and configuration save"""
    render_augmentation_config()
    st.divider()
    render_configuration_summary()


def render_augmentation_config():
    """Data augmentation preset selection and custom config"""
    st.subheader("Data Augmentation")

    preset = st.radio("Augmentation Preset",
                       ["None", "Light", "Moderate", "Heavy", "Custom"],
                       horizontal=True)

    preset_info = {
        "None": "No augmentation applied",
        "Light": "Horizontal flip + Small rotation (±10°)",
        "Moderate": "Horizontal/Vertical flip + Rotation (±20°) + Brightness (±10%)",
        "Heavy": "All flips + Rotation (±30°) + Brightness/Contrast (±20%) + Noise"
    }

    if preset != "Custom":
        st.info(preset_info.get(preset, ""))

    if preset == "Custom":
        st.markdown("**Custom Augmentation Configuration**")

        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Horizontal Flip", value=True)
            st.checkbox("Vertical Flip", value=False)
            rotation = st.checkbox("Random Rotation", value=True)
            if rotation:
                st.slider("Rotation Range (degrees)", 0, 180, 15)

        with col2:
            brightness = st.checkbox("Brightness Adjustment", value=True)
            if brightness:
                st.slider("Brightness Range (%)", 0, 50, 10)

            contrast = st.checkbox("Contrast Adjustment", value=False)
            if contrast:
                st.slider("Contrast Range (%)", 0, 50, 10)

            st.checkbox("Gaussian Noise", value=False)


def render_configuration_summary():
    """Final configuration summary and save button"""
    st.subheader("Configuration Summary")

    train_pct = get_train_split()
    val_of_remaining = get_val_split()
    train_final, val_final, test_final = calculate_split_percentages(train_pct, val_of_remaining)

    dataset_info = get_dataset_info()
    config = {
        "dataset_path": str(DATASET_ROOT.relative_to(Path.cwd())),
        "total_samples": dataset_info['total_train'] + dataset_info['total_val'],
        "num_classes": len(dataset_info['classes']),
        "split": {
            "train": round(train_final, 1),
            "val": round(val_final, 1),
            "test": round(test_final, 1)
        }
    }

    st.json(config)

    _, col2, _ = st.columns([1, 1, 1])

    with col2:
        if st.button("Save Configuration", type="primary", use_container_width=True):
            save_dataset_config(config)
            st.success("Configuration saved!")
            st.balloons()

    if has_dataset_config():
        st.info("Configuration saved. Navigate to **Model** page to continue.")
