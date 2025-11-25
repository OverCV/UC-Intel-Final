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
    augmentation_config = render_augmentation_config()
    st.divider()
    render_configuration_summary(dataset_info, augmentation_config)


def render_augmentation_config():
    """Data augmentation preset selection and custom config"""
    st.subheader("Data Augmentation")

    st.info("🔄 Augmentation is applied during training, not during dataset preparation")

    preset = st.radio("Augmentation Preset",
                       ["None", "Light", "Moderate", "Heavy", "Custom"],
                       horizontal=True,
                       key="augmentation_preset")

    preset_info = {
        "None": "No augmentation applied",
        "Light": "Horizontal flip + Small rotation (±10°)",
        "Moderate": "Horizontal/Vertical flip + Rotation (±20°) + Brightness (±10%)",
        "Heavy": "All flips + Rotation (±30°) + Brightness/Contrast (±20%) + Noise"
    }

    if preset != "Custom":
        st.info(preset_info.get(preset, ""))

    # Store augmentation config
    augmentation_config = {"preset": preset}

    if preset == "Custom":
        st.markdown("**Custom Augmentation Configuration**")

        col1, col2 = st.columns(2)
        with col1:
            h_flip = st.checkbox("Horizontal Flip", value=True, key="aug_h_flip")
            v_flip = st.checkbox("Vertical Flip", value=False, key="aug_v_flip")
            rotation = st.checkbox("Random Rotation", value=True, key="aug_rotation")
            rotation_range = 15
            if rotation:
                rotation_range = st.slider("Rotation Range (degrees)", 0, 180, 15, key="aug_rotation_range")

        with col2:
            brightness = st.checkbox("Brightness Adjustment", value=True, key="aug_brightness")
            brightness_range = 10
            if brightness:
                brightness_range = st.slider("Brightness Range (%)", 0, 50, 10, key="aug_brightness_range")

            contrast = st.checkbox("Contrast Adjustment", value=False, key="aug_contrast")
            contrast_range = 10
            if contrast:
                contrast_range = st.slider("Contrast Range (%)", 0, 50, 10, key="aug_contrast_range")

            noise = st.checkbox("Gaussian Noise", value=False, key="aug_noise")

        # Build custom config
        augmentation_config["custom"] = {
            "horizontal_flip": h_flip,
            "vertical_flip": v_flip,
            "rotation": rotation,
            "rotation_range": rotation_range if rotation else 0,
            "brightness": brightness,
            "brightness_range": brightness_range if brightness else 0,
            "contrast": contrast,
            "contrast_range": contrast_range if contrast else 0,
            "gaussian_noise": noise
        }

    return augmentation_config


def render_configuration_summary(dataset_info, augmentation_config):
    """Final configuration summary and save button"""
    st.subheader("Configuration Summary")

    train_pct = get_train_split()
    val_of_remaining = get_val_split()
    train_final, val_final, test_final = calculate_split_percentages(train_pct, val_of_remaining)

    # Use selected classes from session state, or all classes if none selected
    if "selected_classes" in st.session_state and st.session_state.selected_classes:
        selected_families = sorted(st.session_state.selected_classes)
    else:
        selected_families = sorted(dataset_info['classes'])

    # Calculate totals for selected classes only
    total_train = sum(dataset_info['train_samples'].get(c, 0) for c in selected_families)
    total_val = sum(dataset_info['val_samples'].get(c, 0) for c in selected_families)

    # Get imbalance handling strategy
    imbalance_strategy = st.session_state.get("imbalance_strategy", "Auto Class Weights (Recommended)")
    class_weights = st.session_state.get("class_weights", None) if imbalance_strategy == "Manual Class Weights" else None

    config = {
        "dataset_path": str(DATASET_ROOT.relative_to(Path.cwd())),
        "total_samples": total_train + total_val,
        "num_classes": len(selected_families),
        "selected_families": selected_families,  # CRITICAL: This was missing!
        "split": {
            "train": round(train_final, 1),
            "val": round(val_final, 1),
            "test": round(test_final, 1),
            "stratified": st.session_state.get("stratified_split", True),
            "random_seed": st.session_state.get("random_seed", 73)
        },
        "augmentation": augmentation_config,  # Save augmentation settings
        "preprocessing": {
            "target_size": (224, 224),
            "normalization": "[0,1]",
            "color_mode": "RGB"
        },
        "imbalance_handling": {
            "strategy": imbalance_strategy,
            "class_weights": class_weights,
            "smote_ratio": st.session_state.get("smote_ratio", 0.5) if imbalance_strategy == "Oversampling (SMOTE)" else None
        }
    }

    # Display config with better formatting
    with st.expander("View Full Configuration", expanded=True):
        st.json(config)

    # Show key info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected Classes", len(selected_families))
    with col2:
        st.metric("Total Samples", config["total_samples"])
    with col3:
        st.metric("Augmentation", augmentation_config["preset"])

    _, col2, _ = st.columns([1, 1, 1])

    with col2:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            save_dataset_config(config)
            st.success("✅ Dataset configuration saved successfully!")
            st.balloons()

    if has_dataset_config():
        st.info("✅ Configuration saved. Navigate to **Model** page to continue.")
