"""
Dataset Tab 3: Samples & Preprocessing
Sample image viewer and preprocessing preview
"""

import random

from PIL import Image
import streamlit as st


def render(dataset_info):
    """Render sample viewer and preprocessing preview"""
    render_sample_viewer(dataset_info)
    st.divider()
    render_preprocessing_preview(dataset_info)


def render_sample_viewer(dataset_info):
    """Sample image browser with filtering"""
    st.subheader("Dataset Samples")

    if not dataset_info["sample_paths"]:
        st.warning("No sample images found")
        return

    selected_class = st.selectbox(
        "Filter by Malware Family",
        options=["All"] + dataset_info["classes"],
    )

    if selected_class == "All":
        all_samples = []
        for paths in dataset_info["sample_paths"].values():
            all_samples.extend(paths[:2])
        sample_paths = random.sample(all_samples, min(10, len(all_samples)))
    else:
        sample_paths = dataset_info["sample_paths"].get(selected_class, [])[:8]

    if not sample_paths:
        st.info("No samples available for this class")
        return

    cols = st.columns(5)
    for idx, img_path in enumerate(sample_paths):
        with cols[idx % 5]:
            try:
                img = Image.open(img_path)
                st.image(img, width="stretch")
                st.caption(f"{img.size[0]}x{img.size[1]}")
            except Exception as e:
                st.error(f"Error: {img_path.name}")


def render_preprocessing_preview(dataset_info):
    """Show before/after preprocessing"""
    st.subheader("Preprocessing Preview")

    col1, col2, col3 = st.columns(3)
    with col1:
        target_size = st.selectbox(
            "Target Size", ["224x224", "256x256", "299x299", "512x512"], index=0
        )
    with col2:
        normalization = st.radio(
            "Normalization", ["[0,1] Scale", "[-1,1] Scale", "ImageNet Mean/Std"]
        )
    with col3:
        color_mode = st.radio("Color Mode", ["RGB", "Grayscale"])

    if dataset_info["sample_paths"]:
        sample_class = list(dataset_info["sample_paths"].keys())[0]
        sample_path = dataset_info["sample_paths"][sample_class][0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            try:
                original = Image.open(sample_path)
                st.image(original, use_container_width=True)
                st.caption(f"Size: {original.size[0]}x{original.size[1]}")
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col2:
            st.markdown("**After Preprocessing**")
            try:
                size = int(target_size.split("x")[0])
                processed = Image.open(sample_path)
                processed = processed.resize((size, size), Image.Resampling.LANCZOS)

                if color_mode == "Grayscale":
                    processed = processed.convert("L")

                st.image(processed, use_container_width=True)
                st.caption(f"Size: {size}x{size}, Mode: {color_mode}")
            except Exception as e:
                st.error(f"Error processing: {e}")
