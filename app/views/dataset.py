"""
Page 2: Dataset Configuration
BFS Level: Interface & Section Structure Only
"""

import streamlit as st


def render():
    """Main render function for Dataset page"""
    st.title("Dataset Configuration")

    # Section 1: Dataset Overview
    render_dataset_overview()

    # Section 2: Data Split
    render_data_split()

    # Section 3: Sample Visualization
    render_sample_visualization()

    # Section 4: Class Distribution
    render_class_distribution()

    # Section 5: Preprocessing Configuration
    render_preprocessing()

    # Section 6: Data Augmentation
    render_augmentation()

    # Section 7: Confirm Configuration
    render_confirmation()


def render_dataset_overview():
    """Section 1: Show combined dataset info"""
    st.header("Dataset Overview")
    st.info("Using dataset from: repo/")

    # TODO: Scan repo/malware and show actual stats
    st.metric("Total Samples", "Loading...")
    st.metric("Number of Classes", "Loading...")


def render_data_split():
    """Section 2: Configure train/val/test split"""
    st.header("Train/Validation/Test Split")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_pct = st.slider("Train %", 0, 100, 70)
    with col2:
        val_pct = st.slider("Val %", 0, 100, 15)
    with col3:
        test_pct = st.slider("Test %", 0, 100, 15)

    total = train_pct + val_pct + test_pct
    if total != 100:
        st.warning(f"Sum is {total}%, should be 100%")

    st.checkbox("Stratified Split", value=True)
    st.number_input("Random Seed", value=72)

    # TODO: Add pie chart visualization


def render_sample_visualization():
    """Section 3: Preview dataset images"""
    st.header("Dataset Preview")

    # TODO: Load actual families from dataset
    family = st.selectbox("Filter by Family", ["All"])

    st.info("Image grid - TO BE IMPLEMENTED")
    # TODO: Display 5-column grid of sample images


def render_class_distribution():
    """Section 4: Show class imbalance"""
    st.header("Class Distribution")

    st.info("Bar chart - TO BE IMPLEMENTED")
    # TODO: Plotly bar chart of samples per class
    # TODO: Calculate and show imbalance ratio


def render_preprocessing():
    """Section 5: Image preprocessing config"""
    st.header("Image Preprocessing")

    target_size = st.selectbox("Target Size", ["224x224", "256x256", "299x299"])
    normalization = st.radio("Normalization", ["[0,1]", "[-1,1]", "Z-score"])
    color_mode = st.radio("Color Mode", ["Grayscale", "RGB"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Preprocessing")
        st.info("Sample image - TO BE IMPLEMENTED")
    with col2:
        st.subheader("After Preprocessing")
        st.info("Processed sample - TO BE IMPLEMENTED")


def render_augmentation():
    """Section 6: Data augmentation config"""
    st.header("Data Augmentation")

    preset = st.radio("Augmentation Preset", ["None", "Light", "Moderate", "Heavy", "Custom"])

    if preset == "Custom":
        with st.expander("Configure Custom Augmentation"):
            st.checkbox("Rotation")
            st.checkbox("Horizontal Flip")
            st.checkbox("Vertical Flip")
            st.checkbox("Brightness")
            st.checkbox("Contrast")
            st.checkbox("Gaussian Noise")

    if st.button("Preview Augmentation"):
        st.info("Augmented samples grid - TO BE IMPLEMENTED")


def render_confirmation():
    """Section 7: Summary and navigation"""
    st.divider()
    st.success("Dataset Configuration Complete")

    # TODO: Display actual config as JSON
    st.json({
        "dataset_path": "repo/malware",
        "total_samples": "TBD",
        "split": {"train": 70, "val": 15, "test": 15}
    })

    if st.button("Next: Model Configuration", type="primary"):
        # TODO: Save config and navigate
        st.info("Navigation - TO BE IMPLEMENTED")
