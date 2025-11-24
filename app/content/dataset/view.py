"""
Page 2: Dataset Configuration
"""

import streamlit as st
from pathlib import Path
import plotly.graph_objects as go

from state.cache import get_dataset_info, set_dataset_info
from utils.dataset_utils import (
    scan_dataset,
    calculate_split_percentages,
    DATASET_ROOT
)
from utils.dataset_sections import (
    render_class_distribution,
    render_sample_visualization,
    render_preprocessing,
    render_augmentation,
    render_confirmation,
)


def render():
    """Main render function for Dataset page"""
    st.title("Dataset Configuration")

    # Initialize dataset cache
    dataset_info = get_dataset_info()
    if not dataset_info:
        with st.spinner("Scanning dataset..."):
            dataset_info = scan_dataset()
            set_dataset_info(dataset_info)

    render_dataset_overview(dataset_info)
    render_data_split(dataset_info)
    render_class_distribution(dataset_info)
    render_sample_visualization(dataset_info)
    render_preprocessing(dataset_info)
    render_augmentation()
    render_confirmation()


def render_dataset_overview(dataset_info):
    """Section 1: Show combined dataset info"""
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Training Samples", f"{dataset_info['total_train']:,}")
    with col2:
        st.metric("Total Validation Samples", f"{dataset_info['total_val']:,}")
    with col3:
        st.metric("Number of Classes", len(dataset_info['classes']))

    st.info(f"Dataset location: `{DATASET_ROOT.relative_to(Path.cwd()).as_posix()}/`")

    # Calculate class imbalance
    if dataset_info['train_samples']:
        samples_per_class = list(dataset_info['train_samples'].values())
        max_samples = max(samples_per_class)
        min_samples = min(samples_per_class)
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Samples/Class", max_samples)
        with col2:
            st.metric("Min Samples/Class", min_samples)

        if imbalance_ratio > 2:
            st.warning(f"Class imbalance detected: {imbalance_ratio:.1f}x ratio")
        else:
            st.success("Classes are relatively balanced")


def render_data_split(dataset_info):
    """Section 2: Configure train/val/test split with 2 sliders"""
    st.header("Train/Validation/Test Split")

    # Current split info
    current_train = dataset_info['total_train']
    current_val = dataset_info['total_val']
    total = current_train + current_val

    if total > 0:
        current_train_pct = (current_train / total) * 100
        current_val_pct = (current_val / total) * 100
        st.info(f"Current dataset split: {current_train_pct:.1f}% train, {current_val_pct:.1f}% validation")

    st.markdown("**Configure Split:**")

    # Slider 1: Training percentage
    train_pct = st.slider(
        "Training %",
        min_value=0,
        max_value=100,
        value=70,
        key="train_split",
        help="Percentage of data for training"
    )

    # Slider 2: Validation percentage from remaining
    remaining = 100 - train_pct
    val_of_remaining = st.slider(
        f"Validation % (of remaining {remaining}%)",
        min_value=0,
        max_value=100,
        value=50,
        key="val_split",
        help="Percentage of remaining data for validation (rest goes to test)"
    )

    # Calculate final percentages
    train_final, val_final, test_final = calculate_split_percentages(train_pct, val_of_remaining)

    # Display final split
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train", f"{train_final:.1f}%")
    with col2:
        st.metric("Validation", f"{val_final:.1f}%")
    with col3:
        st.metric("Test", f"{test_final:.1f}%")

    # Pie chart visualization
    fig = go.Figure(data=[go.Pie(
        labels=['Train', 'Validation', 'Test'],
        values=[train_final, val_final, test_final],
        marker_colors=['#98c127', '#8fd7d7', '#ffb255'],
        hole=0.3
    )])
    fig.update_layout(
        title="Data Split Distribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Stratified Split", value=True,
                    help="Maintain class proportions in each split")
    with col2:
        st.number_input("Random Seed", value=42, min_value=0,
                       help="For reproducible splits")
