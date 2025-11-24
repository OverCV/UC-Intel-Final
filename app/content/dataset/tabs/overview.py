"""
Dataset Tab 1: Overview & Split Configuration
Shows dataset statistics and train/val/test split configuration
"""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from utils.dataset_utils import DATASET_ROOT, calculate_split_percentages


def render(dataset_info):
    """Render overview and split configuration tab"""
    render_dataset_overview(dataset_info)
    st.divider()
    render_data_split(dataset_info)


def render_dataset_overview(dataset_info):
    """Dataset statistics and info"""
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Training Samples", f"{dataset_info['total_train']:,}")
    with col2:
        st.metric("Total Validation Samples", f"{dataset_info['total_val']:,}")
    with col3:
        st.metric("Number of Classes", len(dataset_info["classes"]))

    st.info(f"Dataset location: `{DATASET_ROOT.relative_to(Path.cwd()).as_posix()}/`")

    # Calculate class imbalance
    if dataset_info["train_samples"]:
        samples_per_class = list(dataset_info["train_samples"].values())
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
    """Train/val/test split configuration"""
    st.subheader("Train/Validation/Test Split")

    # Current split info
    current_train = dataset_info["total_train"]
    current_val = dataset_info["total_val"]
    total = current_train + current_val

    if total > 0:
        current_train_pct = (current_train / total) * 100
        current_val_pct = (current_val / total) * 100
        st.info(
            f"Current dataset split: {current_train_pct:.1f}% train, {current_val_pct:.1f}% validation"
        )

    st.markdown("**Configure Split:**")

    # Slider 1: Training percentage
    train_pct = st.slider(
        "Training %",
        min_value=0,
        max_value=100,
        value=70,
        key="train_split",
        help="Percentage of data for training",
    )

    # Slider 2: Validation percentage from remaining
    remaining = 100 - train_pct
    val_of_remaining = st.slider(
        f"Validation % (of remaining {remaining}%)",
        min_value=0,
        max_value=100,
        value=50,
        key="val_split",
        help="Percentage of remaining data for validation (rest goes to test)",
    )

    # Calculate final percentages
    train_final, val_final, test_final = calculate_split_percentages(
        train_pct, val_of_remaining
    )

    # Display final split
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train", f"{train_final:.1f}%")
    with col2:
        st.metric("Validation", f"{val_final:.1f}%")
    with col3:
        st.metric("Test", f"{test_final:.1f}%")

    # Pie chart visualization
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Train", "Validation", "Test"],
                values=[train_final, val_final, test_final],
                marker_colors=["#98c127", "#8fd7d7", "#ffb255"],
                hole=0.3,
            )
        ]
    )
    fig.update_layout(
        title="Data Split Distribution",
        height=300,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox(
            "Stratified Split",
            value=True,
            help="Maintain class proportions in each split",
        )
    with col2:
        st.number_input(
            "Random Seed", value=42, min_value=0, help="For reproducible splits"
        )
