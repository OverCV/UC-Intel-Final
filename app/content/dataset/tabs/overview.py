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
    st.divider()
    render_class_imbalance_handling(dataset_info)


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
        step=5,
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
        step=5,
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
            key="stratified_split",
            help="Maintain class proportions in each split",
        )
    with col2:
        st.number_input(
            "Random Seed", value=73, min_value=0,
            key="random_seed",
            help="For reproducible splits"
        )


def render_class_imbalance_handling(dataset_info):
    """Class imbalance mitigation options"""
    st.subheader("Class Imbalance Handling")

    # Calculate imbalance for selected classes
    if "selected_classes" in st.session_state and st.session_state.selected_classes:
        selected_classes = st.session_state.selected_classes
        samples_per_class = [dataset_info["train_samples"].get(c, 0) for c in selected_classes]
    else:
        samples_per_class = list(dataset_info["train_samples"].values())

    if samples_per_class:
        max_samples = max(samples_per_class)
        min_samples = min(samples_per_class) if min(samples_per_class) > 0 else 1
        imbalance_ratio = max_samples / min_samples

        # Show imbalance severity
        if imbalance_ratio > 10:
            st.error(f"⚠️ **Severe Imbalance**: {imbalance_ratio:.1f}:1 ratio between largest and smallest class")
        elif imbalance_ratio > 3:
            st.warning(f"⚠️ **Moderate Imbalance**: {imbalance_ratio:.1f}:1 ratio")
        else:
            st.success(f"✅ **Balanced Dataset**: {imbalance_ratio:.1f}:1 ratio")

    # Handling strategy selection
    st.markdown("### Mitigation Strategy")

    strategy = st.radio(
        "Select imbalance handling method",
        [
            "Auto Class Weights (Recommended)",
            "Manual Class Weights",
            "Oversampling (SMOTE)",
            "Undersampling",
            "No Adjustment"
        ],
        key="imbalance_strategy",
        help="Choose how to handle class imbalance during training"
    )

    # Strategy-specific options
    if strategy == "Auto Class Weights (Recommended)":
        st.info("""
        **Auto Class Weights**: Automatically calculates class weights inversely proportional to class frequencies.
        - Classes with fewer samples get higher weights
        - Balanced loss function during training
        - No data duplication or removal
        """)

    elif strategy == "Manual Class Weights":
        st.markdown("**Set custom weights for each class:**")

        # Get selected classes
        if "selected_classes" in st.session_state and st.session_state.selected_classes:
            classes_to_weight = sorted(st.session_state.selected_classes)
        else:
            classes_to_weight = sorted(dataset_info["classes"])

        # Create weight inputs for each class
        weights = {}
        cols = st.columns(3)
        for i, cls in enumerate(classes_to_weight):
            col_idx = i % 3
            with cols[col_idx]:
                count = dataset_info["train_samples"].get(cls, 0)
                default_weight = 1.0 if count == 0 else (max_samples / count)
                weights[cls] = st.number_input(
                    f"{cls} ({count} samples)",
                    min_value=0.1,
                    max_value=10.0,
                    value=min(default_weight, 10.0),
                    step=0.1,
                    key=f"weight_{cls}",
                    help=f"Weight for {cls} class"
                )

        # Store weights in session state
        st.session_state.class_weights = weights

    elif strategy == "Oversampling (SMOTE)":
        st.info("""
        **SMOTE (Synthetic Minority Over-sampling)**:
        - Generates synthetic samples for minority classes
        - Creates new samples by interpolating between existing ones
        - Increases training set size
        """)

        sampling_ratio = st.slider(
            "Sampling Ratio",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="smote_ratio",
            help="Target ratio of minority to majority class after resampling"
        )

    elif strategy == "Undersampling":
        st.info("""
        **Random Undersampling**:
        - Reduces samples from majority classes
        - Balances dataset by removing data
        - May lose important information
        """)

        st.warning("⚠️ Undersampling reduces your training data size")

    elif strategy == "No Adjustment":
        st.info("""
        **No Adjustment**:
        - Train with natural class distribution
        - May lead to bias towards majority classes
        - Consider if imbalance reflects real-world distribution
        """)

    # Store strategy in session state
    st.session_state.imbalance_handling = strategy
