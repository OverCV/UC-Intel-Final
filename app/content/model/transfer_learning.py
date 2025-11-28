"""Transfer Learning configuration UI"""

from config import PRETRAINED_MODELS
import streamlit as st


def _init_state():
    """Initialize transfer learning session state defaults"""
    if "transfer_base_model" not in st.session_state:
        st.session_state.transfer_base_model = "ResNet50"
    if "transfer_weights" not in st.session_state:
        st.session_state.transfer_weights = "ImageNet"
    if "transfer_strategy" not in st.session_state:
        st.session_state.transfer_strategy = "Feature Extraction"
    if "transfer_unfreeze_layers" not in st.session_state:
        st.session_state.transfer_unfreeze_layers = 10
    if "transfer_global_pooling" not in st.session_state:
        st.session_state.transfer_global_pooling = True
    if "transfer_add_dense" not in st.session_state:
        st.session_state.transfer_add_dense = True
    if "transfer_dense_units" not in st.session_state:
        st.session_state.transfer_dense_units = 512
    if "transfer_dropout" not in st.session_state:
        st.session_state.transfer_dropout = 0.5


def render(num_classes: int) -> dict:
    """Configure Transfer Learning model"""
    _init_state()

    st.header("Transfer Learning Configuration")

    # Base Model Selection
    st.subheader("Pre-trained Model")

    col1, col2 = st.columns(2)

    with col1:
        base_model = st.selectbox(
            "Select Base Model",
            options=PRETRAINED_MODELS,
            key="transfer_base_model",
            help="Pre-trained CNN backbone. ResNet50 is a solid default. EfficientNet offers better accuracy/speed trade-off. VGG is simpler but larger.",
        )

    with col2:
        weights = st.radio(
            "Initial Weights",
            ["ImageNet", "Random"],
            key="transfer_weights",
            help="ImageNet: Start with weights learned from 1M+ natural images (recommended). Random: Train from scratch (slower, needs more data).",
        )

    # Fine-tuning Strategy
    st.subheader("Fine-tuning Strategy")

    strategy = st.radio(
        "Training Strategy",
        ["Feature Extraction", "Partial Fine-tuning", "Full Fine-tuning"],
        key="transfer_strategy",
        help="Feature Extraction: Freeze all base layers, train only classifier (fast, less overfitting). Partial: Unfreeze top layers (balanced). Full: Train everything (best if you have lots of data).",
    )

    unfreeze_layers = 0
    if strategy == "Partial Fine-tuning":
        unfreeze_layers = st.slider(
            "Number of Layers to Unfreeze",
            min_value=0,
            max_value=50,
            key="transfer_unfreeze_layers",
            help="How many layers from the top to make trainable. More layers = more flexibility but higher risk of overfitting. Start with 10-20.",
        )

    # Classification Head
    st.subheader("Classification Head")

    col1, col2, col3 = st.columns(3)

    with col1:
        global_pooling = st.checkbox(
            "Global Average Pooling",
            key="transfer_global_pooling",
            help="Reduces each feature map to a single value. Required for most pre-trained models to connect to dense layers.",
        )

    with col2:
        add_dense = st.checkbox(
            "Add Dense Layer",
            key="transfer_add_dense",
            help="Extra fully-connected layer before output. Adds capacity for learning malware-specific patterns.",
        )

        dense_units = 512
        if add_dense:
            dense_units = st.selectbox(
                "Dense Units",
                options=[256, 512, 1024],
                key="transfer_dense_units",
                help="Number of neurons in the added dense layer. 512 is a good default for ~50 classes.",
            )

    with col3:
        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.7,
            key="transfer_dropout",
            help="Randomly drops neurons during training to prevent overfitting. 0.3-0.5 is typical. Higher = more regularization.",
        )

    st.info(f"Output Layer: {num_classes} units with Softmax activation")

    return {
        "base_model": base_model,
        "weights": weights,
        "strategy": strategy,
        "unfreeze_layers": unfreeze_layers if strategy == "Partial Fine-tuning" else 0,
        "global_pooling": global_pooling,
        "add_dense": add_dense,
        "dense_units": dense_units if add_dense else 0,
        "dropout": dropout,
        "num_classes": num_classes,
    }
