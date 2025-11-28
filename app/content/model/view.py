"""Model Configuration Page - Main Orchestrator"""

from content.model import custom_cnn, summary, transfer_learning, transformer
from state.workflow import get_dataset_config, has_dataset_config
import streamlit as st


def render():
    """Main render function for Model page"""
    st.title("Model Configuration")

    if not has_dataset_config():
        st.warning("Please configure dataset first to determine number of classes")
        st.stop()

    dataset_config = get_dataset_config()
    num_classes = len(dataset_config.get("selected_families", []))

    if num_classes == 0:
        st.error("Dataset has no classes configured")
        st.stop()

    # Section 1: Model Type Selection
    model_type = _render_model_type_selection()

    # Section 2-4: Architecture Configuration (based on type)
    cnn_config = None
    transformer_config = None
    transfer_config = None

    if model_type == "Custom CNN":
        cnn_config = custom_cnn.render(num_classes)
    elif model_type == "Transformer":
        transformer_config = transformer.render(num_classes)
    else:  # Transfer Learning
        transfer_config = transfer_learning.render(num_classes)

    # Section 5: Architecture Summary
    model_config = summary.build_config(
        model_type, num_classes, cnn_config, transfer_config, transformer_config
    )

    if model_config:
        summary.render_summary(model_config)
        summary.render_save(model_config)


def _render_model_type_selection():
    """Section 1: Choose model architecture type"""
    st.header("Model Architecture Type")

    model_type = st.segmented_control(
        "Select Model Type",
        options=["Custom CNN", "Transformer", "Transfer Learning"],
        default="Custom CNN",
        key="model_type",
        help="Choose the type of model architecture",
    )

    descriptions = {
        "Custom CNN": "🔧 **Build from scratch.** Stack convolutional layers manually. Full control over architecture depth, filters, and regularization.",
        "Transformer": "🧪 **Experimental.** Vision Transformer (ViT) architecture. State-of-the-art but computationally expensive and data-hungry.",
        "Transfer Learning": "🎯 **Recommended.** Use pre-trained models (ResNet, VGG, EfficientNet) fine-tuned on your malware dataset. Faster training, better results with less data.",
    }

    st.info(descriptions.get(model_type, ""))

    return model_type
