"""
Page 3: Model Configuration
BFS Level: Interface & Section Structure Only
"""

import streamlit as st


def render():
    """Main render function for Model page"""
    st.title("Model Configuration")

    # Section 1: Model Type Selection
    model_type = render_model_type_selection()

    if model_type == "Custom CNN":
        render_custom_cnn()
    elif model_type == "Transfer Learning":
        render_transfer_learning()

    # Section 5: Architecture Summary (both types)
    render_architecture_summary()

    # Section 7: Continue to Training
    render_navigation()


def render_model_type_selection():
    """Section 1: Choose Custom CNN or Transfer Learning"""
    st.header("Model Architecture")

    model_type = st.radio(
        "Model Type",
        ["Custom CNN", "Transfer Learning"],
        key="model_type",
    )

    return model_type


def render_custom_cnn():
    """Sections 2-4: Custom CNN configuration"""

    # Section 2: Convolutional Layers
    st.header("Convolutional Layers")

    # TODO: Dynamic block addition
    num_blocks = st.number_input("Number of Conv Blocks", 1, 7, 3)

    for i in range(num_blocks):
        with st.expander(f"Convolutional Block {i + 1}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.slider(f"Filters {i + 1}", 32, 512, 64, step=32)
                st.selectbox(f"Kernel Size {i + 1}", ["3x3", "5x5", "7x7"])
                st.selectbox(f"Activation {i + 1}", ["ReLU", "Mish", "Swish", "GELU"])
            with col2:
                st.checkbox(f"Second Conv Layer {i + 1}")
                st.checkbox(f"MaxPooling {i + 1}", value=True)
                st.slider(f"Dropout {i + 1}", 0.0, 0.5, 0.25)

    # Section 3: Dense Layers
    st.header("Dense Layers")

    num_dense = st.number_input("Number of Dense Layers", 1, 5, 2)

    for i in range(num_dense):
        with st.expander(f"Dense Layer {i + 1}", expanded=True):
            st.slider(f"Units {i + 1}", 64, 1024, 256, step=64)
            st.selectbox(f"Dense Activation {i + 1}", ["ReLU", "Mish", "Swish", "GELU"])
            st.slider(f"Dense Dropout {i + 1}", 0.0, 0.7, 0.5)

    # Section 4: Output Layer (Auto)
    st.header("Output Layer")
    st.info("Auto-configured: 25 units (Softmax)")


def render_transfer_learning():
    """Sections 2-4: Transfer learning configuration"""

    # Section 2: Pre-trained Model Selection
    st.header("Pre-trained Model")

    base_model = st.radio(
        "Select Base Model",
        ["VGG16", "VGG19", "ResNet50", "ResNet101", "InceptionV3", "EfficientNetB0"],
    )

    weights = st.radio(
        "Weights", ["ImageNet (recommended)", "Random (train from scratch)"]
    )

    # Section 3: Fine-tuning Strategy
    st.header("Fine-tuning Configuration")

    strategy = st.radio(
        "Strategy",
        ["Feature Extraction (freeze all)", "Partial Fine-tuning", "Full Fine-tuning"],
    )

    if strategy == "Partial Fine-tuning":
        st.slider("Layers to Unfreeze", 0, 50, 10)

    # Section 4: Custom Top Layers
    st.header("Classification Head")

    st.checkbox("Global Average Pooling", value=True)
    add_dense = st.checkbox("Add Dense Layer")
    if add_dense:
        st.slider("Dense Units", 256, 1024, 512)
    st.slider("Dropout", 0.0, 0.7, 0.5)


def render_architecture_summary():
    """Section 5: Model summary"""
    st.header("Model Architecture Summary")

    st.text("Model architecture visualization - TO BE IMPLEMENTED")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", "TBD")
    with col2:
        st.metric("Trainable Parameters", "TBD")
    with col3:
        st.metric("Estimated Memory", "TBD")

    # TODO: Build model and calculate actual params


def render_navigation():
    """Section 7: Continue to training"""
    st.divider()
    st.success("Model Configuration Complete")

    if st.button("Next: Training Configuration", type="primary"):
        st.info("Navigation - TO BE IMPLEMENTED")
