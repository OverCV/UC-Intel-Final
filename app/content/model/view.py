"""
Model Configuration Page
Full implementation with layer builder for Custom CNN
"""

from datetime import datetime
from typing import Any

from config import (
    ACTIVATION_FUNCTIONS,
    KERNEL_SIZES,
    PRETRAINED_MODELS,
)
from content.model.layer_builder import (
    export_layer_stack,
    get_layer_stack,
    init_layer_stack,
)
from content.model.layer_configs import validate_layer_stack
from content.model.layer_renderer import (
    render_add_layer_section,
    render_layer_stack,
    render_output_layer,
    render_preset_selector,
    render_validation_status,
)
from state.workflow import (
    get_dataset_config,
    has_dataset_config,
    save_model_config,
)
import streamlit as st


def render():
    """Main render function for Model page"""
    st.title("Model Configuration")

    # Check dataset config first
    if not has_dataset_config():
        st.warning("Please configure dataset first to determine number of classes")
        st.stop()

    dataset_config = get_dataset_config()
    num_classes = len(dataset_config.get("selected_families", []))

    if num_classes == 0:
        st.error("Dataset has no classes configured")
        st.stop()

    # Initialize layer stack for Custom CNN
    init_layer_stack()

    # Section 1: Model Type Selection
    model_type = render_model_type_selection()

    # Section 2-4: Architecture Configuration (based on type)
    if model_type == "Custom CNN":
        cnn_config = render_custom_cnn(num_classes)
        transfer_config = None
        transformer_config = None
    elif model_type == "Transfer Learning":
        transfer_config = render_transfer_learning(num_classes)
        cnn_config = None
        transformer_config = None
    else:  # Transformer
        transformer_config = render_transformer(num_classes)
        cnn_config = None
        transfer_config = None

    # Section 5: Architecture Summary
    model_config = build_model_config(
        model_type, num_classes, cnn_config, transfer_config, transformer_config
    )

    if model_config:
        render_architecture_summary(model_config)

        # Section 6: Save Configuration
        render_save_section(model_config)


def init_model_state():
    """Initialize session state for model configuration"""
    # Model type
    if "model_type" not in st.session_state:
        st.session_state.model_type = "Custom CNN"

    # Transfer learning defaults
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

    # Transformer defaults
    if "transformer_patch_size" not in st.session_state:
        st.session_state.transformer_patch_size = 16
    if "transformer_embed_dim" not in st.session_state:
        st.session_state.transformer_embed_dim = 768
    if "transformer_depth" not in st.session_state:
        st.session_state.transformer_depth = 12
    if "transformer_num_heads" not in st.session_state:
        st.session_state.transformer_num_heads = 12
    if "transformer_mlp_ratio" not in st.session_state:
        st.session_state.transformer_mlp_ratio = 4.0
    if "transformer_dropout" not in st.session_state:
        st.session_state.transformer_dropout = 0.1


def render_model_type_selection():
    """Section 1: Choose model architecture type"""
    st.header("Model Architecture Type")

    col1, col2 = st.columns([2, 1])

    with col1:
        model_type = st.radio(
            "Select Model Type",
            ["Custom CNN", "Transfer Learning", "Transformer"],
            key="model_type",
            help="Choose the type of model architecture to use",
        )

    with col2:
        if model_type == "Custom CNN":
            st.info("Build CNN from scratch with configurable layers")
        elif model_type == "Transfer Learning":
            st.info("Use pre-trained models and fine-tune")
        else:
            st.info("Vision Transformer architecture")

    return model_type


def render_custom_cnn(num_classes: int) -> dict[str, Any]:
    """Configure Custom CNN using the layer builder"""

    st.header("Custom CNN Configuration")

    # Preset selector
    render_preset_selector()

    st.markdown("---")

    # Layer stack
    render_layer_stack()

    # Output layer (auto-configured)
    render_output_layer(num_classes)

    st.markdown("---")

    # Add layer section
    render_add_layer_section()

    # Validation
    is_valid = render_validation_status()

    # Export configuration
    layer_config = export_layer_stack()
    layer_config["num_classes"] = num_classes
    layer_config["is_valid"] = is_valid

    return layer_config


def render_transfer_learning(num_classes: int) -> dict[str, Any]:
    """Configure Transfer Learning model"""

    # Initialize state
    init_model_state()

    st.header("Transfer Learning Configuration")

    # Base Model Selection
    st.subheader("Pre-trained Model")

    col1, col2 = st.columns(2)

    with col1:
        base_model = st.selectbox(
            "Select Base Model",
            options=PRETRAINED_MODELS,
            key="transfer_base_model",
            help="Pre-trained model to use as backbone",
        )

    with col2:
        weights = st.radio(
            "Initial Weights",
            ["ImageNet", "Random"],
            key="transfer_weights",
            help="Use pre-trained weights or random initialization",
        )

    # Fine-tuning Strategy
    st.subheader("Fine-tuning Strategy")

    strategy = st.radio(
        "Training Strategy",
        ["Feature Extraction", "Partial Fine-tuning", "Full Fine-tuning"],
        key="transfer_strategy",
        help="How to train the pre-trained model",
    )

    unfreeze_layers = 0
    if strategy == "Partial Fine-tuning":
        unfreeze_layers = st.slider(
            "Number of Layers to Unfreeze",
            min_value=0,
            max_value=50,
            key="transfer_unfreeze_layers",
            help="Number of top layers to make trainable",
        )

    # Classification Head
    st.subheader("Classification Head")

    col1, col2, col3 = st.columns(3)

    with col1:
        global_pooling = st.checkbox(
            "Global Average Pooling",
            key="transfer_global_pooling",
            help="Add global pooling before dense layers",
        )

    with col2:
        add_dense = st.checkbox(
            "Add Dense Layer",
            key="transfer_add_dense",
            help="Add extra dense layer before output",
        )

        dense_units = 512
        if add_dense:
            dense_units = st.selectbox(
                "Dense Units", options=[256, 512, 1024], key="transfer_dense_units"
            )

    with col3:
        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.7,
            key="transfer_dropout",
            help="Dropout before final layer",
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


def render_transformer(num_classes: int) -> dict[str, Any]:
    """Configure Vision Transformer model"""

    # Initialize state
    init_model_state()

    st.header("Vision Transformer Configuration")

    st.info("Custom Vision Transformer (ViT) implementation")

    # Patch and Embedding Configuration
    st.subheader("Patch & Embedding Configuration")

    col1, col2 = st.columns(2)

    with col1:
        patch_size = st.selectbox(
            "Patch Size",
            options=[8, 14, 16, 32],
            key="transformer_patch_size",
            help="Size of image patches (smaller = more patches)",
        )

        embed_dim = st.selectbox(
            "Embedding Dimension",
            options=[384, 512, 768, 1024],
            key="transformer_embed_dim",
            help="Dimension of patch embeddings",
        )

    with col2:
        depth = st.slider(
            "Transformer Depth",
            min_value=6,
            max_value=24,
            key="transformer_depth",
            help="Number of transformer blocks",
        )

        num_heads = st.selectbox(
            "Attention Heads",
            options=[6, 8, 12, 16],
            key="transformer_num_heads",
            help="Number of attention heads",
        )

    # MLP Configuration
    st.subheader("MLP Configuration")

    col1, col2 = st.columns(2)

    with col1:
        mlp_ratio = st.slider(
            "MLP Ratio",
            min_value=2.0,
            max_value=8.0,
            step=0.5,
            key="transformer_mlp_ratio",
            help="MLP hidden dim = embed_dim * mlp_ratio",
        )

    with col2:
        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            key="transformer_dropout",
            help="Dropout rate in transformer blocks",
        )

    # Display estimated parameters
    num_patches = (224 // patch_size) ** 2
    st.info(
        f"Image will be split into {num_patches} patches of size {patch_size}x{patch_size}"
    )
    st.info(f"Output Layer: {num_classes} units with Softmax activation")

    return {
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "num_classes": num_classes,
        "num_patches": num_patches,
    }


def build_model_config(
    model_type: str,
    num_classes: int,
    cnn_config: dict[str, Any] | None,
    transfer_config: dict[str, Any] | None,
    transformer_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build complete model configuration"""

    config = {
        "model_type": model_type,
        "num_classes": num_classes,
        "timestamp": datetime.now().isoformat(),
    }

    if model_type == "Custom CNN" and cnn_config:
        num_layers = len(cnn_config.get("layers", []))
        config["architecture"] = f"CNN_{num_layers}_layers"
        config["cnn_config"] = cnn_config
        config["implementation"] = "pytorch"

    elif model_type == "Transfer Learning" and transfer_config:
        config["architecture"] = (
            f"{transfer_config['base_model']}_{transfer_config['strategy'].replace(' ', '_')}"
        )
        config["transfer_config"] = transfer_config
        config["implementation"] = "pytorch"

    elif model_type == "Transformer" and transformer_config:
        config["architecture"] = (
            f"ViT_D{transformer_config['depth']}_H{transformer_config['num_heads']}"
        )
        config["transformer_config"] = transformer_config
        config["implementation"] = "manual"

    return config


def render_architecture_summary(model_config: dict[str, Any]):
    """Display model architecture summary"""

    st.header("Model Architecture Summary")

    # Basic Info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", model_config["model_type"])
    with col2:
        st.metric("Architecture", model_config["architecture"])
    with col3:
        st.metric("Output Classes", model_config["num_classes"])

    # Model-specific details
    if model_config["model_type"] == "Custom CNN":
        cnn_cfg = model_config.get("cnn_config", {})
        layers = cnn_cfg.get("layers", [])
        st.write(f"**Total Layers:** {len(layers)}")

        # Count layer types
        layer_counts = {}
        for layer in layers:
            lt = layer["type"]
            layer_counts[lt] = layer_counts.get(lt, 0) + 1

        if layer_counts:
            counts_str = ", ".join(
                [f"{count} {ltype}" for ltype, count in layer_counts.items()]
            )
            st.write(f"**Layer Composition:** {counts_str}")

    elif model_config["model_type"] == "Transfer Learning":
        transfer_cfg = model_config["transfer_config"]
        st.write(f"**Base Model:** {transfer_cfg['base_model']}")
        st.write(f"**Weights:** {transfer_cfg['weights']}")
        st.write(f"**Strategy:** {transfer_cfg['strategy']}")
        if transfer_cfg["strategy"] == "Partial Fine-tuning":
            st.write(f"**Unfrozen Layers:** {transfer_cfg['unfreeze_layers']}")

    elif model_config["model_type"] == "Transformer":
        transformer_cfg = model_config["transformer_config"]
        st.write(
            f"**Patch Size:** {transformer_cfg['patch_size']}x{transformer_cfg['patch_size']}"
        )
        st.write(f"**Embedding Dim:** {transformer_cfg['embed_dim']}")
        st.write(f"**Depth:** {transformer_cfg['depth']} layers")
        st.write(f"**Attention Heads:** {transformer_cfg['num_heads']}")

    # Placeholder for actual model metrics
    st.subheader("Model Metrics")
    st.info("Model metrics will be calculated when the model is built during training")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", "Will be calculated")
    with col2:
        st.metric("Trainable Parameters", "Will be calculated")
    with col3:
        st.metric("Model Size", "Will be calculated")


def render_save_section(model_config: dict[str, Any]):
    """Save model configuration"""

    st.header("Save Configuration")

    # Check validation for Custom CNN
    is_valid = True
    if model_config["model_type"] == "Custom CNN":
        cnn_cfg = model_config.get("cnn_config", {})
        is_valid = cnn_cfg.get("is_valid", False)

    col1, col2 = st.columns([3, 1])

    with col1:
        if is_valid:
            st.success("Model configuration is ready!")
        else:
            st.warning("Fix validation errors before saving")

        # Show configuration preview
        with st.expander("View Full Configuration", expanded=False):
            st.json(model_config)

    with col2:
        if st.button(
            "Save Model Config",
            type="primary",
            use_container_width=True,
            disabled=not is_valid,
        ):
            save_model_config(model_config)
            st.success("Model configuration saved!")
            st.balloons()

            st.info("You can now proceed to Training Configuration")
