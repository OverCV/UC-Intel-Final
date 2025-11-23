"""
Page 4: Training Configuration
BFS Level: Interface & Section Structure Only
"""

import streamlit as st


def render():
    """Main render function for Training page"""
    st.title("Training Configuration")

    # Section 1: Optimizer Settings
    render_optimizer_settings()

    # Section 2: Learning Rate Scheduler
    render_lr_scheduler()

    # Section 3: Training Parameters
    render_training_parameters()

    # Section 4: Regularization
    render_regularization()

    # Section 5: Class Imbalance Handling
    render_class_imbalance()

    # Section 6: Callbacks & Early Stopping
    render_callbacks()

    # Section 7: Experiment Metadata & Start Training
    render_start_training()


def render_optimizer_settings():
    """Section 1: Optimizer configuration"""
    st.header("Optimizer Configuration")

    optimizer = st.selectbox("Optimizer", ["Adam", "AdamW", "SGD with Momentum", "RMSprop"])
    lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

    with st.expander("Advanced Optimizer Parameters"):
        if optimizer in ["Adam", "AdamW"]:
            st.slider("Beta 1", 0.0, 1.0, 0.9)
            st.slider("Beta 2", 0.0, 1.0, 0.999)
        elif optimizer == "SGD with Momentum":
            st.slider("Momentum", 0.0, 1.0, 0.9)
            st.checkbox("Nesterov", value=True)


def render_lr_scheduler():
    """Section 2: LR scheduling"""
    st.header("Learning Rate Scheduling")

    strategy = st.radio(
        "Strategy",
        ["Constant", "ReduceLROnPlateau", "Cosine Annealing", "Step Decay", "Exponential Decay"]
    )

    if strategy == "ReduceLROnPlateau":
        with st.expander("Scheduler Parameters"):
            st.slider("Reduction Factor", 0.1, 0.9, 0.5)
            st.slider("Patience", 3, 20, 5)

    # TODO: Add LR schedule preview graph


def render_training_parameters():
    """Section 3: Epochs and batch size"""
    st.header("Training Settings")

    epochs = st.slider("Max Epochs", 10, 200, 100)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    st.checkbox("Shuffle Training Data", value=True)

    # TODO: Calculate and show training time estimate


def render_regularization():
    """Section 4: Regularization config"""
    st.header("Regularization")

    use_l2 = st.checkbox("L2 Weight Decay")
    if use_l2:
        st.slider("Lambda", 0.0001, 0.01, 0.0001, format="%.4f")

    st.info("Dropout configured in Model Architecture")


def render_class_imbalance():
    """Section 5: Handle imbalanced classes"""
    st.header("Class Imbalance Strategy")

    method = st.radio(
        "Method",
        ["Auto Class Weights (recommended)", "Focal Loss", "No Adjustment"]
    )

    with st.expander("Class Distribution"):
        st.info("Bar chart - TO BE IMPLEMENTED")
        # TODO: Show class distribution from dataset config


def render_callbacks():
    """Section 6: Training callbacks"""
    st.header("Training Callbacks")

    early_stopping = st.checkbox("Early Stopping", value=True)
    if early_stopping:
        st.slider("Patience", 5, 30, 10)

    checkpointing = st.checkbox("Model Checkpointing", value=True)
    if checkpointing:
        st.radio("Save Best By", ["Val Loss", "Val Accuracy"])

    st.checkbox("TensorBoard Logging")


def render_start_training():
    """Section 7: Start training"""
    st.header("Experiment Details")

    exp_name = st.text_input("Experiment Name", value="exp_001")
    st.text_area("Description (optional)")
    st.multiselect("Tags", ["mish", "relu", "baseline", "experiment"])

    st.divider()

    with st.form("start_training_form"):
        st.info("""
        **Training Configuration Summary**
        - Dataset: MalImg (TBD samples, TBD classes)
        - Model: TBD
        - Epochs: TBD | Batch: TBD | LR: TBD
        """)

        st.warning("Training will take approximately X hours")

        submitted = st.form_submit_button("START TRAINING", type="primary")
        if submitted:
            st.success("Training started - TO BE IMPLEMENTED")
            # TODO: Start training, show embedded monitor
