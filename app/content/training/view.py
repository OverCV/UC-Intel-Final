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

    optimizer = st.selectbox(
        "Optimizer",
        ["Adam", "AdamW", "SGD with Momentum", "RMSprop"],
        help="Algorithm that updates model weights. Adam/AdamW are good defaults. SGD can give better generalization but requires tuning.",
    )
    lr = st.slider(
        "Learning Rate",
        0.0001,
        0.01,
        0.001,
        format="%.4f",
        help="Controls how much weights change per update. Too high = unstable training. Too low = very slow learning. Start with 0.001.",
    )

    with st.expander("Advanced Optimizer Parameters"):
        if optimizer in ["Adam", "AdamW"]:
            st.slider(
                "Beta 1",
                0.0,
                1.0,
                0.9,
                help="Exponential decay rate for first moment (mean). Controls momentum. Default 0.9 works well.",
            )
            st.slider(
                "Beta 2",
                0.0,
                1.0,
                0.999,
                help="Exponential decay rate for second moment (variance). Affects adaptive learning rate. Default 0.999 is stable.",
            )
        elif optimizer == "SGD with Momentum":
            st.slider(
                "Momentum",
                0.0,
                1.0,
                0.9,
                help="Helps accelerate SGD in relevant direction. Reduces oscillations. Typical values: 0.9-0.99.",
            )
            st.checkbox(
                "Nesterov",
                value=True,
                help="Nesterov momentum looks ahead before making updates. Usually improves convergence.",
            )


def render_lr_scheduler():
    """Section 2: LR scheduling"""
    st.header("Learning Rate Scheduling")

    strategy = st.radio(
        "Strategy",
        [
            "Constant",
            "ReduceLROnPlateau",
            "Cosine Annealing",
            "Step Decay",
            "Exponential Decay",
        ],
        help="How learning rate changes during training. Constant=fixed, ReduceLROnPlateau=decreases when stuck, Cosine=smooth decay, Step/Exponential=scheduled decay.",
    )

    if strategy == "ReduceLROnPlateau":
        with st.expander("Scheduler Parameters"):
            st.slider(
                "Reduction Factor",
                0.1,
                0.9,
                0.5,
                help="Multiply learning rate by this when plateau detected. 0.5 means cut LR in half.",
            )
            st.slider(
                "Patience",
                3,
                20,
                5,
                help="Number of epochs with no improvement before reducing LR. Higher = more patient before changes.",
            )


def render_training_parameters():
    """Section 3: Epochs and batch size"""
    st.header("Training Settings")

    epochs = st.slider(
        "Max Epochs",
        10,
        200,
        100,
        help="Maximum training iterations through entire dataset. More epochs = longer training. Use early stopping to avoid overfitting.",
    )
    batch_size = st.selectbox(
        "Batch Size",
        [16, 32, 64, 128],
        index=1,
        help="Number of samples processed before weight update. Larger = faster but needs more memory. Smaller = more stable gradients. 32 is a good default.",
    )
    st.checkbox(
        "Shuffle Training Data",
        value=True,
        help="Randomize sample order each epoch. Recommended to prevent learning order-dependent patterns.",
    )


def render_regularization():
    """Section 4: Regularization config"""
    st.header("Regularization")

    use_l2 = st.checkbox(
        "L2 Weight Decay",
        help="Penalizes large weights to prevent overfitting. Adds small penalty proportional to weight magnitude. Recommended for most models.",
    )
    if use_l2:
        st.slider(
            "Lambda",
            0.0001,
            0.01,
            0.0001,
            format="%.4f",
            help="Weight decay strength. Higher = stronger regularization. Start with 0.0001-0.001. Too high can underfit.",
        )

    st.info("Dropout configured in Model Architecture")


def render_class_imbalance():
    """Section 5: Handle imbalanced classes"""
    st.header("Class Imbalance Strategy")

    method = st.radio(
        "Method",
        ["Auto Class Weights (recommended)", "Focal Loss", "No Adjustment"],
        help="Handle datasets where some classes have far fewer samples. Auto weights penalize errors on rare classes more. Focal Loss focuses on hard examples.",
    )

    with st.expander("Class Distribution"):
        st.info("Bar chart - TO BE IMPLEMENTED")
        # TODO: Show class distribution from dataset config


def render_callbacks():
    """Section 6: Training callbacks"""
    st.header("Training Callbacks")

    early_stopping = st.checkbox(
        "Early Stopping",
        value=True,
        help="Stop training if validation performance stops improving. Prevents overfitting and saves time.",
    )
    if early_stopping:
        st.slider(
            "Patience",
            5,
            30,
            10,
            help="Number of epochs to wait for improvement before stopping. Higher = more chances to improve, but may waste time.",
        )

    checkpointing = st.checkbox(
        "Model Checkpointing",
        value=True,
        help="Save model weights at best performance point. Lets you recover best model even if training continues past optimal point.",
    )
    if checkpointing:
        st.radio(
            "Save Best By",
            ["Val Loss", "Val Accuracy"],
            help="Which metric determines 'best' model. Val Loss is safer (optimizes what's being minimized). Val Accuracy is more intuitive.",
        )

    st.checkbox(
        "TensorBoard Logging",
        help="Log training metrics to TensorBoard for visualization. Creates detailed charts of loss, accuracy, learning rate over time.",
    )


def render_start_training():
    """Section 7: Start training"""
    st.header("Experiment Details")

    exp_name = st.text_input(
        "Experiment Name",
        value="exp_001",
        help="Unique identifier for this training run. Use descriptive names like 'resnet18_aug_heavy' to track experiments.",
    )
    st.text_area(
        "Description (optional)",
        help="Notes about this experiment: What are you testing? What hypothesis? Any special configuration?",
    )
    st.multiselect(
        "Tags",
        ["mish", "relu", "baseline", "experiment"],
        help="Labels to organize experiments. Useful for grouping related runs or marking special configurations.",
    )

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
