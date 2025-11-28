"""
Monitor Page - Experiment Composition and Training Monitor
Compose model + training configs, start training, watch progress
"""

from components.experiment_row import render_experiment_row
from state.workflow import (
    create_experiment,
    delete_experiment,
    get_experiments,
    get_model_library,
    get_training_library,
    update_experiment,
)
import streamlit as st


def render():
    """Main render function for Monitor page"""
    st.title("Training Monitor")

    # Get libraries
    models = get_model_library()
    trainings = get_training_library()

    # Check prerequisites
    if not models:
        st.warning("No models saved. Create a model in the Model page first.")

    if not trainings:
        st.warning("No training configs saved. Create one in the Training page first.")

    # Header with add button
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Experiments")

    with col2:
        if st.button("+ New Experiment", type="primary", use_container_width=True):
            _create_new_experiment(models, trainings)
            st.rerun()

    # Render experiments
    experiments = get_experiments()

    if not experiments:
        st.info(
            "No experiments yet. Click '+ New Experiment' to create one, "
            "then select a model and training config to start training."
        )
    else:
        for exp in experiments:
            render_experiment_row(
                experiment=exp,
                models=models,
                trainings=trainings,
                on_update=_handle_experiment_update,
                on_delete=_handle_experiment_delete,
                on_start=_handle_start_training,
                on_pause=_handle_pause_training,
                on_stop=_handle_stop_training,
                on_view_results=_handle_view_results,
            )


def _create_new_experiment(models: list, trainings: list):
    """Create a new experiment with defaults"""
    exp_count = len(get_experiments()) + 1
    name = f"Experiment {exp_count}"

    # Default to first model and training if available
    model_id = models[0]["id"] if models else None
    training_id = trainings[0]["id"] if trainings else None

    create_experiment(name, model_id, training_id)


def _handle_experiment_update(exp_id: str, updates: dict):
    """Handle experiment config changes"""
    update_experiment(exp_id, updates)
    st.rerun()


def _handle_experiment_delete(exp_id: str):
    """Handle experiment deletion"""
    delete_experiment(exp_id)
    st.rerun()


def _handle_start_training(exp_id: str):
    """Handle start training button"""
    # Update status to training
    update_experiment(
        exp_id,
        {
            "status": "training",
            "started_at": __import__("datetime").datetime.now().isoformat(),
            "current_epoch": 0,
            "metrics": {},
        },
    )

    st.toast("Training started!")
    st.info("Training functionality will be implemented with actual PyTorch training loop.")
    st.rerun()


def _handle_pause_training(exp_id: str):
    """Handle pause training button"""
    update_experiment(exp_id, {"status": "paused"})
    st.toast("Training paused")
    st.rerun()


def _handle_stop_training(exp_id: str):
    """Handle stop training button"""
    update_experiment(exp_id, {"status": "ready", "current_epoch": 0, "metrics": {}})
    st.toast("Training stopped")
    st.rerun()


def _handle_view_results(exp_id: str):
    """Handle view results button"""
    st.session_state.selected_experiment_id = exp_id
    st.toast("Experiment selected. Navigate to Results page to view details.")
