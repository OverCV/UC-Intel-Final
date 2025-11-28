"""
Monitor Page - Experiment Composition and Training Monitor
Compose model + training configs, start training, watch progress
"""

from components.experiment_row import render_experiment_row
from state.persistence import read_session_data
from state.workflow import (
    create_experiment,
    delete_experiment,
    get_experiments,
    get_model_library,
    get_session_id,
    get_training_library,
    update_experiment,
)
from training.worker import pause_training, resume_training, start_training, stop_training
import streamlit as st


def render():
    """Main render function for Monitor page"""
    st.title("Training Monitor")

    # Get experiments - reload from file if training is active
    # This is needed because the background thread writes to file, not session_state
    experiments = _get_experiments_with_live_updates()
    has_active_training = any(exp.get("status") == "training" for exp in experiments)

    if has_active_training:
        # Refresh every 2 seconds during training
        st.markdown(
            """<meta http-equiv="refresh" content="2">""",
            unsafe_allow_html=True,
        )

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

    # Render experiments (use the already-fetched list)
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
    start_training(exp_id)
    st.toast("Training started! Check terminal for progress.")
    st.rerun()


def _handle_pause_training(exp_id: str):
    """Handle pause training button"""
    pause_training(exp_id)
    st.toast("Training paused")
    st.rerun()


def _handle_stop_training(exp_id: str):
    """Handle stop training button"""
    stop_training(exp_id)
    st.toast("Training stopped")
    st.rerun()


def _handle_view_results(exp_id: str):
    """Handle view results button"""
    st.session_state.selected_experiment_id = exp_id
    st.toast("Experiment selected. Navigate to Results page to view details.")


def _get_experiments_with_live_updates() -> list:
    """Get experiments, reloading from file if any are training.

    During training, the background thread writes updates to the JSON file,
    not to st.session_state. This function ensures we read fresh data.
    """
    experiments = get_experiments()

    # Check if any training is active
    has_active = any(exp.get("status") in ("training", "paused") for exp in experiments)

    if has_active:
        # Reload from file to get latest metrics from background thread
        session_id = get_session_id()
        if session_id:
            session_data = read_session_data(session_id)
            if session_data:
                file_experiments = session_data.get("experiments", [])
                # Update session state with file data for consistency
                st.session_state.experiments = file_experiments
                return file_experiments

    return experiments
