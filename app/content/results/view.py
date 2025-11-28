"""
Results & Evaluation Page
Display training results, metrics, and visualizations for completed experiments
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from state.workflow import (
    get_experiments,
    get_model_from_library,
    get_training_from_library,
)


def render():
    """Main render function for Results page"""
    st.title("Results & Evaluation")

    # Get completed experiments
    experiments = get_experiments()
    completed = [exp for exp in experiments if exp.get("status") == "completed"]

    if not completed:
        st.info("No completed experiments yet. Train a model first.")
        return

    # Experiment selector
    selected_exp = _render_experiment_selector(completed)
    if not selected_exp:
        return

    st.divider()

    # Render sections
    _render_experiment_summary(selected_exp)
    _render_test_metrics(selected_exp)
    _render_training_history(selected_exp)

    # Placeholder sections for future implementation
    _render_confusion_matrix_placeholder()
    _render_classification_report_placeholder()
    _render_export_placeholder(selected_exp)


def _render_experiment_selector(completed: list) -> dict | None:
    """Render experiment selector dropdown"""
    exp_options = {exp["id"]: exp.get("name", exp["id"]) for exp in completed}

    selected_id = st.selectbox(
        "Select Experiment",
        options=list(exp_options.keys()),
        format_func=lambda x: exp_options.get(x, x),
    )

    for exp in completed:
        if exp["id"] == selected_id:
            return exp

    return None


def _render_experiment_summary(experiment: dict):
    """Render experiment metadata summary"""
    st.header("Experiment Summary")

    # Get model and training config names
    model_entry = get_model_from_library(experiment.get("model_id"))
    training_entry = get_training_from_library(experiment.get("training_id"))

    model_name = model_entry.get("name", "Unknown") if model_entry else "Unknown"
    model_type = model_entry.get("model_type", "") if model_entry else ""
    training_name = training_entry.get("name", "Unknown") if training_entry else "Unknown"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", model_name)
        st.caption(f"Type: {model_type}")

    with col2:
        st.metric("Training Config", training_name)
        st.caption(f"ID: {experiment.get('training_id', 'N/A')}")

    with col3:
        st.metric("Duration", experiment.get("duration", "N/A"))
        st.caption(f"Best Epoch: {experiment.get('best_epoch', 'N/A')}/{experiment.get('current_epoch', 'N/A')}")


def _render_test_metrics(experiment: dict):
    """Render final test metrics"""
    st.header("Final Performance")

    metrics = experiment.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        train_loss = metrics.get("train_loss", 0)
        st.metric("Train Loss", f"{train_loss:.4f}")

    with col2:
        train_acc = metrics.get("train_acc", 0)
        st.metric("Train Accuracy", f"{train_acc*100:.2f}%")

    with col3:
        val_loss = metrics.get("val_loss", 0)
        st.metric("Val Loss", f"{val_loss:.4f}")

    with col4:
        val_acc = metrics.get("val_acc", 0)
        st.metric("Val Accuracy", f"{val_acc*100:.2f}%")


def _render_training_history(experiment: dict):
    """Render training history charts"""
    st.header("Training History")

    history = experiment.get("history", {})

    if not history:
        st.warning("No training history available for this experiment.")
        return

    # Prepare data
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if not epochs:
        st.warning("Training history is empty.")
        return

    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Loss", "Accuracy", "Learning Rate"])

    with tab1:
        _render_loss_chart(epochs, history)

    with tab2:
        _render_accuracy_chart(epochs, history)

    with tab3:
        _render_lr_chart(epochs, history)


def _render_loss_chart(epochs: list, history: dict):
    """Render loss curves"""
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        st.info("No loss data available.")
        return

    df = pd.DataFrame({
        "Epoch": epochs * 2,
        "Loss": train_loss + val_loss,
        "Type": ["Train"] * len(train_loss) + ["Validation"] * len(val_loss),
    })

    fig = px.line(
        df,
        x="Epoch",
        y="Loss",
        color="Type",
        title="Loss Curves",
        color_discrete_map={"Train": "#636EFA", "Validation": "#EF553B"},
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_accuracy_chart(epochs: list, history: dict):
    """Render accuracy curves"""
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    if not train_acc:
        st.info("No accuracy data available.")
        return

    # Convert to percentages
    train_acc_pct = [acc * 100 for acc in train_acc]
    val_acc_pct = [acc * 100 for acc in val_acc]

    df = pd.DataFrame({
        "Epoch": epochs * 2,
        "Accuracy (%)": train_acc_pct + val_acc_pct,
        "Type": ["Train"] * len(train_acc_pct) + ["Validation"] * len(val_acc_pct),
    })

    fig = px.line(
        df,
        x="Epoch",
        y="Accuracy (%)",
        color="Type",
        title="Accuracy Curves",
        color_discrete_map={"Train": "#636EFA", "Validation": "#EF553B"},
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        legend_title="",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_lr_chart(epochs: list, history: dict):
    """Render learning rate schedule"""
    lr_history = history.get("lr", [])

    if not lr_history:
        st.info("No learning rate data available.")
        return

    df = pd.DataFrame({
        "Epoch": epochs,
        "Learning Rate": lr_history,
    })

    fig = px.line(
        df,
        x="Epoch",
        y="Learning Rate",
        title="Learning Rate Schedule",
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        hovermode="x unified",
    )
    fig.update_traces(line_color="#00CC96")

    st.plotly_chart(fig, use_container_width=True)


def _render_confusion_matrix_placeholder():
    """Placeholder for confusion matrix - requires test set inference"""
    st.header("Confusion Matrix")
    st.info(
        "Confusion matrix requires running inference on the test set. "
        "This will be available after implementing test evaluation."
    )


def _render_classification_report_placeholder():
    """Placeholder for classification report"""
    st.header("Classification Report")
    st.info(
        "Classification report (precision, recall, F1 per class) requires test predictions. "
        "Coming soon."
    )


def _render_export_placeholder(experiment: dict):
    """Placeholder for export functionality"""
    st.header("Export")

    col1, col2 = st.columns(2)

    with col1:
        # Export history as CSV
        history = experiment.get("history", {})
        if history:
            df = pd.DataFrame(history)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Training History (CSV)",
                data=csv,
                file_name=f"{experiment['id']}_history.csv",
                mime="text/csv",
            )
        else:
            st.button("Download Training History (CSV)", disabled=True)

    with col2:
        st.button("Download Model (.pt)", disabled=True, help="Coming soon")
