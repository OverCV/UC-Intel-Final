"""
Workflow State Management
Core ML workflow configuration and session state
"""

import streamlit as st
from datetime import datetime
from typing import Any, Dict, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """Type definition for workflow state fields"""
    session_id: str
    dataset_config: Dict[str, Any]
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    training_active: bool
    monitor_config: Dict[str, Any]
    results: Optional[Dict[str, Any]]


def init_workflow_state() -> None:
    """Initialize workflow state with default values"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()

    if 'dataset_config' not in st.session_state:
        st.session_state.dataset_config = {}

    if 'model_config' not in st.session_state:
        st.session_state.model_config = {}

    if 'training_config' not in st.session_state:
        st.session_state.training_config = {}

    if 'training_active' not in st.session_state:
        st.session_state.training_active = False

    if 'monitor_config' not in st.session_state:
        st.session_state.monitor_config = {}

    if 'results' not in st.session_state:
        st.session_state.results = None


def generate_session_id() -> str:
    """Generate unique session ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}"


# Session ID
def get_session_id() -> str:
    """Get current session ID"""
    return st.session_state.get('session_id', '')


# Dataset Configuration
def save_dataset_config(config: Dict[str, Any]) -> None:
    """Save dataset configuration to session state"""
    st.session_state.dataset_config = config


def get_dataset_config() -> Dict[str, Any]:
    """Retrieve dataset configuration"""
    return st.session_state.get('dataset_config', {})


def has_dataset_config() -> bool:
    """Check if dataset is configured"""
    return bool(st.session_state.get('dataset_config'))


# Model Configuration
def save_model_config(config: Dict[str, Any]) -> None:
    """Save model configuration to session state"""
    st.session_state.model_config = config


def get_model_config() -> Dict[str, Any]:
    """Retrieve model configuration"""
    return st.session_state.get('model_config', {})


def has_model_config() -> bool:
    """Check if model is configured"""
    return bool(st.session_state.get('model_config'))


# Training Configuration
def save_training_config(config: Dict[str, Any]) -> None:
    """Save training configuration to session state"""
    st.session_state.training_config = config


def get_training_config() -> Dict[str, Any]:
    """Retrieve training configuration"""
    return st.session_state.get('training_config', {})


def has_training_config() -> bool:
    """Check if training is configured"""
    return bool(st.session_state.get('training_config'))


# Training Status
def is_training_active() -> bool:
    """Check if training is currently active"""
    return st.session_state.get('training_active', False)


def set_training_active(active: bool) -> None:
    """Set training active status"""
    st.session_state.training_active = active


# Results
def save_results(results: Dict[str, Any]) -> None:
    """Save training results to session state"""
    st.session_state.results = results


def get_results() -> Optional[Dict[str, Any]]:
    """Retrieve training results"""
    return st.session_state.get('results')


def has_results() -> bool:
    """Check if results are available"""
    return st.session_state.get('results') is not None


# Session Management
def clear_workflow_state() -> None:
    """Clear workflow-related session state"""
    keys_to_clear = [
        'session_id',
        'dataset_config',
        'model_config',
        'training_config',
        'training_active',
        'monitor_config',
        'results',
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Reinitialize with fresh values
    init_workflow_state()
