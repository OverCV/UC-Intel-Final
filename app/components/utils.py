"""
Utility Functions
Helper functions for GPU, memory, and session management
"""

from state.cache import clear_cache_state
from state.persistence import save_session
from state.workflow import clear_workflow_state, get_session_id
import streamlit as st
import torch


def check_gpu_available() -> bool:
    """Check if GPU is available"""
    try:
        return torch.cuda.is_available()
    except Exception as exception:
        st.error(f"Failed to check GPU availability: {exception}")
        return False


def get_memory_info() -> str:
    """Get memory info string"""
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            return f"{allocated:.1f}/{total:.1f}GB"
        else:
            return "N/A"
    except Exception as exception:
        st.error(f"Failed to get memory info: {exception}")
        return "N/A"


def clear_session():
    """
    Clear all session state (workflow + cache, preserving UI preferences)
    Saves current session to disk before clearing
    """
    # Save current session before clearing
    current_session_id = get_session_id()
    if current_session_id:
        save_session(current_session_id)

    # Clear state
    clear_workflow_state()
    clear_cache_state()
    # Note: Intentionally NOT clearing UI state (theme) to preserve user preferences
