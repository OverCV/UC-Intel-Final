"""
Utility Functions
Helper functions for GPU, memory, and session management
"""

import torch

from state.cache import clear_cache_state
from state.ui import clear_ui_state
from state.workflow import clear_workflow_state


def check_gpu_available() -> bool:
    """Check if GPU is available"""
    try:
        return torch.cuda.is_available()
    except:
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
    except:
        return "N/A"


def clear_session():
    """Clear all session state (workflow + cache, preserving UI preferences)"""
    clear_workflow_state()
    clear_cache_state()
    # Note: Intentionally NOT clearing UI state (theme) to preserve user preferences
