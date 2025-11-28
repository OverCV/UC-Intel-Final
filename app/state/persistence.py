"""
Session Persistence
Save and load training sessions to/from disk
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import streamlit as st

# Sessions storage directory
SESSIONS_DIR = Path(__file__).parent.parent.parent / "sessions"


def get_sessions_directory() -> Path:
    """Get the sessions storage directory, creating it if needed"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def save_session(session_id: str) -> bool:
    """
    Save current session state to disk

    Args:
        session_id: Session identifier to save

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        sessions_dir = get_sessions_directory()
        filepath = sessions_dir / f"{session_id}.json"

        # Gather workflow state
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "dataset_config": st.session_state.get("dataset_config", {}),
            # Libraries
            "model_library": st.session_state.get("model_library", []),
            "training_library": st.session_state.get("training_library", []),
            # Experiments
            "experiments": st.session_state.get("experiments", []),
            # Legacy (backwards compatibility)
            "model_config": st.session_state.get("model_config", {}),
            "training_config": st.session_state.get("training_config", {}),
            "training_active": st.session_state.get("training_active", False),
            "monitor_config": st.session_state.get("monitor_config", {}),
            "results": st.session_state.get("results"),
        }

        # Write to file
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

        return True

    except Exception as e:
        st.error(f"Failed to save session: {e}")
        return False


def load_session(session_id: str) -> bool:
    """
    Load session state from disk into st.session_state

    Args:
        session_id: Session identifier to load

    Returns:
        True if loaded successfully, False otherwise
    """
    try:
        sessions_dir = get_sessions_directory()
        filepath = sessions_dir / f"{session_id}.json"

        if not filepath.exists():
            st.error(f"Session file not found: {session_id}")
            return False

        # Read from file
        with open(filepath) as f:
            session_data = json.load(f)

        # Restore workflow state
        st.session_state.session_id = session_data.get("session_id", session_id)
        st.session_state.dataset_config = session_data.get("dataset_config", {})
        # Libraries
        st.session_state.model_library = session_data.get("model_library", [])
        st.session_state.training_library = session_data.get("training_library", [])
        # Experiments
        st.session_state.experiments = session_data.get("experiments", [])
        # Legacy (backwards compatibility)
        st.session_state.model_config = session_data.get("model_config", {})
        st.session_state.training_config = session_data.get("training_config", {})
        st.session_state.training_active = session_data.get("training_active", False)
        st.session_state.monitor_config = session_data.get("monitor_config", {})
        st.session_state.results = session_data.get("results")

        return True

    except Exception as e:
        st.error(f"Failed to load session: {e}")
        return False


def list_saved_sessions() -> list[str]:
    """
    Get list of all saved session IDs

    Returns:
        List of session IDs sorted by modification time (newest first)
    """
    try:
        sessions_dir = get_sessions_directory()

        # Find all .json files
        session_files = list(sessions_dir.glob("*.json"))

        # Sort by modification time (newest first)
        session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Extract session IDs (filename without .json)
        session_ids = [f.stem for f in session_files]

        return session_ids

    except Exception as e:
        st.error(f"Failed to list sessions: {e}")
        return []


def delete_session(session_id: str) -> bool:
    """
    Delete a saved session file

    Args:
        session_id: Session identifier to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        sessions_dir = get_sessions_directory()
        filepath = sessions_dir / f"{session_id}.json"

        if filepath.exists():
            filepath.unlink()
            return True
        else:
            st.warning(f"Session file not found: {session_id}")
            return False

    except Exception as e:
        st.error(f"Failed to delete session: {e}")
        return False


def get_session_metadata(session_id: str) -> dict[str, Any] | None:
    """
    Get metadata for a session without loading full state

    Args:
        session_id: Session identifier

    Returns:
        Dictionary with metadata or None if not found
    """
    try:
        sessions_dir = get_sessions_directory()
        filepath = sessions_dir / f"{session_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        return {
            "session_id": data.get("session_id"),
            "created_at": data.get("created_at"),
            "has_dataset": bool(data.get("dataset_config")),
            "model_count": len(data.get("model_library", [])),
            "training_count": len(data.get("training_library", [])),
            "experiment_count": len(data.get("experiments", [])),
            # Legacy
            "has_model": bool(data.get("model_config")),
            "has_training": bool(data.get("training_config")),
            "has_results": data.get("results") is not None,
        }

    except Exception as exception:
        st.error(f"Failed to get session metadata: {exception}")
        return None
