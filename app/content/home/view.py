"""
Page 1: Home & Setup
Project overview and quick start guide
"""

from components.utils import check_gpu_available
import streamlit as st
import streamlit.components.v1 as components


def render():
    """Main render function for Home page"""
    st.title("Malware Classification with Deep Learning")

    # Section 1: Project Overview
    render_project_overview()

    # Section 2: System Status & Quick Actions
    render_quick_start()


def render_project_overview():
    """Section 1: What this app does"""
    st.header("Project Overview")

    with st.expander("What does this app do?"):
        st.markdown("""
        - Load combined malware dataset
        - Configure CNN architectures
        - Train models with different hyperparameters
        - Evaluate and visualize results
        """)

    # Workflow diagram (rendered via Mermaid JS)
    mermaid_code = """
    flowchart LR
        D[Dataset] --> M[Model Library]
        M --> T[Training Library]
        T --> Mo[Monitor]
        Mo --> R[Results]

        D -.- D1[Configure data]
        M -.- M1[Save models]
        T -.- T1[Save configs]
        Mo -.- Mo1[Run training]
        R -.- R1[Analyze]
    """

    components.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true, theme:'neutral'}});</script>
        <div class="mermaid">{mermaid_code}</div>
        """,
        height=400,
    )


def render_quick_start():
    """Section 2: System status and workflow guide"""
    st.header("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "Combined Malware")
    with col2:
        st.metric("Available Models", "5")
    with col3:
        gpu_status = "Available" if check_gpu_available() else "Not Available"
        st.metric("GPU Status", gpu_status)

    st.divider()

    st.header("Workflow")
    st.markdown("""
    **To start a new training session:**
    1. Create a new session using the **"New Session"** button in the header
    2. Navigate to **Dataset** page to configure your data
    3. Go to **Model** page to design your architecture
    4. Configure training parameters in **Training** page
    5. Monitor your training in **Monitor** page
    6. View results in **Results** and **Interpretability** pages

    **To resume a previous session:**
    - Use the **"Past Sessions"** dropdown in the header to load a previous session
    """)
