# Malware Classification Streamlit App

## BFS Implementation - Level 1 Complete

This is the initial interface structure. All pages are functional skeletons showing layout and navigation. No deep implementation yet.

## Structure

```
app/
├── main.py                  # Main entry point (single-page app with session state navigation)
├── config.py                # Shared constants
├── views/                   # View modules (renamed from pages to avoid Streamlit auto-routing)
│   ├── __init__.py
│   ├── home.py             # Page 1: Home & Setup
│   ├── dataset.py          # Page 2: Dataset Configuration
│   ├── model.py            # Page 3: Model Configuration
│   ├── training.py         # Page 4: Training Configuration
│   ├── monitor.py          # Page 5: Training Monitor
│   ├── results.py          # Page 6: Results & Evaluation
│   └── interpretability.py # Page 7: Model Interpretability
└── utils/                  # Utility modules
    ├── __init__.py
    └── session_state.py    # Session state management
```

## Navigation System

**Single-page app** with session state navigation:
- NO file-based routing (URL stays at `localhost:8501`)
- Selectbox in sidebar for page navigation
- Current page stored in `st.session_state.current_page`
- Views stored in `views/` (not `pages/`) to avoid Streamlit's auto-discovery

## Running the App

```bash
# From project root
cd app
streamlit run main.py
```

## What Works Now

- ✅ Single navigation system (selectbox in sidebar)
- ✅ No URL routing conflicts
- ✅ Page structure and section headers
- ✅ UI elements (buttons, sliders, inputs)
- ✅ Session state initialization
- ✅ Configuration constants

## What's NOT Implemented Yet (marked with TODO)

- Dataset loading from `repo/malware`
- Model architecture building
- Training execution
- Results visualization (Plotly charts)
- GPU detection
- File storage/loading
- Mermaid diagrams
- All actual computation

## Next Steps

Review the interface and layout. Once approved, we'll implement depth-first:
1. Dataset loading and visualization
2. Model builder backend
3. Training pipeline
4. Results generation
5. Interpretability tools
