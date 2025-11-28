"""
Dataset Tab 2: Class Distribution
Visualizes malware family distribution and allows class selection
"""

import plotly.graph_objects as go
import streamlit as st


def render(dataset_info):
    """Render class distribution visualizations with selection interface"""
    st.subheader("Class Distribution & Selection")

    if not dataset_info['samples']:
        st.warning("No data found")
        return

    # Get all classes
    all_classes = sorted(dataset_info['samples'].keys())

    # Initialize session state for selected classes
    if "selected_classes" not in st.session_state:
        st.session_state.selected_classes = all_classes.copy()  # Select all by default

    # Class selection interface
    st.markdown("### Select Classes to Include")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Select All", width="stretch"):
            st.session_state.selected_classes = all_classes.copy()
            st.rerun()

    with col2:
        if st.button("Deselect All", width="stretch"):
            st.session_state.selected_classes = []
            st.rerun()

    with col3:
        # Quick filter for large/small classes
        threshold = st.number_input(
            "Min samples per class",
            min_value=0,
            max_value=1000,
            value=0,
            help="Filter classes with fewer samples than this threshold"
        )

        if threshold > 0:
            filtered_classes = [
                cls for cls in all_classes
                if dataset_info['samples'].get(cls, 0) >= threshold
            ]
            if st.button(f"Select classes with ≥{threshold} samples", width="stretch"):
                st.session_state.selected_classes = filtered_classes
                st.rerun()

    # Multi-select for individual class selection
    selected = st.multiselect(
        "Selected Classes (you can search by typing)",
        options=all_classes,
        default=st.session_state.selected_classes,
        key="class_selector",
        help="Select which malware families to include in your dataset"
    )

    # Update session state
    st.session_state.selected_classes = selected

    # Show selection summary
    if selected:
        total_samples = sum(dataset_info['samples'].get(c, 0) for c in selected)
        
        # Get split info
        use_cross_validation = st.session_state.get("use_cross_validation", False)
        
        if use_cross_validation:
            n_folds = st.session_state.get("n_folds", 5)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Classes", len(selected))
            with col2:
                st.metric("Total Samples", f"{total_samples:,}")
            with col3:
                st.metric("K-Folds", n_folds)
        else:
            from utils.dataset_utils import calculate_split_percentages
            
            train_pct = st.session_state.get("train_split", 70)
            val_of_remaining = st.session_state.get("val_split", 50)
            train_final, val_final, test_final = calculate_split_percentages(train_pct, val_of_remaining)
            
            train_samples = int(total_samples * train_final / 100)
            val_samples = int(total_samples * val_final / 100)
            test_samples = int(total_samples * test_final / 100)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Selected Classes", len(selected))
            with col2:
                st.metric("Total Samples", f"{total_samples:,}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Train ({train_final:.1f}%)", f"{train_samples:,}")
            with col2:
                st.metric(f"Val ({val_final:.1f}%)", f"{val_samples:,}")
            with col3:
                st.metric(f"Test ({test_final:.1f}%)", f"{test_samples:,}")
    else:
        st.warning("⚠️ No classes selected! Please select at least one class.")

    st.divider()

    # Visualization of distribution
    st.markdown("### Distribution Visualization")

    # Use selected classes for visualization
    display_classes = selected if selected else all_classes
    sample_counts = [dataset_info['samples'].get(c, 0) for c in display_classes]
    
    # Get split configuration from session state
    use_cross_validation = st.session_state.get("use_cross_validation", False)
    
    if use_cross_validation:
        # Cross-Validation: Show total samples per class
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Total Samples (used in CV)',
            x=display_classes,
            y=sample_counts,
            marker_color='#98c127',
            text=sample_counts,
            textposition='outside'
        ))
        
        n_folds = st.session_state.get("n_folds", 5)
        title_text = f"Samples per Malware Family - {n_folds}-Fold Cross-Validation"
    else:
        # Fixed split: Show Train/Val/Test breakdown
        from utils.dataset_utils import calculate_split_percentages
        
        train_pct = st.session_state.get("train_split", 70)
        val_of_remaining = st.session_state.get("val_split", 50)
        train_final, val_final, test_final = calculate_split_percentages(train_pct, val_of_remaining)
        
        # Calculate samples per class for each split
        train_counts = [int(dataset_info['samples'].get(c, 0) * train_final / 100) for c in display_classes]
        val_counts = [int(dataset_info['samples'].get(c, 0) * val_final / 100) for c in display_classes]
        test_counts = [int(dataset_info['samples'].get(c, 0) * test_final / 100) for c in display_classes]
        
        # Grouped bar chart with train/val/test
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=f'Train ({train_final:.1f}%)',
            x=display_classes,
            y=train_counts,
            marker_color='#98c127',
            text=train_counts,
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name=f'Validation ({val_final:.1f}%)',
            x=display_classes,
            y=val_counts,
            marker_color='#8fd7d7',
            text=val_counts,
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name=f'Test ({test_final:.1f}%)',
            x=display_classes,
            y=test_counts,
            marker_color='#ffb255',
            text=test_counts,
            textposition='outside'
        ))
        
        title_text = f"Samples per Malware Family - Train/Val/Test Split"

    fig.update_layout(
        title=title_text,
        xaxis_title="Malware Family",
        yaxis_title="Number of Samples",
        barmode='group',
        height=500,
        xaxis={'tickangle': -45},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        showlegend=True
    )

    st.plotly_chart(fig, width="stretch")

    # Class imbalance detection
    if selected and len(selected) > 1:
        selected_counts = [dataset_info['samples'].get(c, 0) for c in selected]
        max_samples = max(selected_counts)
        min_samples = min(selected_counts) if min(selected_counts) > 0 else 1
        imbalance_ratio = max_samples / min_samples

        if imbalance_ratio > 10:
            st.error(f"⚠️ **Severe Class Imbalance Detected!** Ratio: {imbalance_ratio:.1f}:1")
            st.markdown("""
            **Recommended actions:**
            - Use class weights during training (Auto Class Weights)
            - Consider SMOTE or other oversampling techniques
            - Apply stratified sampling for train/val/test splits
            """)
        elif imbalance_ratio > 3:
            st.warning(f"⚠️ **Moderate Class Imbalance** Ratio: {imbalance_ratio:.1f}:1")
            st.markdown("Consider using class weights during training.")

    # Top and bottom classes (from selected)
    if selected:
        col1, col2 = st.columns(2)

        selected_items = [(c, dataset_info['samples'].get(c, 0)) for c in selected]

        with col1:
            st.markdown("**Most Common Selected Classes**")
            top_5 = sorted(selected_items, key=lambda x: x[1], reverse=True)[:5]
            for cls, count in top_5:
                st.text(f"{cls}: {count:,} samples")

        with col2:
            st.markdown("**Least Common Selected Classes**")
            bottom_5 = sorted(selected_items, key=lambda x: x[1])[:5]
            for cls, count in bottom_5:
                st.text(f"{cls}: {count:,} samples")