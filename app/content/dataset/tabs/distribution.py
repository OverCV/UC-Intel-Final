"""
Dataset Tab 2: Class Distribution
Visualizes malware family distribution and allows class selection
"""

import plotly.graph_objects as go
import streamlit as st


def render(dataset_info):
    """Render class distribution visualizations with selection interface"""
    st.subheader("Class Distribution & Selection")

    if not dataset_info['train_samples']:
        st.warning("No training data found")
        return

    # Get all classes
    all_classes = sorted(dataset_info['train_samples'].keys())

    # Initialize session state for selected classes
    if "selected_classes" not in st.session_state:
        st.session_state.selected_classes = all_classes.copy()  # Select all by default

    # Class selection interface
    st.markdown("### Select Classes to Include")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_classes = all_classes.copy()
            st.rerun()

    with col2:
        if st.button("Deselect All", use_container_width=True):
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
                if dataset_info['train_samples'].get(cls, 0) >= threshold
            ]
            if st.button(f"Select classes with ≥{threshold} samples", use_container_width=True):
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
        total_train = sum(dataset_info['train_samples'].get(c, 0) for c in selected)
        total_val = sum(dataset_info['val_samples'].get(c, 0) for c in selected)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Selected Classes", len(selected))
        with col2:
            st.metric("Training Samples", f"{total_train:,}")
        with col3:
            st.metric("Validation Samples", f"{total_val:,}")
        with col4:
            st.metric("Total Samples", f"{total_train + total_val:,}")
    else:
        st.warning("⚠️ No classes selected! Please select at least one class.")

    st.divider()

    # Visualization of distribution
    st.markdown("### Distribution Visualization")

    # Use selected classes for visualization
    display_classes = selected if selected else all_classes
    train_counts = [dataset_info['train_samples'].get(c, 0) for c in display_classes]
    val_counts = [dataset_info['val_samples'].get(c, 0) for c in display_classes]

    # Grouped bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Training',
        x=display_classes,
        y=train_counts,
        marker_color='#98c127',
        text=train_counts,
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Validation',
        x=display_classes,
        y=val_counts,
        marker_color='#8fd7d7',
        text=val_counts,
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Samples per Malware Family ({len(selected)} classes selected)",
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

    st.plotly_chart(fig, use_container_width=True)

    # Class imbalance detection
    if selected and len(selected) > 1:
        selected_train_counts = [dataset_info['train_samples'].get(c, 0) for c in selected]
        max_samples = max(selected_train_counts)
        min_samples = min(selected_train_counts) if min(selected_train_counts) > 0 else 1
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

        selected_train_items = [(c, dataset_info['train_samples'].get(c, 0)) for c in selected]

        with col1:
            st.markdown("**Most Common Selected Classes**")
            top_5 = sorted(selected_train_items, key=lambda x: x[1], reverse=True)[:5]
            for cls, count in top_5:
                st.text(f"{cls}: {count:,} samples")

        with col2:
            st.markdown("**Least Common Selected Classes**")
            bottom_5 = sorted(selected_train_items, key=lambda x: x[1])[:5]
            for cls, count in bottom_5:
                st.text(f"{cls}: {count:,} samples")