"""
Dataset Tab 2: Class Distribution
Visualizes malware family distribution across train/val sets
"""

import plotly.graph_objects as go
import streamlit as st


def render(dataset_info):
    """Render class distribution visualizations"""
    st.subheader("Class Distribution")

    if not dataset_info['train_samples']:
        st.warning("No training data found")
        return

    classes = sorted(dataset_info['train_samples'].keys())
    train_counts = [dataset_info['train_samples'].get(c, 0) for c in classes]
    val_counts = [dataset_info['val_samples'].get(c, 0) for c in classes]

    # Grouped bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Training',
        x=classes,
        y=train_counts,
        marker_color='#98c127'
    ))
    fig.add_trace(go.Bar(
        name='Validation',
        x=classes,
        y=val_counts,
        marker_color='#8fd7d7'
    ))

    fig.update_layout(
        title="Samples per Malware Family",
        xaxis_title="Malware Family",
        yaxis_title="Number of Samples",
        barmode='group',
        height=500,
        xaxis={'tickangle': -45},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa')
    )

    st.plotly_chart(fig, width='stretch')

    # Top and bottom classes
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most Common Classes**")
        top_5 = sorted(dataset_info['train_samples'].items(),
                       key=lambda x: x[1], reverse=True)[:5]
        for cls, count in top_5:
            st.text(f"{cls}: {count:,} samples")

    with col2:
        st.markdown("**Least Common Classes**")
        bottom_5 = sorted(dataset_info['train_samples'].items(),
                          key=lambda x: x[1])[:5]
        for cls, count in bottom_5:
            st.text(f"{cls}: {count:,} samples")
