"""
Evaluation visualization components for the Enterprise LLM Trust Framework dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import sys
import os
from typing import Optional

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.logging import get_logger

logger = get_logger(__name__)


def render_trust_score_gauge(trust_score: float, title: str = "Trust Score"):
    """
    Render a gauge chart for trust score.
    
    Args:
        trust_score: Trust score value (0.0 to 1.0)
        title: Title for the gauge
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = trust_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.8},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.4], 'color': "lightgray"},
                {'range': [0.4, 0.7], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9}}))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)


def render_evaluation_radar_chart(evaluation_results: dict, title: str = "Evaluation Profile"):
    """
    Render a radar chart for evaluation results.
    
    Args:
        evaluation_results: Dictionary mapping metric names to scores
        title: Title for the chart
    """
    if not evaluation_results:
        st.info("No evaluation data available for radar chart.")
        return
    
    categories = list(evaluation_results.keys())
    values = list(evaluation_results.values())
    
    # Close the plot by repeating the first value
    categories += [categories[0]]
    values += [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Evaluation Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_evaluation_heatmap(evaluation_data: pd.DataFrame, title: str = "Evaluation Heatmap"):
    """
    Render a heatmap for evaluation results across multiple prompts or runs.
    
    Args:
        evaluation_data: DataFrame with metrics as columns and runs/prompts as rows
        title: Title for the heatmap
    """
    if evaluation_data.empty:
        st.info("No data available for evaluation heatmap.")
        return
    
    fig = px.imshow(
        evaluation_data.values,
        labels=dict(x="Metrics", y="Runs/Prompts", color="Score"),
        x=evaluation_data.columns,
        y=evaluation_data.index,
        color_continuous_scale="RdYlGn",
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_bar_chart(comparison_data: pd.DataFrame, 
                              x_col: str, 
                              y_col: str, 
                              color_col: Optional[str] = None,
                              title: str = "Comparison Chart"):
    """
    Render a bar chart for comparing LLMs or configurations.
    
    Args:
        comparison_data: DataFrame with comparison data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Optional column name for color grouping
        title: Title for the chart
    """
    if comparison_data.empty:
        st.info("No data available for comparison chart.")
        return
    
    if color_col and color_col in comparison_data.columns:
        fig = px.bar(
            comparison_data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            barmode='group'
        )
    else:
        fig = px.bar(
            comparison_data,
            x=x_col,
            y=y_col,
            title=title
        )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_time_series_chart(time_series_data: pd.DataFrame,
                           time_col: str,
                           value_col: str,
                           title: str = "Time Series Chart"):
    """
    Render a time series chart.
    
    Args:
        time_series_data: DataFrame with time series data
        time_col: Column name for time values
        value_col: Column name for values to plot
        title: Title for the chart
    """
    if time_series_data.empty:
        st.info("No data available for time series chart.")
        return
    
    # Ensure time column is datetime
    if time_col in time_series_data.columns:
        time_series_data = time_series_data.copy()
        time_series_data[time_col] = pd.to_datetime(time_series_data[time_col])
    
    fig = px.line(
        time_series_data,
        x=time_col,
        y=value_col,
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_metric_distribution(metric_data: pd.Series,
                             metric_name: str = "Metric",
                             title: str = "Metric Distribution"):
    """
    Render a distribution plot for a metric.
    
    Args:
        metric_data: Series containing metric values
        metric_name: Name of the metric
        title: Title for the chart
    """
    if metric_data.empty:
        st.info(f"No data available for {metric_name} distribution.")
        return
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{metric_name} Distribution", f"{metric_name} Box Plot"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=metric_data, name="Distribution"),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=metric_data, name="Box Plot"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text=title)
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(correlation_matrix: pd.DataFrame,
                             title: str = "Metric Correlation Heatmap"):
    """
    Render a correlation heatmap for metrics.
    
    Args:
        correlation_matrix: DataFrame containing correlation values
        title: Title for the heatmap
    """
    if correlation_matrix.empty:
        st.info("No data available for correlation heatmap.")
        return
    
    fig = px.imshow(
        correlation_matrix.values,
        labels=dict(x="Metrics", y="Metrics", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pipeline_timeline(pipeline_steps: list, title: str = "Evaluation Pipeline Timeline"):
    """
    Render a timeline showing the steps in the evaluation pipeline.
    
    Args:
        pipeline_steps: List of dictionaries with 'step', 'duration', and optionally 'status'
        title: Title for the chart
    """
    if not pipeline_steps:
        st.info("No pipeline steps available for timeline.")
        return
    
    # Prepare data for timeline
    df = pd.DataFrame(pipeline_steps)
    if df.empty:
        st.info("No pipeline steps available for timeline.")
        return
    
    # Calculate cumulative times for positioning
    df['start'] = df['duration'].cumsum().shift(1).fillna(0)
    df['end'] = df['duration'].cumsum()
    
    # Color by status if available
    if 'status' in df.columns:
        color_map = {'completed': 'green', 'running': 'blue', 'failed': 'red', 'pending': 'gray'}
        df['color'] = df['status'].map(lambda x: color_map.get(x, 'gray'))
    else:
        df['color'] = 'blue'
    
    fig = go.Figure()
    
    # Add bars for each step
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['duration']],
            y=[row['step']],
            base=[row['start']],
            orientation='h',
            marker_color=row['color'],
            name=row['step'],
            showlegend=False,
            hovertext=f"Duration: {row['duration']:.2f}s<br>Status: {row.get('status', 'N/A')}"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Pipeline Step",
        barmode='stack',
        height=max(200, len(pipeline_steps) * 40 + 50)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_model_radar_comparison(model_comparison_data: dict, title: str = "Model Comparison Radar"):
    """
    Render a radar chart comparing multiple models across evaluation dimensions.
    
    Args:
        model_comparison_data: Dictionary mapping model names to their evaluation scores
        title: Title for the chart
    """
    if not model_comparison_data:
        st.info("No model comparison data available.")
        return
    
    # Get all unique evaluation dimensions
    all_dimensions = set()
    for model_scores in model_comparison_data.values():
        all_dimensions.update(model_scores.keys())
    
    all_dimensions = sorted(list(all_dimensions))
    
    fig = go.Figure()
    
    for model_name, scores in model_comparison_data.items():
        # Prepare values for radar chart (close the loop)
        values = [scores.get(dim, 0) for dim in all_dimensions]
        values += [values[0]]  # Close the loop
        dimensions = all_dimensions + [all_dimensions[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=dimensions,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Export functions for easy importing
__all__ = [
    'render_trust_score_gauge',
    'render_evaluation_radar_chart',
    'render_evaluation_heatmap',
    'render_comparison_bar_chart',
    'render_time_series_chart',
    'render_metric_distribution',
    'render_correlation_heatmap',
    'render_pipeline_timeline',
    'render_model_radar_comparison'
]