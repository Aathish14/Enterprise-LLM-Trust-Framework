"""
Analytics reports for the Enterprise LLM Trust Framework dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from observation.mlflow_integration import mlflow_tracker
from config.logging import get_logger

logger = get_logger(__name__)


def render_analytics_reports():
    """Render analytics reports section."""
    st.header("📊 Analytics Reports")
    
    # Report type selector
    report_type = st.selectbox(
        "Select Report Type",
        [
            "Evaluation Trends",
            "Model Performance Comparison",
            "Trust Score Distribution",
            "Latency Analysis",
            "Failure Analysis",
            "Experiment Success Rates"
        ]
    )
    
    if report_type == "Evaluation Trends":
        render_evaluation_trends_report()
    elif report_type == "Model Performance Comparison":
        render_model_performance_report()
    elif report_type == "Trust Score Distribution":
        render_trust_score_distribution_report()
    elif report_type == "Latency Analysis":
        render_latency_analysis_report()
    elif report_type == "Failure Analysis":
        render_failure_analysis_report()
    elif report_type == "Experiment Success Rates":
        render_experiment_success_report()


def render_evaluation_trends_report():
    """Render evaluation trends report."""
    st.subheader("📈 Evaluation Trends Over Time")
    
    try:
        # Get experiment data
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No data available for evaluation trends.")
            return
        
        # Convert timestamps
        experiments['start_time'] = pd.to_datetime(experiments['start_time'], unit='ms')
        experiments['end_time'] = pd.to_datetime(experiments['end_time'], unit='ms')
        experiments['duration_seconds'] = experiments['duration'] / 1000
        
        # Filter finished experiments
        finished_experiments = experiments[experiments['status'] == 'FINISHED'].copy()
        
        if finished_experiments.empty:
            st.info("No finished experiments found.")
            return
        
        # Daily aggregation
        finished_experiments['date'] = finished_experiments['start_time'].dt.date
        daily_stats = finished_experiments.groupby('date').agg({
            'run_id': 'count',
            'duration_seconds': 'mean',
        }).reset_index()
        daily_stats.columns = ['date', 'experiment_count', 'avg_duration']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Experiment Count', 'Average Duration'),
            vertical_spacing=0.12
        )
        
        # Experiment count
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['experiment_count'],
                     mode='lines+markers', name='Experiments'),
            row=1, col=1
        )
        
        # Average duration
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['avg_duration'],
                     mode='lines+markers', name='Avg Duration (s)', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Experiments", len(finished_experiments))
        
        with col2:
            st.metric("Avg Daily Experiments", f"{daily_stats['experiment_count'].mean():.1f}")
        
        with col3:
            st.metric("Overall Avg Duration", f"{daily_stats['avg_duration'].mean():.2f}s")
    
    except Exception as e:
        st.error(f"Error generating evaluation trends report: {e}")
        logger.error(f"Evaluation trends report error: {e}")


def render_model_performance_report():
    """Render model performance comparison report."""
    st.subheader("🏆 Model Performance Comparison")
    
    try:
        # Get experiment data
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No data available for model performance comparison.")
            return
        
        # Extract model information from parameters or tags
        # This is simplified - in practice, we'd need to parse the run data properly
        st.info("Model performance comparison requires specific tagging of runs with model information.")
        st.write("Currently showing general experiment statistics:")
        
        # Basic statistics by experiment
        if 'experiment_id' in experiments.columns:
            experiment_stats = experiments.groupby('experiment_id').agg({
                'run_id': 'count',
                'duration': 'mean'
            }).reset_index()
            experiment_stats.columns = ['Experiment ID', 'Run Count', 'Avg Duration (ms)']
            experiment_stats['Avg Duration (s)'] = (experiment_stats['Avg Duration (ms)'] / 1000).round(2)
            
            st.dataframe(experiment_stats[['Experiment ID', 'Run Count', 'Avg Duration (s)']], 
                        use_container_width=True)
        else:
            st.info("No experiment ID data available.")
    
    except Exception as e:
        st.error(f"Error generating model performance report: {e}")
        logger.error(f"Model performance report error: {e}")


def render_trust_score_distribution_report():
    """Render trust score distribution report."""
    st.subheader("🎯 Trust Score Distribution")
    
    try:
        # Get experiment data with metrics
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No data available for trust score distribution.")
            return
        
        # Look for trust score metrics
        trust_score_cols = [col for col in experiments.columns if 'trust_score' in col.lower() and 'metric' in col.lower()]
        
        if not trust_score_cols:
            st.info("No trust score metrics found in the data.")
            st.write("Available metric columns:", [col for col in experiments.columns if 'metric' in col.lower()][:10])
            return
        
        # For simplicity, let's assume we have a trust_score_overall metric
        trust_score_col = 'trust_score_overall' if 'trust_score_overall' in experiments.columns else trust_score_cols[0]
        
        # Filter valid scores
        scores_df = experiments[[trust_score_col]].dropna()
        scores_df = scores_df[(scores_df[trust_score_col] >= 0) & (scores_df[trust_score_col] <= 1)]
        
        if scores_df.empty:
            st.info("No valid trust score data found.")
            return
        
        # Create histogram
        fig = px.histogram(
            scores_df, 
            x=trust_score_col,
            nbins=20,
            title='Trust Score Distribution',
            labels={trust_score_col: 'Trust Score', 'count': 'Frequency'}
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.subheader("Trust Score Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{scores_df[trust_score_col].mean():.3f}")
        
        with col2:
            st.metric("Median", f"{scores_df[trust_score_col].median():.3f}")
        
        with col3:
            st.metric("Std Dev", f"{scores_df[trust_score_col].std():.3f}")
        
        with col4:
            st.metric("Count", len(scores_df))
    
    except Exception as e:
        st.error(f"Error generating trust score distribution report: {e}")
        logger.error(f"Trust score distribution report error: {e}")


def render_latency_analysis_report():
    """Render latency analysis report."""
    st.subheader("⏱️ Latency Analysis")
    
    try:
        # Get experiment data
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No data available for latency analysis.")
            return
        
        # Look for duration or latency metrics
        duration_cols = [col for col in experiments.columns if 'duration' in col.lower() or 'latency' in col.lower()]
        
        if not duration_cols:
            st.info("No duration or latency metrics found in the data.")
            # Use the standard duration column if available
            if 'duration' in experiments.columns:
                duration_cols = ['duration']
            else:
                return
        
        duration_col = duration_cols[0]
        
        # Filter valid durations
        latency_df = experiments[[duration_col]].dropna()
        latency_df = latency_df[latency_df[duration_col] >= 0]
        
        if latency_df.empty:
            st.info("No valid latency data found.")
            return
        
        # Convert to seconds if needed
        if experiments[duration_col].median() > 1000:  # Likely in milliseconds
            latency_df[duration_col] = latency_df[duration_col] / 1000
            duration_label = "Latency (seconds)"
        else:
            duration_label = "Duration"
        
        # Create box plot
        fig = px.box(
            latency_df, 
            y=duration_col,
            title='Latency Distribution',
            labels={duration_col: duration_label}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.subheader("Latency Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{latency_df[duration_col].mean():.2f}")
        
        with col2:
            st.metric("Median", f"{latency_df[duration_col].median():.2f}")
        
        with col3:
            st.metric("P95", f"{latency_df[duration_col].quantile(0.95):.2f}")
        
        with col4:
            st.metric("P99", f"{latency_df[duration_col].quantile(0.99):.2f}")
    
    except Exception as e:
        st.error(f"Error generating latency analysis report: {e}")
        logger.error(f"Latency analysis report error: {e}")


def render_failure_analysis_report():
    """Render failure analysis report."""
    st.subheader("❌ Failure Analysis")
    
    try:
        # Get experiment data
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No data available for failure analysis.")
            return
        
        # Count failed vs finished experiments
        if 'status' in experiments.columns:
            status_counts = experiments['status'].value_counts()
            
            # Create pie chart
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Experiment Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show failure details if any
            failed_experiments = experiments[experiments['status'] != 'FINISHED']
            if not failed_experiments.empty:
                st.subheader("Failed Experiments")
                st.dataframe(
                    failed_experiments[['run_id', 'status', 'start_time']].head(10),
                    use_container_width=True
                )
            else:
                st.success("No failed experiments found!")
        else:
            st.info("No status data available for failure analysis.")
    
    except Exception as e:
        st.error(f"Error generating failure analysis report: {e}")
        logger.error(f"Failure analysis report error: {e}")


def render_experiment_success_report():
    """Render experiment success rates report."""
    st.subheader("✅ Experiment Success Rates")
    
    try:
        # Get experiment data
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No data available for experiment success rates.")
            return
        
        # Calculate success rate by experiment if experiment_id exists
        if 'experiment_id' in experiments.columns and 'status' in experiments.columns:
            success_rates = experiments.groupby('experiment_id').apply(
                lambda x: (x['status'] == 'FINISHED').sum() / len(x) * 100
            ).reset_index()
            success_rates.columns = ['Experiment ID', 'Success Rate (%)']
            
            # Create bar chart
            fig = px.bar(
                success_rates,
                x='Experiment ID',
                y='Success Rate (%)',
                title='Experiment Success Rates',
                color='Success Rate (%)',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.dataframe(success_rates, use_container_width=True)
            
            # Overall statistics
            overall_success_rate = (experiments['status'] == 'FINISHED').sum() / len(experiments) * 100
            st.metric("Overall Success Rate", f"{overall_success_rate:.1f}%")
        else:
            # Overall success rate
            if 'status' in experiments.columns:
                success_rate = (experiments['status'] == 'FINISHED').sum() / len(experiments) * 100
                st.metric("Overall Success Rate", f"{success_rate:.1f}%")
            else:
                st.info("No status data available.")
    
    except Exception as e:
        st.error(f"Error generating experiment success report: {e}")
        logger.error(f"Experiment success report error: {e}")