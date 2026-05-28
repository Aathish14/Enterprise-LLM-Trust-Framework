"""
Streamlit dashboard for the Enterprise LLM Trust Framework.
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
from observation.experiment_tracking import experiment_tracker
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enterprise LLM Trust Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main dashboard application."""
    st.title("🤖 Enterprise LLM Trust Framework")
    st.markdown("### Monitoring and Evaluation Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Experiments", "Model Comparison", "Metrics", "Settings"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Experiments":
        show_experiments()
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Metrics":
        show_metrics()
    elif page == "Settings":
        show_settings()

def show_overview():
    """Show overview page."""
    st.header("📊 Overview")
    
    # Get experiment data
    try:
        experiments = mlflow_tracker.search_runs(max_results=100)
        if experiments.empty:
            st.info("No experiments found. Run some evaluations to see data here.")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(experiments)
        
        # Show key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Experiments", len(df))
        
        with col2:
            successful_runs = len(df[df['status'] == 'FINISHED']) if 'status' in df.columns else 0
            st.metric("Successful Runs", successful_runs)
        
        with col3:
            avg_duration = df['duration'].mean() / 1000 if 'duration' in df.columns and not df['duration'].isna().all() else 0
            st.metric("Avg Duration (s)", f"{avg_duration:.2f}")
        
        with col4:
            st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
        
        # Show recent experiments
        st.subheader("Recent Experiments")
        if not df.empty:
            # Select relevant columns for display
            display_cols = ['run_id', 'experiment_id', 'status', 'start_time', 'end_time', 'duration']
            available_cols = [col for col in display_cols if col in df.columns]
            if available_cols:
                display_df = df[available_cols].copy()
                if 'start_time' in display_df.columns:
                    display_df['start_time'] = pd.to_datetime(display_df['start_time'], unit='ms')
                if 'end_time' in display_df.columns:
                    display_df['end_time'] = pd.to_datetime(display_df['end_time'], unit='ms')
                if 'duration' in display_df.columns:
                    display_df['duration'] = (display_df['duration'] / 1000).round(2)  # Convert to seconds
                
                st.dataframe(display_df.head(10), use_container_width=True)
        
        # Show experiment trends over time
        st.subheader("Experiment Trends")
        if not df.empty and 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
            df_daily = df.set_index('start_time').resample('D').size().reset_index()
            df_daily.columns = ['date', 'count']
            
            fig = px.line(df_daily, x='date', y='count', title='Experiments per Day')
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading experiment data: {e}")
        logger.error(f"Dashboard error: {e}")

def show_experiments():
    """Show experiments page."""
    st.header("🧪 Experiments")
    
    try:
        # Get experiments
        experiments = mlflow_tracker.search_runs(max_results=1000)
        
        if experiments.empty:
            st.info("No experiments found.")
            return
        
        # Experiment selector
        experiment_names = experiments['experiment_id'].unique() if 'experiment_id' in experiments.columns else []
        if len(experiment_names) > 0:
            selected_experiment = st.selectbox(
                "Select Experiment",
                options=experiment_names,
                format_func=lambda x: f"Experiment {x}"
            )
            
            # Filter runs for selected experiment
            if 'experiment_id' in experiments.columns:
                experiment_runs = experiments[experiments['experiment_id'] == selected_experiment]
            else:
                experiment_runs = experiments
            
            if not experiment_runs.empty:
                st.subheader(f"Runs for Experiment {selected_experiment}")
                
                # Display runs
                display_cols = ['run_id', 'status', 'start_time', 'end_time', 'duration']
                available_cols = [col for col in display_cols if col in experiment_runs.columns]
                if available_cols:
                    display_df = experiment_runs[available_cols].copy()
                    if 'start_time' in display_df.columns:
                        display_df['start_time'] = pd.to_datetime(display_df['start_time'], unit='ms')
                    if 'end_time' in display_df.columns:
                        display_df['end_time'] = pd.to_datetime(display_df['end_time'], unit='ms')
                    if 'duration' in display_df.columns:
                        display_df['duration'] = (display_df['duration'] / 1000).round(2)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Show metrics for selected run
                    if len(display_df) > 0:
                        selected_run_idx = st.selectbox(
                            "Select Run to View Details",
                            options=range(len(display_df)),
                            format_func=lambda x: f"Run {display_df.iloc[x]['run_id'][:8]}..."
                        )
                        
                        if selected_run_idx is not None:
                            selected_run_id = experiment_runs.iloc[selected_run_idx]['run_id']
                            show_run_details(selected_run_id)
            else:
                st.info("No runs found for selected experiment.")
        else:
            st.info("No experiments available.")
    
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        logger.error(f"Experiments page error: {e}")

def show_run_details(run_id: str):
    """Show details for a specific run."""
    try:
        # Get run data
        run_data = mlflow_tracker.client.get_run(run_id)
        
        st.subheader(f"Run Details: {run_id[:8]}...")
        
        # Display run info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Status:**", run_data.info.status)
            st.write("**Start Time:**", datetime.fromtimestamp(run_data.info.start_time/1000))
            if run_data.info.end_time:
                st.write("**End Time:**", datetime.fromtimestamp(run_data.info.end_time/1000))
                duration = (run_data.info.end_time - run_data.info.start_time) / 1000
                st.write("**Duration:**", f"{duration:.2f} seconds")
        
        with col2:
            st.write("**Parameters:**")
            for key, value in run_data.data.params.items():
                st.write(f"- {key}: {value}")
        
        # Display metrics
        if run_data.data.metrics:
            st.write("**Metrics:**")
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": v} 
                for k, v in run_data.data.metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True)
        
        # Display tags
        if run_data.data.tags:
            st.write("**Tags:**")
            tags_df = pd.DataFrame([
                {"Tag": k, "Value": v} 
                for k, v in run_data.data.tags.items()
            ])
            st.dataframe(tags_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading run details: {e}")
        logger.error(f"Run details error: {e}")

def show_model_comparison():
    """Show model comparison page."""
    st.header("⚖️ Model Comparison")
    
    st.info("Model comparison functionality will be implemented here.")
    st.write("This page will show comparative benchmark results between different LLMs.")

def show_metrics():
    """Show metrics page."""
    st.header("📈 Metrics")
    
    st.info("Detailed metrics visualization will be implemented here.")
    st.write("This page will show detailed charts and graphs of evaluation metrics.")

def show_settings():
    """Show settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("Configuration")
    
    # Display current settings
    st.write("**MLflow Tracking URI:**", settings.MLFLOW_TRACKING_URI)
    st.write("**Weights & Biases Project:**", settings.WANDB_PROJECT)
    st.write("**Debug Mode:**", settings.DEBUG)
    
    st.subheader("Trust Score Weights")
    weights_df = pd.DataFrame([
        {"Dimension": k, "Weight": v} 
        for k, v in settings.TRUST_SCORE_WEIGHTS.items()
    ])
    st.dataframe(weights_df, use_container_width=True)
    
    # Allow weight adjustment
    st.write("Adjust Trust Score Weights (must sum to 1.0):")
    new_weights = {}
    total = 0.0
    
    for dimension, weight in settings.TRUST_SCORE_WEIGHTS.items():
        new_weight = st.slider(
            f"{dimension}", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(weight), 
            step=0.01,
            key=f"weight_{dimension}"
        )
        new_weights[dimension] = new_weight
        total += new_weight
    
    st.write(f"**Total Weight:** {total:.3f}")
    
    if abs(total - 1.0) > 0.001:
        st.warning("Weights should sum to 1.0 for proper normalization")
    else:
        if st.button("Apply Weights"):
            # In a real app, we would save these to a config file or database
            st.success("Weights applied! (Not persisted in this demo)")

if __name__ == "__main__":
    main()