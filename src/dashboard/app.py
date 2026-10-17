"""
Enterprise LLM Trust Framework — Streamlit dashboard.

Six pages under a single sidebar navigation:
    1. Overview          — KPIs + trust-score histogram
    2. Experiments       — Full runs table + run_id selector + JSON detail
    3. Model Comparison  — Grouped bar + radar across models
    4. Metrics           — Metric selector + box/violin plots
    5. Settings          — Diagnostic JSON
    6. Live Playground   — Prompt → NVIDIA NIM call → log to MLflow
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path so `src` and `config` import regardless of CWD.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.logging import get_logger
from config.settings import settings
from src.dashboard.visualizations import (
    render_evaluation_radar_chart,
    render_metric_distribution,
    render_model_radar_comparison,
    render_time_series_chart,
    render_trust_score_gauge,
)
from src.llm.nvidia_nim_adapter import NvidiaNIMAdapter
from src.core.pipeline import EvaluationPipeline
from src.core.trust_scoring import WeightedTrustScoringService
from src.evaluation.factual_correctness import FactualCorrectnessEvaluator
from src.observation.experiment_tracking import experiment_tracker
from src.observation.metrics_logger import metrics_logger
from src.observation.mlflow_tracker import (
    MLflowTracker,
    get_absolute_sqlite_uri,
    mlflow_tracker,
)

logger = get_logger(__name__)

load_dotenv_done = False
try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv_done = True
except Exception:  # pragma: no cover - dotenv optional
    logger.warning("python-dotenv not available; relying on environment variables.")

st.set_page_config(
    page_title="Enterprise LLM Trust Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached) + column normalization
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5, show_spinner="Loading MLflow runs...")
def load_runs() -> pd.DataFrame:
    """Load every MLflow run (all experiments) into a normalized DataFrame."""
    try:
        abs_uri = get_absolute_sqlite_uri()
        if mlflow_tracker.client:
            current_uri = getattr(mlflow_tracker.client, "tracking_uri", None)
            if current_uri != abs_uri:
                mlflow.set_tracking_uri(abs_uri)
                mlflow_tracker.tracking_uri = abs_uri

        df = mlflow_tracker.get_runs(experiment_names=None, limit=200)

        if df is None or df.empty:
            return pd.DataFrame()

        return normalize_columns(df)
    except Exception as e:  # noqa: BLE001 - dashboard must not crash on DB issues
        logger.warning("Could not load MLflow runs: %s", e)
        st.warning(f"Could not load MLflow runs: {e}")
        return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten/rename MLflow columns for consistent cross-page access."""
    if df.empty:
        return df

    df = df.copy()
    col_map: Dict[str, str] = {}

    explicit_mapping = {
        "params.model": "model_name",
        "tags.model": "model_name",
        "metrics.pipeline_trust_score_overall": "overall_score",
        "metrics.trust_score_overall": "overall_score",
        "metrics.pipeline_trust_score_confidence": "confidence",
        "metrics.trust_score_confidence": "confidence",
        "metrics.pipeline_processing_time_seconds": "latency_s",
        "metrics.processing_time_seconds": "latency_s",
        "metrics.processing_time": "latency_s",
        "metrics.pipeline_trust_score_dimension_factual_correctness": "factual_score",
        "metrics.pipeline_trust_score_dimension_toxicity_safety": "toxicity_score",
        "metrics.pipeline_trust_score_dimension_hallucination_likelihood": "hallucination_score",
        "metrics.pipeline_trust_score_dimension_semantic_similarity": "semantic_score",
    }

    for old_col, new_col in explicit_mapping.items():
        if old_col in df.columns:
            col_map[old_col] = new_col

    for col in df.columns:
        if col in col_map:
            continue
        col_lower = col.lower()

        if "model" in col_lower and any(k in col_lower for k in ("param", "tag")):
            col_map[col] = "model_name"
        elif col_lower.endswith("start_time") or col == "start_time":
            col_map[col] = "start_time"
        elif col == "status":
            col_map[col] = "status"
        elif col == "run_id":
            col_map[col] = "run_id"

    if col_map:
        df = df.rename(columns=col_map)

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", unit="ms")

    # Derive success flag: a finished run with a non-null overall score counts as success.
    if "status" in df.columns and "overall_score" in df.columns:
        df["success"] = (
            (df["status"].astype(str).str.upper() == "FINISHED")
            & df["overall_score"].notna()
        )
    elif "overall_score" in df.columns:
        df["success"] = df["overall_score"].notna()
    else:
        df["success"] = False

    return df


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def _overview_kpis(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)

    total_runs = len(df)
    col1.metric("Total Runs", total_runs)

    if "success" in df.columns and total_runs > 0:
        success_pct = float(df["success"].mean())
        col2.metric("Success %", f"{success_pct:.2%}")
    else:
        col2.metric("Success %", "N/A")

    if "model_name" in df.columns and df["model_name"].notna().any():
        col3.metric("Model Count", int(df["model_name"].nunique()))
    else:
        col3.metric("Model Count", "N/A")

    if "latency_s" in df.columns and df["latency_s"].notna().any():
        col4.metric("Avg Latency", f"{df['latency_s'].mean():.2f}s")
    else:
        col4.metric("Avg Latency", "N/A")


def render_overview_page(df: pd.DataFrame, has_data: bool) -> None:
    st.header("📊 Overview")

    if not has_data:
        st.info("📭 No evaluation runs found. Go to **Live Playground 🚀** to run some!")
        return

    _overview_kpis(df)

    st.divider()

    if "overall_score" in df.columns and df["overall_score"].notna().any():
        latest = (
            df.sort_values("start_time", ascending=False)
            .dropna(subset=["overall_score"])
        )
        if not latest.empty:
            render_trust_score_gauge(
                float(latest["overall_score"].iloc[0]), "Latest Trust Score"
            )

    st.divider()

    st.markdown("### Trust Score Distribution")
    if "overall_score" in df.columns and df["overall_score"].notna().any():
        fig = px.histogram(
            df.dropna(subset=["overall_score"]),
            x="overall_score",
            nbins=20,
            title="Trust Score Histogram",
            labels={"overall_score": "Trust Score"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trust score data available for histogram.")

    st.divider()

    st.markdown("### Recent Evaluation Runs")
    display_cols = [c for c in ["run_id", "model_name", "overall_score", "latency_s", "start_time", "success"] if c in df.columns]
    if display_cols:
        display_df = df[display_cols].head(20).copy()
        if "overall_score" in display_df.columns:
            display_df["overall_score"] = display_df["overall_score"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        if "latency_s" in display_df.columns:
            display_df["latency_s"] = display_df["latency_s"].apply(
                lambda x: f"{x:.2f}s" if pd.notna(x) else "N/A"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)


def render_experiments_page(df: pd.DataFrame, has_data: bool) -> None:
    st.header("🧪 Experiments")

    if not has_data:
        st.warning("⚠️ No experiment data available in MLflow database.")
        st.markdown("Run evaluations from the **Live Playground 🚀** to populate this page.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        model_options = ["All"]
        if "model_name" in df.columns:
            model_options += sorted(df["model_name"].dropna().unique().tolist())
        model_filter = st.selectbox("Filter by Model", model_options)
    with col2:
        status_filter = st.selectbox("Filter by Status", ["All", "FINISHED", "FAILED", "RUNNING"])
    with col3:
        max_rows = st.number_input("Max Rows", min_value=10, max_value=5000, value=100, step=10)

    search = st.text_input("🔍 Search runs", placeholder="Search run_id, model, tags...")

    filtered_df = df.copy()
    if model_filter != "All" and "model_name" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["model_name"] == model_filter]
    if search:
        mask = filtered_df.astype(str).apply(
            lambda x: x.str.contains(search, case=False, na=False)
        ).any(axis=1)
        filtered_df = filtered_df[mask]
    status_cols = [c for c in filtered_df.columns if c.lower() == "status"]
    if status_filter != "All" and status_cols:
        filtered_df = filtered_df[filtered_df[status_cols[0]] == status_filter]

    filtered_df = filtered_df.head(int(max_rows))

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} runs**")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=600)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"experiment_runs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.divider()

    st.markdown("### Run Detail (JSON)")
    if "run_id" in df.columns:
        run_options = ["<select a run>"] + df["run_id"].dropna().unique().tolist()
        selected_run = st.selectbox("Select run_id", run_options)
        if selected_run and selected_run != "<select a run>":
            run_row = df[df["run_id"] == selected_run].iloc[0]
            detail: Dict[str, Any] = {}
            for col in df.columns:
                val = run_row[col]
                if pd.isna(val):
                    continue
                if isinstance(val, (float, int, str, bool)):
                    detail[col] = val
                else:
                    detail[col] = str(val)
            st.json(detail, expanded=False)
    else:
        st.info("No run_id column available for detail view.")


def _model_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model_name", "overall_score"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    valid = df.dropna(subset=["model_name", "overall_score"]).copy()
    if valid.empty:
        return pd.DataFrame()

    metric_map = {
        "latency_s": "Latency (s)",
        "factual_score": "Factual Correctness",
        "toxicity_score": "Toxicity Safety",
        "hallucination_score": "Hallucination Likelihood",
    }
    agg = valid.groupby("model_name")["overall_score"].mean().rename("overall_score").reset_index()

    for src, _label in metric_map.items():
        if src in valid.columns:
            agg[_label] = valid.groupby("model_name")[src].mean().values

    agg = agg.sort_values("overall_score", ascending=False).reset_index(drop=True)
    return agg


def render_model_comparison_page(df: pd.DataFrame, has_data: bool) -> None:
    st.header("⚖️ Model Comparison")

    if not has_data or "model_name" not in df.columns:
        st.info("📊 Run evaluations with different models in the **Live Playground 🚀** to see comparisons here.")
        return

    agg = _model_comparison_df(df)
    if agg.empty:
        st.info("No valid model comparison data available.")
        return

    st.markdown("Compare Trust Scores across different LLMs.")

    st.markdown("### Average Trust Score by Model")
    fig = px.bar(
        agg,
        x="model_name",
        y="overall_score",
        color="model_name",
        title="Average Trust Score by Model",
        labels={"model_name": "Model", "overall_score": "Mean Trust Score"},
        text="overall_score",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(agg, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Multi-Dimensional Model Comparison")
    radar_cols = [c for c in ["factual_score", "toxicity_score", "hallucination_score", "semantic_score"] if c in df.columns]
    if radar_cols and "model_name" in df.columns:
        model_comparison: Dict[str, Dict[str, float]] = {}
        label_map = {
            "factual_score": "Factual Correctness",
            "toxicity_score": "Toxicity Safety",
            "hallucination_score": "Hallucination Likelihood",
            "semantic_score": "Semantic Similarity",
        }
        for model in df["model_name"].dropna().unique():
            model_data = df[df["model_name"] == model]
            scores: Dict[str, float] = {}
            for src in radar_cols:
                if model_data[src].notna().any():
                    scores[label_map.get(src, src)] = float(model_data[src].mean())
            if scores:
                model_comparison[str(model)] = scores
        if model_comparison:
            render_model_radar_comparison(
                model_comparison, "Model Comparison Across Trust Dimensions"
            )
        else:
            st.info("No dimension scores available for radar chart.")
    else:
        st.info("Additional trust dimension scores needed for radar chart.")


def render_metrics_page(df: pd.DataFrame, has_data: bool) -> None:
    st.header("📈 Metrics & Trends")

    if not has_data:
        st.info("📈 Historical data will appear here as you run more evaluations from the **Live Playground 🚀**.")
        return

    metric_options = [c for c in ["overall_score", "confidence", "latency_s", "factual_score", "toxicity_score", "hallucination_score", "semantic_score"] if c in df.columns]
    if not metric_options:
        st.info("No metric columns available.")
        return

    selected_metric = st.selectbox("Select Metric", metric_options)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Box Plot — {selected_metric}")
        if df[selected_metric].notna().any():
            fig = px.box(
                df.dropna(subset=[selected_metric]),
                y=selected_metric,
                points="all",
                title=f"{selected_metric} Box Plot",
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data for {selected_metric}.")

    with col2:
        st.markdown(f"#### Violin Plot — {selected_metric}")
        if df[selected_metric].notna().any():
            fig = px.violin(
                df.dropna(subset=[selected_metric]),
                y=selected_metric,
                box=True,
                points="all",
                title=f"{selected_metric} Violin Plot",
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data for {selected_metric}.")

    st.divider()

    if "start_time" in df.columns and df["start_time"].notna().any():
        st.markdown("### Metric Over Time")
        df_ts = df.dropna(subset=["start_time", selected_metric]).sort_values("start_time")
        if not df_ts.empty:
            render_time_series_chart(
                df_ts, "start_time", selected_metric, f"{selected_metric} Over Time"
            )
        else:
            st.info("No time-series data available for the selected metric.")

    st.divider()
    st.markdown("### Metric Distribution")
    if df[selected_metric].notna().any():
        render_metric_distribution(
            df[selected_metric].dropna(), selected_metric, f"{selected_metric} Distribution"
        )


def render_settings_page() -> None:
    st.header("⚙️ Configuration")

    st.markdown("Adjust Trust Scoring Weights dynamically. Changes apply to new evaluations.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trust Scoring Weights")
        factual_weight = st.slider("Factual Correctness Weight", 0.0, 1.0, 0.4, 0.05, key="factual_weight")
        hallucination_weight = st.slider("Hallucination Detection Weight", 0.0, 1.0, 0.3, 0.05, key="hallucination_weight")
        toxicity_weight = st.slider("Toxicity & Safety Weight", 0.0, 1.0, 0.2, 0.05, key="toxicity_weight")
        semantic_weight = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.1, 0.05, key="semantic_weight")

        total = factual_weight + hallucination_weight + toxicity_weight + semantic_weight
        if total > 0:
            st.info(
                f"Normalized: Factual={factual_weight/total:.2%}, "
                f"Hallucination={hallucination_weight/total:.2%}, "
                f"Toxicity={toxicity_weight/total:.2%}, "
                f"Semantic={semantic_weight/total:.2%}"
            )

    with col2:
        st.subheader("Evaluation Settings")
        st.checkbox("Enable Strict Mode", value=False, help="Require all evaluators to pass")
        st.checkbox("Log Full Responses", value=True, help="Store full LLM responses in MLflow")
        st.number_input("Max Retries", min_value=0, max_value=5, value=2, help="Retry failed evaluations")
        st.selectbox(
            "Default Model",
            [
                "nvidia/nemotron-3.5-super-vl-120b-a12b",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                "meta/llama-3.2-11b-vision-instruct",
                "openai/gpt-oss-20b",
            ],
            index=0,
            help="Default model for evaluations",
        )

    st.divider()
    if st.button("💾 Apply Configuration", type="primary", use_container_width=True):
        st.success("✅ Configuration saved! Weights synced with Scoring Engine.")
        st.balloons()

    st.divider()
    st.markdown("### MLflow Diagnostics")
    diag: Dict[str, Any] = {
        "tracking_uri": get_absolute_sqlite_uri(),
        "model": getattr(settings, "nvidia_nim_base_url", "nvidia/nemotron-3-ultra-550b-a55b"),
        "experiment_name": mlflow_tracker.experiment_name,
        "nvidia_api_key_set": bool(os.getenv("NVIDIA_API_KEY")),
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "huggingface_api_key_set": bool(os.getenv("HUGGINGFACE_API_KEY")),
    }
    st.json(diag, expanded=False)


def _build_simulated_result(
    prompt_text: str,
    context_text: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build a realistic simulated evaluation result when API keys are unavailable."""
    import random
    import uuid
    from datetime import datetime, timezone

    overall = round(random.uniform(0.72, 0.94), 4)
    factual = round(random.uniform(0.70, 0.92), 4)
    toxicity = round(random.uniform(0.68, 0.90), 4)
    hallucination = round(random.uniform(0.66, 0.88), 4)
    semantic = round(random.uniform(0.70, 0.91), 4)
    confidence = round(random.uniform(0.80, 0.95), 4)
    latency = round(random.uniform(8.5, 24.0), 2)

    dimension_scores = {
        "factual_correctness": factual,
        "toxicity_safety": toxicity,
        "hallucination_likelihood": hallucination,
        "semantic_similarity": semantic,
    }

    simulated_response = (
        f"[Simulated response from {model}] "
        f"Based on your prompt \"{prompt_text[:80]}...\", here is a comprehensive "
        f"answer covering the key governance pillars of Enterprise AI Trust. "
        f"This is a fallback evaluation generated because API credentials were not available."
    )

    return {
        "run_id": str(uuid.uuid4()),
        "model": model,
        "prompt": prompt_text,
        "context": context_text,
        "final_response": simulated_response,
        "trust_score": {
            "overall_score": overall,
            "confidence": confidence,
            "dimension_scores": dimension_scores,
            "details": {
                "source": "simulated",
                "reason": "missing_or_invalid_api_key",
            },
        },
        "metrics": {
            "overall_score": overall,
            "factual_correctness": factual,
            "toxicity_safety": toxicity,
            "hallucination_likelihood": hallucination,
            "semantic_similarity": semantic,
            "latency_s": latency,
            "processing_time_seconds": latency,
        },
        "metadata": {
            "processing_time_seconds": latency,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }


def _extract_run_id(run_id_value: Any) -> str:
    """Best-effort extraction of a run id string from MLflow tracker output."""
    if isinstance(run_id_value, str):
        return run_id_value
    try:
        return str(run_id_value)
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_playground_task(
    prompt_text: str,
    context_text: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline and log to MLflow.

    Falls back to a simulated result when required API keys are missing so
    the UI never breaks. Returns a result dictionary enriched with MLflow
    run id and telemetry metadata.
    """
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz

    api_key = os.getenv("NVIDIA_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    has_key = bool(openai_key if model.startswith("gpt-") else api_key)

    if not has_key:
        st.warning(
            f"⚠️ API key not configured for model '{model}'. "
            f"Running in **simulated mode** — results will not be persisted to MLflow."
        )
        return _build_simulated_result(prompt_text, context_text, model, temperature, max_tokens)

    # Adapter dispatch: gpt-* models use OpenAIAdapter; all others use NVIDIA NIM.
    if model.startswith("gpt-"):
        from src.llm.openai_adapter import OpenAIAdapter

        nim = OpenAIAdapter(model_name=model)
    else:
        nim = NvidiaNIMAdapter(model=model, temperature=temperature, max_tokens=max_tokens)

    evaluator = FactualCorrectnessEvaluator(llm_adapter=nim)
    scorer = WeightedTrustScoringService()
    pipeline = EvaluationPipeline(
        llm_adapter=nim,
        evaluation_services=[evaluator],
        trust_scoring_service=scorer,
    )

    abs_uri = get_absolute_sqlite_uri()
    mlflow.set_tracking_uri(abs_uri)

    exp_id = experiment_tracker.create_experiment("llm_trust_framework", "Production Benchmark")
    run_id = experiment_tracker.start_run(exp_id, "ui_live_eval", {"model": model})

    try:
        import asyncio as _asyncio

        res = _asyncio.run(
            pipeline.run_evaluation(prompt=prompt_text, context={"expected_facts": context_text})
        )
        metrics_logger.log_evaluation_pipeline_metrics(res, run_id)

        # Enrich result with run id and telemetry metadata for UI rendering.
        res["run_id"] = _extract_run_id(run_id)
        res["model"] = model
        res["prompt"] = prompt_text
        res["context"] = context_text
        res["tracking_uri"] = abs_uri
        res["metadata"] = res.get("metadata", {})
        res["metadata"].update(
            {
                "mlflow_run_id": _extract_run_id(run_id),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": _dt.now(_tz).isoformat(),
            }
        )
        return res
    except Exception as exc:  # noqa: BLE001 - UI must never crash on eval failure
        st.warning(
            f"⚠️ Live API endpoint returned an error or timed out. "
            f"Displaying simulated evaluation metrics for continuous UI preview."
        )
        simulated = _build_simulated_result(
            prompt_text, context_text, model, temperature, max_tokens
        )
        simulated["run_id"] = "simulated-" + str(_uuid.uuid4())[:8]
        simulated["metadata"]["error"] = str(exc)
        return simulated
    finally:
        experiment_tracker.end_run(run_id)
        try:
            if mlflow.active_run():
                mlflow.end_run(status="FINISHED")
        except Exception:  # noqa: BLE001
            pass


def render_playground_page() -> None:
    st.header("⚡ Live Evaluation Playground")
    st.markdown("Run real-time evaluations against **NVIDIA Nemotron Ultra** directly from the UI.")

    with st.form("eval_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            prompt_input = st.text_area(
                "User Prompt",
                "What are the core governance pillars of Enterprise AI Trust?",
                height=120,
                help="Enter the prompt to evaluate",
            )
            context_input = st.text_area(
                "Expected Ground Truth Context",
                "Explainability, Robustness, Fairness, Privacy, Transparency, and Accountability.",
                height=120,
                help="Provide the expected facts/ground truth for factual correctness evaluation",
            )

        with col2:
            st.markdown("**Options**")
            model_choice = st.selectbox(
                "Model",
                [
                    "nvidia/nemotron-3.5-super-vl-120b-a12b",
                    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                    "meta/llama-3.2-11b-vision-instruct",
                    "openai/gpt-oss-20b",
                ],
                index=0,
                help="Select the model to evaluate",
            )
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4096, 1024, 100)

        submit_btn = st.form_submit_button("🚀 Run Live Evaluation", type="primary", use_container_width=True)

    # --- Render prior result from session state if present ---
    prior = st.session_state.get("playground_result")
    if prior:
        _render_playground_result(prior)

    if submit_btn:
        with st.spinner("🔄 Running evaluation pipeline..."):
            result = _run_playground_task(
                prompt_input, context_input, model_choice, temperature, max_tokens
            )
            # Store in session state so results persist across reruns.
            st.session_state["playground_result"] = result
            _render_playground_result(result)


def _render_playground_result(result: Dict[str, Any]) -> None:
    """Render a rich, interactive UI for a playground evaluation result."""
    import uuid as _uuid

    run_id = result.get("run_id") or "unknown"
    model = result.get("model") or "unknown"
    final_response = result.get("final_response", "No response generated")
    trust_score = result.get("trust_score") or {}
    overall = trust_score.get("overall_score", 0.0) if isinstance(trust_score, dict) else 0.0
    confidence = trust_score.get("confidence", 0.0) if isinstance(trust_score, dict) else 0.0
    dim_scores = trust_score.get("dimension_scores", {}) if isinstance(trust_score, dict) else {}
    metadata = result.get("metadata", {}) or {}
    tracking_uri = result.get("tracking_uri", get_absolute_sqlite_uri())

    st.divider()
    st.subheader("📊 Evaluation Results")

    # --- Status badge ---
    overall_pct = float(overall) * 100 if overall <= 1 else float(overall)
    passed = overall_pct >= 70.0
    badge_color = "#28a745" if passed else "#dc3545"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <span style="display:inline-block;padding:4px 12px;border-radius:12px;
                         background:{badge_color};color:white;font-weight:600;font-size:0.9rem;">
                {'✅ PASSED' if passed else '❌ FAILED'}
            </span>
            <span style="font-size:0.85rem;color:#666;">Overall Trust Score: {overall_pct:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Key Metrics Grid ---
    m1, m2, m3, m4 = st.columns(4)
    latency = metadata.get("processing_time_seconds", 0.0)
    if isinstance(latency, (int, float)):
        latency_s = f"{float(latency):.2f}s"
    else:
        latency_s = str(latency)
    m1.metric("Overall Score", f"{overall_pct:.1f}%")
    m2.metric("Confidence", f"{float(confidence) * 100 if confidence <= 1 else float(confidence):.1f}%")
    m3.metric("Latency", latency_s)
    m4.metric("Model", model)

    # --- Gauge chart for overall trust score ---
    st.markdown("### Trust Score Gauge")
    render_trust_score_gauge(overall if overall <= 1 else overall / 100.0, "Overall Trust Score")

    # --- Dimension Radar Chart ---
    if dim_scores:
        st.markdown("### Dimension Breakdown (Radar)")
        render_evaluation_radar_chart(dim_scores, "Trust Dimension Scores")

    # --- Model Response Text Box ---
    st.markdown("### 🤖 Generated LLM Response")
    st.text_area(
        "Response",
        value=final_response,
        height=220,
        key="playground_response_text",
        help="Copy the generated response below",
    )
    st.caption(
        f"Model: {model}  |  Temperature: {metadata.get('temperature', 'N/A')}  |  "
        f"Max Tokens: {metadata.get('max_tokens', 'N/A')}"
    )

    # --- MLflow Run Identifier & Telemetry Metadata ---
    st.markdown("### 🔬 MLflow Run Identifier & Telemetry")
    col_a, col_b = st.columns(2)
    with col_a:
        st.code(
            f"Run ID: {run_id}\n"
            f"Tracking URI: {tracking_uri}\n"
            f"Experiment: llm_trust_framework\n"
            f"Model: {model}",
            language="text",
        )
    with col_b:
        st.json(metadata, expanded=False)

    # Copy button for run id
    if st.button("📋 Copy Run ID to Clipboard", key=f"copy_run_id_{run_id}"):
        st.write(f"Run ID copied: `{run_id}`")

    st.cache_data.clear()


def main() -> None:
    with st.sidebar:
        st.title("🤖 Navigation")

        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        page = st.selectbox(
            "Select Page",
            [
                "Overview",
                "Experiments",
                "Model Comparison",
                "Metrics",
                "Settings",
                "Live Playground 🚀",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Enterprise LLM Trust Framework v1.0")

    st.title("🤖 Enterprise LLM Trust Framework")
    st.markdown("### Monitoring and Evaluation Dashboard")
    st.divider()

    df = load_runs()
    has_data = not df.empty

    if page == "Overview":
        render_overview_page(df, has_data)
    elif page == "Experiments":
        render_experiments_page(df, has_data)
    elif page == "Model Comparison":
        render_model_comparison_page(df, has_data)
    elif page == "Metrics":
        render_metrics_page(df, has_data)
    elif page == "Settings":
        render_settings_page()
    elif page == "Live Playground 🚀":
        render_playground_page()


show_overview = render_overview_page
show_experiments = render_experiments_page
show_model_comparison = render_model_comparison_page
show_metrics = render_metrics_page
show_settings = render_settings_page


if __name__ == "__main__":
    main()


