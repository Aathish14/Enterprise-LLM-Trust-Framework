"""
Unit tests for the dashboard visualizations module.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_import_visualizations():
    """Test that the visualizations module can be imported."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        import src.dashboard.visualizations
        assert src.dashboard.visualizations is not None


def test_render_trust_score_gauge_exists():
    """Test that the render_trust_score_gauge function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_trust_score_gauge
        assert callable(render_trust_score_gauge)


def test_render_evaluation_radar_chart_exists():
    """Test that the render_evaluation_radar_chart function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_evaluation_radar_chart
        assert callable(render_evaluation_radar_chart)


def test_render_time_series_chart_exists():
    """Test that the render_time_series_chart function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_time_series_chart
        assert callable(render_time_series_chart)


def test_render_comparison_bar_chart_exists():
    """Test that the render_comparison_bar_chart function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_comparison_bar_chart
        assert callable(render_comparison_bar_chart)


def test_render_metric_distribution_exists():
    """Test that the render_metric_distribution function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_metric_distribution
        assert callable(render_metric_distribution)


def test_render_correlation_heatmap_exists():
    """Test that the render_correlation_heatmap function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_correlation_heatmap
        assert callable(render_correlation_heatmap)


def test_render_pipeline_timeline_exists():
    """Test that the render_pipeline_timeline function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_pipeline_timeline
        assert callable(render_pipeline_timeline)


def test_render_model_radar_comparison_exists():
    """Test that the render_model_radar_comparison function exists."""
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from src.dashboard.visualizations import render_model_radar_comparison
        assert callable(render_model_radar_comparison)


if __name__ == "__main__":
    pytest.main([__file__])