"""
Unit tests for the Streamlit dashboard app.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_set_page_config_called():
    """Test that Streamlit page config is called."""
    with patch('streamlit.set_page_config') as mock_set_page_config:
        # Import the module to trigger the page config call
        import src.dashboard.app
        # Reload to ensure we get fresh state
        if 'src.dashboard.app' in sys.modules:
            del sys.modules['src.dashboard.app']
        
        # Re-import to test
        with patch('streamlit.set_page_config') as mock_set_page_config:
            import src.dashboard.app
            mock_set_page_config.assert_called()


def test_main_function_exists():
    """Test that main function exists in app module."""
    with patch('streamlit.set_page_config'):
        import src.dashboard.app
        assert hasattr(src.dashboard.app, 'main')
        assert callable(src.dashboard.app.main)


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components."""
    with patch('streamlit.title') as mock_title, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.sidebar.selectbox') as mock_selectbox:
        
        mock_sidebar.title = Mock()
        mock_selectbox.return_value = "Overview"
        
        yield {
            'title': mock_title,
            'markdown': mock_markdown,
            'sidebar': mock_sidebar,
            'selectbox': mock_selectbox
        }


def test_show_overview_function_exists():
    """Test that show_overview function exists."""
    with patch('streamlit.set_page_config'):
        import src.dashboard.app
        assert hasattr(src.dashboard.app, 'show_overview')
        assert callable(src.dashboard.app.show_overview)


def test_show_experiments_function_exists():
    """Test that show_experiments function exists."""
    with patch('streamlit.set_page_config'):
        import src.dashboard.app
        assert hasattr(src.dashboard.app, 'show_experiments')
        assert callable(src.dashboard.app.show_experiments)


def test_show_model_comparison_function_exists():
    """Test that show_model_comparison function exists."""
    with patch('streamlit.set_page_config'):
        import src.dashboard.app
        assert hasattr(src.dashboard.app, 'show_model_comparison')
        assert callable(src.dashboard.app.show_model_comparison)


def test_show_metrics_function_exists():
    """Test that show_metrics function exists."""
    with patch('streamlit.set_page_config'):
        import src.dashboard.app
        assert hasattr(src.dashboard.app, 'show_metrics')
        assert callable(src.dashboard.app.show_metrics)


def test_show_settings_function_exists():
    """Test that show_settings function exists."""
    with patch('streamlit.set_page_config'):
        import src.dashboard.app
        assert hasattr(src.dashboard.app, 'show_settings')
        assert callable(src.dashboard.app.show_settings)


def test_playground_result_widget_keys_unique_per_result():
    """Regression test for DuplicateWidgetID crash.

    The generated-response text_area previously used a hardcoded key
    "playground_response_text". Rendering two distinct results in one
    Streamlit script run (once from session state, once from a new
    submission) registered the same widget key twice and raised
    DuplicateWidgetID. The fix makes the key run_id-specific:
    f"playground_response_text_{run_id}".

    This test calls _render_playground_result() twice with different
    run_id values and asserts that st.text_area is invoked with two
    distinct, deterministic keys.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

    from src.dashboard.app import _render_playground_result

    result_a = {
        "run_id": "run-a-123",
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "prompt": "What is AI trust?",
        "context": "Expected context",
        "final_response": "Response A text",
        "trust_score": {
            "overall_score": 0.85,
            "confidence": 0.90,
            "dimension_scores": {
                "factual_correctness": 0.8,
                "toxicity_safety": 0.9,
                "hallucination_likelihood": 0.85,
                "semantic_similarity": 0.75,
            },
        },
        "metadata": {
            "is_simulated": False,
            "source": "live",
            "processing_time_seconds": 10.5,
        },
    }

    result_b = {
        "run_id": "run-b-456",
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "prompt": "What is AI trust?",
        "context": "Expected context",
        "final_response": "Response B text",
        "trust_score": {
            "overall_score": 0.75,
            "confidence": 0.85,
            "dimension_scores": {
                "factual_correctness": 0.7,
                "toxicity_safety": 0.85,
                "hallucination_likelihood": 0.8,
                "semantic_similarity": 0.7,
            },
        },
        "metadata": {
            "is_simulated": True,
            "source": "simulated",
            "processing_time_seconds": 12.0,
        },
    }

    registered_keys = []

    def capture_text_area(*args, **kwargs):
        key = kwargs.get("key")
        if key is not None:
            registered_keys.append(key)
        return Mock()

    mock_col = Mock()
    mock_col.metric = Mock()

    def mock_columns(n):
        cols = []
        for _ in range(n):
            col = Mock()
            col.metric = Mock()
            col.__enter__ = Mock(return_value=col)
            col.__exit__ = Mock(return_value=False)
            cols.append(col)
        return cols

    with patch("streamlit.text_area", side_effect=capture_text_area), \
         patch("streamlit.divider"), \
         patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=mock_columns), \
         patch("streamlit.caption"), \
         patch("streamlit.code"), \
         patch("streamlit.json"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.write"), \
         patch("streamlit.cache_data.clear"), \
         patch("src.dashboard.app.render_trust_score_gauge"), \
         patch("src.dashboard.app.render_evaluation_radar_chart"):

        _render_playground_result(result_a)
        _render_playground_result(result_b)

    text_area_keys = [k for k in registered_keys if k is not None]
    assert len(text_area_keys) == 2, \
        f"Expected 2 text_area calls, got {len(text_area_keys)}: {text_area_keys}"
    assert text_area_keys[0] == "playground_response_text_run-a-123", \
        f"Unexpected key: {text_area_keys[0]}"
    assert text_area_keys[1] == "playground_response_text_run-b-456", \
        f"Unexpected key: {text_area_keys[1]}"
    assert len(set(text_area_keys)) == len(text_area_keys), \
        f"Duplicate keys detected: {text_area_keys}"


if __name__ == "__main__":
    pytest.main([__file__])