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


if __name__ == "__main__":
    pytest.main([__file__])