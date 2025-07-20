"""
Pytest configuration file
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for tests."""
    return pd.DataFrame({
        'id': range(1000),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
