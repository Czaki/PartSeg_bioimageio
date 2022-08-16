from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Return the path to the data directory."""
    return Path(__file__).parent.parent.parent.parent / "test_data"
