import pytest
from src.config import load_config


@pytest.fixture
def cfg_lstm():
    return load_config("configs/lstm.yaml")


@pytest.fixture
def cfg_rnn():
    return load_config("configs/default.yaml")
