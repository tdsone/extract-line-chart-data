import importlib
import logging
import sys
from pathlib import Path

# Ensure src on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

import plextract.utils.logging as log_module


def test_logger_level_respects_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    importlib.reload(log_module)

    assert log_module.LOG_LEVEL == logging.DEBUG
    assert log_module.logger.name == "plextract"
