"""Shared pytest fixtures.

Tests run with CWD = backend/ so config paths (data/, models/, etc.) resolve
relative to that directory, matching the uvicorn runtime.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Ensure `import app`, `import api`, `import services`, `import src` all work.
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session", autouse=True)
def _chdir_backend():
    old = os.getcwd()
    os.chdir(BACKEND_DIR)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app import app

    with TestClient(app) as c:
        yield c
