import pytest
from fastapi.testclient import TestClient
from fastapi_app import app


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI test client."""
    return TestClient(app)
