import os
import pytest

@pytest.fixture
def comfyui_host():
    host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    if ":" not in host:
        return f"{host}:8188"
    return host
