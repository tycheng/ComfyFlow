import os
import httpx
import pytest

def get_comfyui_host():
    host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    if ":" not in host:
        return f"{host}:8188"
    return host

@pytest.fixture
def comfyui_host():
    return get_comfyui_host()

def pytest_sessionstart(session):
    """
    Check if the ComfyUI server is reachable before running any tests.
    """
    host = get_comfyui_host()
    url = f"http://{host}/system_stats"

    print(f"\nChecking ComfyUI server connection at {url}...")
    try:
        # We use a short timeout as this is just a pre-flight check
        response = httpx.get(url, timeout=2.0)
        response.raise_for_status()
        print("ComfyUI server is reachable.")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        pytest.exit(
            f"\n\nERROR: Could not connect to ComfyUI server at {host}.\n"
            f"Make sure ComfyUI is running and COMFYUI_HOST is set correctly.\n"
            f"Error details: {e}\n",
            returncode=1
        )
