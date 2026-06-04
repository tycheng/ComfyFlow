import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from comfyflow import AsyncComfyClient, Workflow
from comfyflow.media import HAS_MOVIEPY, VideoClip
from comfyflow.registry import SchemaRegistry

MOCK_SCHEMA = {
    "LoadVideo": {
        "input": {
            "required": {
                "video": [["example.mp4"], {"video_upload": True}]
            }
        },
        "output": ["IMAGE", "AUDIO"],
        "output_name": ["images", "audio"],
        "name": "LoadVideo",
        "category": "video"
    }
}

@pytest.fixture
def registry():
    return SchemaRegistry(MOCK_SCHEMA)

@pytest.fixture
async def mock_client(registry):
    client = await AsyncComfyClient.create("127.0.0.1:8188")
    client.registry = registry
    return client

@pytest.mark.asyncio
async def test_upload_video_file(mock_client, tmp_path):
    # Create a dummy video file
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy mp4 content")

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {
            "name": "test_video.mp4",
            "subfolder": "comfyflow",
            "type": "input"
        }

        res = await mock_client.upload_video(video_path)
        
        assert res["name"] == "test_video.mp4"
        mock_post.assert_called_once()
        
        # Verify multipart data
        files = mock_post.call_args.kwargs["files"]
        assert "image" in files # ComfyUI uses 'image' key even for videos
        assert files["image"][0] == "test_video.mp4"
        assert files["image"][2] == "video/mp4"

@pytest.mark.asyncio
async def test_upload_videoclip(mock_client):
    if not HAS_MOVIEPY:
        pytest.skip("MoviePy not installed")

    import numpy as np
    try:
        from moviepy.editor import ColorClip
    except ImportError:
        from moviepy import ColorClip

    # Create a tiny 1-second dummy clip
    clip = ColorClip(size=(64, 64), color=(255, 0, 0), duration=1)
    # MoviePy 2.0+ needs fps to write
    clip.fps = 24

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {
            "name": "clip_upload.mp4",
            "subfolder": "comfyflow",
            "type": "input"
        }

        # We also need to mock the cleanup in the real code if we want to be strict,
        # but here we just test that it calls upload_file correctly.
        res = await mock_client.upload_video(clip)

        assert res["name"] == "clip_upload.mp4"
        mock_post.assert_called_once()
        
        files = mock_post.call_args.kwargs["files"]
        assert files["image"][0].startswith("upload_")
        assert files["image"][0].endswith(".mp4")

@pytest.mark.asyncio
async def test_workflow_video_upload(mock_client, tmp_path):
    wf = Workflow(mock_client)
    video_path = tmp_path / "wf_video.mp4"
    video_path.write_bytes(b"wf dummy content")

    # Manually create node since we are using mock registry
    wf.LoadVideo(video=str(video_path))

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {
            "name": "uploaded_wf_video.mp4",
            "subfolder": "comfyflow"
        }

        await mock_client.ensure_media_uploaded(wf)

        # Check if the node input was updated
        assert wf.nodes[0].inputs["video"] == os.path.join("comfyflow", "uploaded_wf_video.mp4")
