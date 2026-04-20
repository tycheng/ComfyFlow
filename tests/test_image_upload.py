import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from comfyflow import Workflow, ComfyClient
from comfyflow.registry import SchemaRegistry

MOCK_SCHEMA = {
    "LoadImage": {
        "input": {
            "required": {
                "image": [["example.png", "test.jpg"], {"image_upload": True}]
            }
        },
        "output": ["IMAGE", "MASK"],
        "output_name": ["image", "mask"],
        "name": "LoadImage",
        "category": "image"
    }
}

@pytest.fixture
def registry():
    reg = SchemaRegistry()
    reg._parse(MOCK_SCHEMA)
    return reg

@pytest.fixture
def mock_client(registry):
    client = MagicMock()
    client.registry = registry
    client.server_address = "127.0.0.1:8188"
    client.upload_image = AsyncMock(return_value={"name": "uploaded_image.png"})
    # client.run is already async generator if needed, but we mostly test workflow setup here
    return client

@pytest.mark.asyncio
async def test_load_image_validation(registry, mock_client, tmp_path):
    wf = Workflow(mock_client)
    
    # 1. test with valid enum value
    node = wf.LoadImage(image="example.png")
    assert node._node.inputs["image"] == "example.png"
    
    # 2. test with invalid enum value (should fail)
    with pytest.raises(ValueError, match="Invalid value 'nonexistent.png'"):
        wf.LoadImage(image="nonexistent.png")
        
    # 3. test with local file path (should pass because of image_upload: true)
    test_file = tmp_path / "my_local_image.png"
    test_file.write_text("fake image data")
    
    node = wf.LoadImage(image=str(test_file))
    assert node._node.inputs["image"] == str(test_file)

@pytest.mark.asyncio
async def test_transparent_upload(registry, mock_client, tmp_path):
    wf = Workflow(mock_client)
    test_file = tmp_path / "upload_me.png"
    test_file.write_text("data")
    
    wf.LoadImage(image=str(test_file))
    
    # mocking what comfyclient.run does
    real_client = ComfyClient()
    real_client.upload_image = AsyncMock(return_value={"name": "uploaded_file.png"})
    
    await real_client.ensure_images_uploaded(wf)
    
    assert wf.nodes[0].inputs["image"] == "uploaded_file.png"
    real_client.upload_image.assert_called_once()

@pytest.mark.asyncio
async def test_explicit_upload(mock_client):
    from comfyflow.client import ComfyClient
    client = ComfyClient()
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {"name": "test_uploaded.png"}
        
        res = await client.upload_image(b"fake_bytes", filename="test.png")
        assert res["name"] == "test_uploaded.png"
        mock_post.assert_called_once()
