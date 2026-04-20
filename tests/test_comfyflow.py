import pytest
from unittest.mock import patch, MagicMock
from comfyflow.models import OutputRef
from comfyflow import Workflow

MOCK_SCHEMA = {
    "CheckpointLoaderSimple": {
        "input": {
            "required": {
                "ckpt_name": [["anything.safetensors", "v1-5-pruned.safetensors"], {}]
            }
        },
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["model", "clip", "vae"],
        "name": "CheckpointLoaderSimple",
        "category": "loaders"
    },
    "CLIPTextEncode": {
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True}],
                "clip": ["CLIP", {}]
            }
        },
        "output": ["CONDITIONING"],
        "name": "CLIPTextEncode",
        "category": "conditioning"
    }
}

@pytest.fixture
async def workflow():
    mock_cli = MagicMock()
    # Mock registry with direct node injection
    mock_registry = MagicMock()
    mock_registry.nodes = {}
    
    # Manually populate the mock registry
    from comfyflow.registry import SchemaRegistry
    # We can just create a real registry instance and inject the data
    registry = SchemaRegistry()
    registry._parse(MOCK_SCHEMA)
    
    mock_cli.registry = registry
    
    wf = Workflow(mock_cli)
    return wf


@pytest.mark.asyncio
async def test_workflow_initialization(workflow):
    assert "CheckpointLoaderSimple" in workflow.client.registry.nodes
    assert "CLIPTextEncode" in workflow.client.registry.nodes

@pytest.mark.asyncio
async def test_node_creation(workflow):
    ckpt = workflow.CheckpointLoaderSimple(ckpt_name="anything.safetensors")
    assert len(workflow.nodes) == 1
    assert ckpt._node.schema.name == "CheckpointLoaderSimple"
    assert isinstance(ckpt.clip, OutputRef)

@pytest.mark.asyncio
async def test_validation_invalid_input(workflow):
    # This now fails earlier due to missing required input
    with pytest.raises(ValueError, match="requires input 'ckpt_name'"):
        workflow.CheckpointLoaderSimple(invalid_param="error")

    # This passes the required input check but fails on input key check
    with pytest.raises(ValueError, match="does not have input 'invalid_param'"):
        workflow.CheckpointLoaderSimple(ckpt_name="anything.safetensors", invalid_param="error")

@pytest.mark.asyncio
async def test_validation_invalid_enum(workflow):
    with pytest.raises(ValueError, match="Invalid value 'wrong.safetensors'"):
        workflow.CheckpointLoaderSimple(ckpt_name="wrong.safetensors")

@pytest.mark.asyncio
async def test_api_export(workflow):
    ckpt = workflow.CheckpointLoaderSimple(ckpt_name="anything.safetensors")
    workflow.CLIPTextEncode(text="a cat", clip=ckpt.clip)

    api_json = workflow.to_api_json()
    assert "1" in api_json
    assert "2" in api_json
    assert api_json["1"]["class_type"] == "CheckpointLoaderSimple"
    assert api_json["2"]["inputs"]["clip"] == ["1", 1]

@pytest.mark.asyncio
async def test_node_outputs_auto_resolution(workflow):
    # Test that passing NodeOutputs directly defaults to the first output
    ckpt = workflow.CheckpointLoaderSimple(ckpt_name="anything.safetensors")
    # CLIPTextEncode expects CLIP (slot 1), but we pass the whole node (slot 0 is MODEL)
    # This specifically tests our _resolve_input logic in export.py
    workflow.CLIPTextEncode(text="test", clip=ckpt)

    api_json = workflow.to_api_json()
    # Should default to slot 0 (MODEL) which is incorrect for CLIP but confirms resolution logic works
    assert api_json["2"]["inputs"]["clip"] == ["1", 0]

@pytest.mark.asyncio
async def test_ui_export(workflow):
    ckpt = workflow.CheckpointLoaderSimple(ckpt_name="anything.safetensors")
    workflow.CLIPTextEncode(text="a cat", clip=ckpt.clip)

    ui_json = workflow.to_ui_json()
    assert len(ui_json["nodes"]) == 2
    assert len(ui_json["links"]) == 1
    assert ui_json["nodes"][0]["type"] == "CheckpointLoaderSimple"
