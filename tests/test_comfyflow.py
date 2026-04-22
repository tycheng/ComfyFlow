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
                "text": ["STRING", {"multiline": True, "default": "default prompt"}],
                "clip": ["CLIP", {}]
            }
        },
        "output": ["CONDITIONING"],
        "name": "CLIPTextEncode",
        "category": "conditioning"
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512, "min": 64, "max": 4096}],
                "height": ["INT", {"default": 512, "min": 64, "max": 4096}],
                "batch_size": ["INT", {"default": 1, "min": 1, "max": 64}]
            }
        },
        "output": ["LATENT"],
        "name": "EmptyLatentImage",
        "category": "latent"
    },
    "SamplerCustom": {
        "input": {
            "required": {
                "sampler_name": [["euler", "euler_ancestral", "heun"], {"default": "euler"}],
                "scheduler": [["normal", "karras", "simple"], {"default": "normal"}]
            }
        },
        "output": ["SAMPLER"],
        "name": "SamplerCustom",
        "category": "sampling"
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
    registry = SchemaRegistry(MOCK_SCHEMA)

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
async def test_default_values(workflow):
    # EmptyLatentImage has defaults for all fields
    latent = workflow.EmptyLatentImage()
    assert latent._node.inputs["width"] == 512
    assert latent._node.inputs["height"] == 512
    assert latent._node.inputs["batch_size"] == 1

    # CLIPTextEncode has default for text but clip is required (no default)
    ckpt = workflow.CheckpointLoaderSimple(ckpt_name="anything.safetensors")
    encode = workflow.CLIPTextEncode(clip=ckpt.clip)
    assert encode._node.inputs["text"] == "default prompt"

@pytest.mark.asyncio
async def test_range_validation(workflow):
    # Below min
    with pytest.raises(ValueError, match="is below minimum 64"):
        workflow.EmptyLatentImage(width=32)

    # Above max
    with pytest.raises(ValueError, match="is above maximum 4096"):
        workflow.EmptyLatentImage(height=5000)

    # Valid
    workflow.EmptyLatentImage(width=1024)

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
async def test_multi_enum_validation(workflow):
    # Valid
    sampler = workflow.SamplerCustom(sampler_name="euler_ancestral", scheduler="karras")
    assert sampler._node.inputs["sampler_name"] == "euler_ancestral"
    assert sampler._node.inputs["scheduler"] == "karras"

    # Invalid sampler_name
    with pytest.raises(ValueError, match="Invalid value 'ddim' for 'sampler_name'"):
        workflow.SamplerCustom(sampler_name="ddim")

    # Invalid scheduler
    with pytest.raises(ValueError, match="Invalid value 'exponential' for 'scheduler'"):
        workflow.SamplerCustom(scheduler="exponential")

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
