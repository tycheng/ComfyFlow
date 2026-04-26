
import pytest
from unittest.mock import MagicMock
from comfyflow import Workflow
from comfyflow.registry import SchemaRegistry

UI_TEST_SCHEMA = {
    "ComplexNode": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "denoise": ["FLOAT", {"default": 1.0}]
            },
            "optional": {
                "model": ["MODEL", {}],
                "positive": ["CONDITIONING", {}]
            }
        },
        "output": ["LATENT"],
        "output_name": ["latent"],
        "name": "ComplexNode"
    },
    "Dummy": {
        "input": {"required": {}},
        "output": ["MODEL", "CONDITIONING"],
        "output_name": ["model", "cond"],
        "name": "Dummy"
    }
}

@pytest.fixture
def workflow():
    mock_cli = MagicMock()
    mock_cli.registry = SchemaRegistry(UI_TEST_SCHEMA)
    return Workflow(mock_cli)

def test_ui_widget_ordering_and_slots(workflow):
    dummy = workflow.Dummy()
    
    # Connect model but leave positive disconnected (if it was optional, but here it's required)
    # Actually let's just connect both to see slots
    node = workflow.ComplexNode(
        model=dummy.model,
        seed=123,
        steps=20,
        cfg=5.5,
        positive=dummy.cond,
        denoise=1.0
    )

    ui_json = workflow.to_ui_json()
    
    # Find the ComplexNode in exported json
    node_ui = next(n for n in ui_json["nodes"] if n["type"] == "ComplexNode")
    
    # 1. Check ports (inputs array)
    # Schema order for ports: model, positive
    assert len(node_ui["inputs"]) == 2
    assert node_ui["inputs"][0]["name"] == "model"
    assert node_ui["inputs"][1]["name"] == "positive"
    
    # 2. Check links slots
    # Find links targeting this node
    node_links = [l for l in ui_json["links"] if l[3] == node_ui["id"]]
    node_links.sort(key=lambda x: x[4]) # sort by target slot
    
    assert node_links[0][4] == 0 # model slot
    assert node_links[1][4] == 1 # positive slot
    
    # 3. Check widgets_values
    # Expected: [seed, control_after_generate, steps, cfg, denoise]
    # "randomize" is added for seed
    expected_widgets = [123, "randomize", 20, 5.5, 1.0]
    assert node_ui["widgets_values"] == expected_widgets

def test_ui_disconnected_ports(workflow):
    dummy = workflow.Dummy()
    
    # Only connect positive, leave model disconnected
    # (In real ComfyUI required ports MUST be connected, but we test UI structure)
    node = workflow.ComplexNode(
        # model NOT connected
        seed=456,
        steps=30,
        cfg=7.0,
        positive=dummy.cond,
        denoise=0.5
    )

    ui_json = workflow.to_ui_json()
    node_ui = next(n for n in ui_json["nodes"] if n["type"] == "ComplexNode")
    
    # Even if model is disconnected, it should be in the inputs array as a port
    assert len(node_ui["inputs"]) == 2
    assert node_ui["inputs"][0]["name"] == "model"
    assert node_ui["inputs"][0]["link"] is None
    assert node_ui["inputs"][1]["name"] == "positive"
    assert node_ui["inputs"][1]["link"] is not None
    
    # The link for positive should still be at slot 1
    link = next(l for l in ui_json["links"] if l[3] == node_ui["id"])
    assert link[4] == 1 # slot 1
