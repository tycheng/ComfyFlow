import pytest
from unittest.mock import MagicMock
from comfyflow import Workflow
from comfyflow.registry import SchemaRegistry

MOCK_SCHEMA = {
    "NodeA": {
        "input": {"required": {}},
        "output": ["OUT"],
        "output_name": ["out"],
        "name": "NodeA"
    },
    "NodeB": {
        "input": {"required": {"in_a": ["OUT", {}]}},
        "output": ["OUT"],
        "output_name": ["out"],
        "name": "NodeB"
    },
    "NodeC": {
        "input": {"required": {"in_b": ["OUT", {}]}},
        "output": ["OUT"],
        "output_name": ["out"],
        "name": "NodeC"
    }
}

@pytest.fixture
def workflow():
    mock_cli = MagicMock()
    mock_cli.registry = SchemaRegistry(MOCK_SCHEMA)
    return Workflow(mock_cli)

def test_linear_layout(workflow):
    a = workflow.NodeA()
    b = workflow.NodeB(in_a=a)
    c = workflow.NodeC(in_b=b)

    ui_json = workflow.to_ui_json()
    
    # NodeA should be at X=0
    assert ui_json["nodes"][0]["pos"][0] == 0
    # NodeB should be at X=300
    assert ui_json["nodes"][1]["pos"][0] == 300
    # NodeC should be at X=600
    assert ui_json["nodes"][2]["pos"][0] == 600

def test_manual_pos_override(workflow):
    a = workflow.NodeA()
    a._node.pos = [100, 100]
    b = workflow.NodeB(in_a=a)

    ui_json = workflow.to_ui_json()
    
    # NodeA should keep its manual position
    assert ui_json["nodes"][0]["pos"] == [100, 100]
    # NodeB should still be auto-laid out based on depth=1
    assert ui_json["nodes"][1]["pos"][0] == 300

def test_branching_layout(workflow):
    a = workflow.NodeA()
    b1 = workflow.NodeB(in_a=a)
    b2 = workflow.NodeB(in_a=a)
    
    ui_json = workflow.to_ui_json()
    
    # b1 and b2 should be in the same column
    assert ui_json["nodes"][1]["pos"][0] == 300
    assert ui_json["nodes"][2]["pos"][0] == 300
    
    # b1 is at y=0, b2 should be below b1 (y1 + height1 + padding)
    y1 = ui_json["nodes"][1]["pos"][1]
    y2 = ui_json["nodes"][2]["pos"][1]
    h1 = ui_json["nodes"][1]["size"][1]
    
    assert y2 >= y1 + h1 + 50 # 50 is the padding_y in export.py
