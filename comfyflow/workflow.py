from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
from .client import ComfyClient, AsyncComfyClient
from .models import NodeSchema, NodeInstance, NodeOutputs, OutputRef

def is_image_upload_field(input_info: Any) -> bool:
    if not isinstance(input_info, (list, tuple)) or len(input_info) < 2:
        return False
    metadata = input_info[1]
    return isinstance(metadata, dict) and metadata.get("image_upload") is True

def is_local_resource(value: Any) -> bool:
    if isinstance(value, (Path, bytes, Image.Image)):
        return True
    if isinstance(value, str):
        try:
            p = Path(value)
            return p.exists() and p.is_file()
        except:
            return False
    return False

class NodeFactory:

    def __init__(self, workflow, schema: NodeSchema):
        self.workflow = workflow
        self.schema = schema

    def __call__(self, **kwargs) -> NodeOutputs:
        final_inputs = kwargs.copy()

        # 1. fill defaults and check required inputs
        for key, input_info in self.schema.required_inputs.items():
            if key not in final_inputs:
                metadata = input_info[1] if isinstance(input_info, (list, tuple)) and len(input_info) > 1 else {}
                if "default" in metadata:
                    final_inputs[key] = metadata["default"]
                else:
                    raise ValueError(f"Node '{self.schema.name}' requires input '{key}'")

        # 2. fill defaults for optional inputs
        for key, input_info in self.schema.optional_inputs.items():
            if key not in final_inputs:
                metadata = input_info[1] if isinstance(input_info, (list, tuple)) and len(input_info) > 1 else {}
                if "default" in metadata:
                    final_inputs[key] = metadata["default"]

        # 3. check for unknown kwargs
        for key in kwargs:
            if key not in self.schema.inputs:
                raise ValueError(f"Node '{self.schema.name}' does not have input '{key}'")

        # 4. strict validation of enum/list values and ranges
        for key, value in final_inputs.items():
            # Skip validation for connections
            if isinstance(value, (OutputRef, NodeOutputs, NodeInstance)):
                continue

            input_info = self.schema.inputs.get(key)
            if not input_info or not isinstance(input_info, (list, tuple)):
                continue

            type_info = input_info[0]
            metadata = input_info[1] if len(input_info) > 1 else {}

            # enum validation
            if isinstance(type_info, list):
                allowed_values = type_info

                # special case for image upload: allow local resources even if not in allowed_values
                if is_image_upload_field(input_info) and is_local_resource(value):
                    continue

                if value not in allowed_values:
                    raise ValueError(f"Invalid value '{value}' for '{key}' in node '{self.schema.name}'. Allowed: {allowed_values}")

            # range validation
            if isinstance(value, (int, float)) and isinstance(metadata, dict):
                if "min" in metadata and value < metadata["min"]:
                    raise ValueError(f"Value {value} for '{key}' in node '{self.schema.name}' is below minimum {metadata['min']}")
                if "max" in metadata and value > metadata["max"]:
                    raise ValueError(f"Value {value} for '{key}' in node '{self.schema.name}' is above maximum {metadata['max']}")

        node_id = str(len(self.workflow.nodes) + 1)

        # maintain schema order for inputs
        ordered_inputs = {}
        for key in self.schema.inputs:
            if key in final_inputs:
                ordered_inputs[key] = final_inputs[key]

        node = NodeInstance(id=node_id, schema=self.schema, inputs=ordered_inputs)

        # basic auto-layout: stagger nodes horizontally
        node.pos = [200 * (len(self.workflow.nodes) % 5), 200 * (len(self.workflow.nodes) // 5)]

        self.workflow.nodes.append(node)
        return NodeOutputs(node)

class Workflow:

    def __init__(self, client: ComfyClient | AsyncComfyClient):
        self.nodes: List[NodeInstance] = []
        self.client = client

    def __getattr__(self, name: str) -> NodeFactory:
        schema = self.client.registry.get(name)
        return NodeFactory(self, schema)

    def to_api_json(self) -> Dict[str, Any]:
        from .export import to_api_json
        return to_api_json(self)

    def to_ui_json(self) -> Dict[str, Any]:
        from .export import to_ui_json
        return to_ui_json(self)

    def iter_uploads(self):
        for node in self.nodes:
            for key, value in node.inputs.items():
                input_info = node.schema.inputs.get(key)
                if is_image_upload_field(input_info) and is_local_resource(value):
                    yield node, key, value
