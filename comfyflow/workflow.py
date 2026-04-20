from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from PIL import Image
from .client import ComfyClient
from .models import NodeSchema, NodeInstance, NodeOutputs

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
        # validate input keys and ensure all required inputs are present
        for key in self.schema.required_inputs:
            if key not in kwargs:
                raise ValueError(f"Node '{self.schema.name}' requires input '{key}'")

        for key in kwargs:
            if key not in self.schema.inputs:
                raise ValueError(f"Node '{self.schema.name}' does not have input '{key}'")

        # strict validation of enum/list values
        for key, value in kwargs.items():
            input_info = self.schema.inputs[key]
            if isinstance(input_info[0], list):
                allowed_values = input_info[0]

                # special case for image upload: allow local resources even if not in allowed_values
                if is_image_upload_field(input_info) and is_local_resource(value):
                    continue

                if value not in allowed_values:
                    raise ValueError(f"Invalid value '{value}' for '{key}' in node '{self.schema.name}'. Allowed: {allowed_values}")

        node_id = str(len(self.workflow.nodes) + 1)
        node = NodeInstance(id=node_id, schema=self.schema, inputs=kwargs)

        # basic auto-layout: stagger nodes horizontally
        node.pos = [200 * (len(self.workflow.nodes) % 5), 200 * (len(self.workflow.nodes) // 5)]

        self.workflow.nodes.append(node)
        return NodeOutputs(node)

class Workflow:

    def __init__(self, client: ComfyClient):
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

    async def run(self):
        async for output in self.client.run(self):
            yield output

    def iter_uploads(self):
        for node in self.nodes:
            for key, value in node.inputs.items():
                input_info = node.schema.inputs.get(key)
                if is_image_upload_field(input_info) and is_local_resource(value):
                    yield node, key, value
