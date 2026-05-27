from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Union
from .client import ComfyClient, AsyncComfyClient
from .models import NodeSchema, NodeInstance, NodeOutputs, OutputRef
from .media import HAS_MOVIEPY, VideoClip

def is_media_upload_field(input_info: Any) -> bool:
    """
    Checks if a node input field is designated for image or video upload.
    ComfyUI nodes often flag these in their metadata.
    """
    if not isinstance(input_info, (list, tuple)) or len(input_info) < 2:
        return False
    metadata = input_info[1]
    if not isinstance(metadata, dict):
        return False
    return metadata.get("image_upload") is True or metadata.get("video_upload") is True

def is_local_resource(value: Any) -> bool:
    """
    Determines if a value represents a local media resource that needs to be uploaded.
    Supported types: file paths, raw bytes, PIL Images, and MoviePy VideoClips.
    """
    if isinstance(value, (Path, bytes, Image.Image)):
        return True
    if HAS_MOVIEPY and isinstance(value, VideoClip):
        return True
    if isinstance(value, str):
        try:
            p = Path(value)
            return p.exists() and p.is_file()
        except:
            return False
    return False

class NodeFactory:
    """
    A factory for creating NodeInstance objects based on a specific ComfyUI node schema.
    Handles input validation, default values, and dynamic input keys.
    """

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
            if key in self.schema.inputs:
                continue

            # support dynamic inputs with dot notation (e.g. resize_type.width)
            # Some custom nodes use dynamic parameters based on a mode selector.
            if "." in key:
                prefix = key.split(".")[0]
                if prefix in self.schema.inputs:
                    continue

            raise ValueError(f"Node '{self.schema.name}' does not have input '{key}'")

        # 4. strict validation of enum/list values and ranges
        for key, value in final_inputs.items():
            # Skip validation for connections (OutputRef)
            if isinstance(value, (OutputRef, NodeOutputs, NodeInstance)):
                continue

            input_info = self.schema.inputs.get(key)
            if not input_info or not isinstance(input_info, (list, tuple)):
                continue

            type_info = input_info[0]
            metadata = input_info[1] if len(input_info) > 1 else {}

            # enum validation (lists in schema represent selectable options)
            if isinstance(type_info, list):
                allowed_values = type_info

                # special case for media upload: allow local resources even if not in allowed_values
                # ComfyUI often lists filenames, but we want to allow passing local objects/paths.
                if is_media_upload_field(input_info) and is_local_resource(value):
                    continue

                if value not in allowed_values:
                    raise ValueError(f"Invalid value '{value}' for '{key}' in node '{self.schema.name}'. Allowed: {allowed_values}")

            # range validation for numeric inputs
            if isinstance(value, (int, float)) and isinstance(metadata, dict):
                if "min" in metadata and value < metadata["min"]:
                    raise ValueError(f"Value {value} for '{key}' in node '{self.schema.name}' is below minimum {metadata['min']}")
                if "max" in metadata and value > metadata["max"]:
                    raise ValueError(f"Value {value} for '{key}' in node '{self.schema.name}' is above maximum {metadata['max']}")

        node_id = str(len(self.workflow.nodes) + 1)

        # include all final_inputs to support dynamic keys (essential for complex custom nodes)
        node = NodeInstance(id=node_id, schema=self.schema, inputs=final_inputs)

        # default position, will be calculated during export if not changed
        node.pos = [0.0, 0.0]

        self.workflow.nodes.append(node)
        return NodeOutputs(node)

class Workflow:
    """
    Represents a ComfyUI workflow (a collection of connected nodes).
    Provides methods for constructing the graph and exporting it to JSON formats.
    """

    def __init__(self, client: Union[ComfyClient, AsyncComfyClient]):
        self.nodes: List[NodeInstance] = []
        self.client = client

    def __getattr__(self, name: str) -> NodeFactory:
        """Dynamically creates a NodeFactory based on the node names in the registry."""
        schema = self.client.registry.get(name)
        return NodeFactory(self, schema)

    def to_api_json(self) -> Dict[str, Any]:
        """Exports the workflow to the ComfyUI API JSON format (suitable for /prompt)."""
        from .export import to_api_json
        return to_api_json(self)

    def to_ui_json(self) -> Dict[str, Any]:
        """Exports the workflow to the ComfyUI UI JSON format (suitable for browser loading)."""
        from .export import to_ui_json
        return to_ui_json(self)

    def iter_uploads(self):
        """Iterates over all nodes and inputs to find local resources that need uploading."""
        for node in self.nodes:
            for key, value in node.inputs.items():
                input_info = node.schema.inputs.get(key)
                if is_media_upload_field(input_info) and is_local_resource(value):
                    yield node, key, value
