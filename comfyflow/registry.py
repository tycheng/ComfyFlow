from typing import Dict, Any
from .models import NodeSchema

class SchemaRegistry:

    def __init__(self, data: Dict[str, Any]):
        self.nodes: Dict[str, NodeSchema] = {}
        for name, info in data.items():
            inputs = info.get("input", {})
            required = inputs.get("required", {})
            optional = inputs.get("optional", {})

            outputs = []
            output_types = info.get("output", [])
            output_names = info.get("output_name", [])

            # if output_name is missing or shorter than output, use type as name
            for i, type_name in enumerate(output_types):
                name_val = output_names[i] if i < len(output_names) else type_name
                outputs.append((name_val, type_name))

            self.nodes[name] = NodeSchema(
                name=name,
                required_inputs=required,
                optional_inputs=optional,
                outputs=outputs,
                category=info.get("category", ""),
                display_name=info.get("display_name", name)
            )

    def get(self, name: str) -> NodeSchema:
        if not self.nodes:
            raise RuntimeError("Registry not initialized. Call fetch() first.")
        if name not in self.nodes:
            raise AttributeError(f"Node '{name}' not found in ComfyUI registry.")
        return self.nodes[name]
