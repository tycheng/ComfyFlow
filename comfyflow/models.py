from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

@dataclass
class NodeSchema:
    name: str
    required_inputs: Dict[str, Any]
    optional_inputs: Dict[str, Any]
    outputs: List[Tuple[str, str]]
    category: str = ""
    display_name: str = ""

    @property
    def inputs(self):
        return {**self.required_inputs, **self.optional_inputs}

@dataclass
class NodeInstance:
    id: str
    schema: NodeSchema
    inputs: Dict[str, Any] = field(default_factory=dict)
    # used for layout
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0])
    size: List[float] = field(default_factory=lambda: [210.0, 80.0])

@dataclass
class OutputRef:
    node: NodeInstance
    slot: int
    name: str
    type: str

class NodeOutputs:

    def __init__(self, node: NodeInstance):
        self._node = node
        # mapping output name to its slot index and type
        self._outputs = {name: (i, type) for i, (name, type) in enumerate(node.schema.outputs)}

    def __getattr__(self, name: str) -> OutputRef:
        if name in self._outputs:
            slot, type = self._outputs[name]
            return OutputRef(self._node, slot, name, type)
        available_names = ", ".join(self._outputs)
        raise AttributeError(f"Node '{self._node.schema.name}' has no output named '{name}'! Available names: {available_names}")

    def __getitem__(self, index: int) -> OutputRef:
        if 0 <= index < len(self._node.schema.outputs):
            name, type = self._node.schema.outputs[index]
            return OutputRef(self._node, index, name, type)
        raise IndexError(f"Node '{self._node.schema.name}' output index {index} out of range! Available slot indices: [0, {len(self._node.schema.outputs)}]")
