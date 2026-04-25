from typing import Any, Dict
from .models import OutputRef, NodeOutputs

def resolve_input(value: Any) -> Any:
    if isinstance(value, NodeOutputs):
        # default to the first output if the whole NodeOutputs object is passed
        return value[0]
    return value

def to_api_json(workflow) -> Dict[str, Any]:
    api_format = {}
    for node in workflow.nodes:
        # ensure ID is a string of an integer
        node_id = str(int(node.id))
        node_data = {
            "class_type": node.schema.name,
            "inputs": {}
        }
        for key, raw_value in node.inputs.items():
            value = resolve_input(raw_value)
            if isinstance(value, OutputRef):
                # ensure connection format is [node_id, slot_index]
                node_data["inputs"][key] = [str(int(value.node.id)), value.slot]
            else:
                node_data["inputs"][key] = value
        api_format[node_id] = node_data
    return api_format

def to_ui_json(workflow) -> Dict[str, Any]:
    # standard comfyui .json format is complex, this is a simplified version
    # that many loaders can still handle, or used as a base.
    ui_format = {
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }

    links = []
    link_id_counter = 1

    # map node_id -> node_ui for easy access
    node_map = {}
    max_node_id = 0

    for node in workflow.nodes:
        node_id_int = int(node.id)
        max_node_id = max(max_node_id, node_id_int)
        node_ui = {
            "id": node_id_int,
            "type": node.schema.name,
            "pos": node.pos,
            "size": [210, 80], # default size
            "flags": {},
            "order": node_id_int,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
            "widgets_values": []
        }

        # outputs
        for _, (name, type_name) in enumerate(node.schema.outputs):
            node_ui["outputs"].append({
                "name": name,
                "type": type_name,
                "links": []
            })

        node_map[node.id] = node_ui
        ui_format["nodes"].append(node_ui)

    # second pass to create links and fill inputs/outputs
    for node in workflow.nodes:
        node_ui = node_map[node.id]

        for key, raw_value in node.inputs.items():
            value = resolve_input(raw_value)
            if isinstance(value, OutputRef):
                link_id = link_id_counter
                link_id_counter += 1

                # the target slot is the index in the node's "inputs" array in UI
                input_slot = len(node_ui["inputs"])

                links.append([
                    link_id,
                    int(value.node.id),
                    value.slot,
                    int(node.id),
                    input_slot,
                    value.type
                ])

                node_ui["inputs"].append({
                    "name": key,
                    "type": value.type,
                    "link": link_id
                })

                # also register this link in the output node
                origin_node_ui = node_map[value.node.id]
                if value.slot < len(origin_node_ui["outputs"]):
                    origin_node_ui["outputs"][value.slot]["links"].append(link_id)
            else:
                # widget value
                node_ui["widgets_values"].append(value)

    ui_format["links"] = links
    ui_format["last_node_id"] = max_node_id
    ui_format["last_link_id"] = link_id_counter - 1
    return ui_format
