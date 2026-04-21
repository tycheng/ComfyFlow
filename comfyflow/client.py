import os
import json
import uuid
import httpx
import struct
import asyncio
import mimetypes
import websockets
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union, Any
from .registry import SchemaRegistry

# async client
class AsyncComfyClient:

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.registry = SchemaRegistry({})
        self.models: Dict[str, List[str]] = {}

    @staticmethod
    async def create(server_address: str = "127.0.0.1:8188"):
        cli = AsyncComfyClient(server_address)
        await cli.init()
        return cli

    @property
    def checkpoints(self) -> List[str]:
        return self.models.get("checkpoints", [])

    @property
    def loras(self) -> List[str]:
        return self.models.get("loras", [])

    @property
    def vaes(self) -> List[str]:
        return self.models.get("vaes", [])

    @property
    def diffusion_models(self) -> List[str]:
        return self.models.get("diffusion_models", [])

    @staticmethod
    def decode_comfy_image(binary_data):
        if len(binary_data) < 8:
            return None

        # read the event type (first 4 bytes)
        event_type = struct.unpack(">I", binary_data[:4])[0]

        # event_type == 1 is for PREVIEW_IMAGE
        if event_type != 1:
            return None

        # extract image data (skip first 8 bytes)
        image_bytes = binary_data[8:]
        return Image.open(BytesIO(image_bytes))

    async def init(self):
        # pre-load models
        model_types = ["checkpoints", "loras", "vaes", "diffusion_models"]
        async with httpx.AsyncClient() as client:
            for m_type in model_types:
                response = await client.get(f"http://{self.server_address}/models/{m_type}")
                if response.status_code == 200:
                    self.models[m_type] = response.json()

        # pre-load schema
        url = f"http://{self.server_address}/object_info"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            self.registry = SchemaRegistry(data)

    async def ensure_images_uploaded(self, workflow):
        for node, key, value in workflow.iter_uploads():
            result = await self.upload_image(value)
            # update node input with the path relative to input folder (name + subfolder)
            if result.get("subfolder"):
                node.inputs[key] = os.path.join(result['subfolder'], result['name'])
            else:
                node.inputs[key] = result["name"]

    async def upload_image(
        self,
        image: Union[str, Path, bytes, Image.Image],
        subfolder: str = "comfyflow",
        type: str = "input"
    ) -> Dict[str, Any]:
        url = f"http://{self.server_address}/upload/image"

        filename = None
        if isinstance(image, (str, Path)):
            path = Path(image)
            filename = path.name
            content = open(path, "rb")
            mime_type = mimetypes.guess_type(filename)[0] or "image/png"
        elif isinstance(image, Image.Image):
            filename = f"upload_{uuid.uuid4()}.png"
            fmt = "PNG"

            buf = BytesIO()
            image.save(buf, format=fmt)
            buf.seek(0)
            content = buf
            mime_type = "image/png"
        else:
            filename = f"upload_{uuid.uuid4()}.png"
            content = BytesIO(image)
            mime_type = "image/png"

        files = {"image": (filename, content, mime_type)}
        data = {"overwrite": "true", "type": type, "subfolder": subfolder}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()

    async def run(self, workflow):
        await self.ensure_images_uploaded(workflow)
        prompt = workflow.to_api_json()
        node_types = {node.id: node.schema.name for node in workflow.nodes}

        async def get_prompt_id(client):
            response = await client.post(
                f"http://{self.server_address}/prompt",
                json={"prompt": prompt, "client_id": self.client_id}
            )
            if response.status_code != 200:
                raise RuntimeError(f"ComfyUI Error: {response.json()}")
            return response.json()["prompt_id"]

        async def fetch_images(client, output_data):
            for img_info in output_data.get("images", []):
                img_res = await client.get(
                    f"http://{self.server_address}/view",
                    params={
                        "filename": img_info["filename"],
                        "subfolder": img_info["subfolder"],
                        "type": img_info["type"]
                    }
                )
                if img_res.status_code == 200:
                    yield Image.open(BytesIO(img_res.content))

        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        async with httpx.AsyncClient() as client, websockets.connect(ws_url) as ws:
            prompt_id = await get_prompt_id(client)
            current_node_id = None

            async for message in ws:
                if not isinstance(message, str):
                    # decode binary image (PreviewImage)
                    if current_node_id and node_types.get(str(current_node_id)) == "PreviewImage":
                        image = AsyncComfyClient.decode_comfy_image(message)
                        if image:
                            yield str(current_node_id), image
                    continue

                msg = json.loads(message)
                if msg["type"] == "executing":
                    current_node_id = msg["data"]["node"]
                    if current_node_id is None and msg["data"]["prompt_id"] == prompt_id:
                        break # execution finished

                if msg["type"] == "executed" and msg["data"]["prompt_id"] == prompt_id:
                    node_id = msg["data"]["node"]
                    output = msg["data"]["output"]
                    if output:
                        async for image in fetch_images(client, output):
                            yield str(node_id), image

# sync client
class ComfyClient:

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.wrapper = AsyncComfyClient(server_address)
        asyncio.run(self.wrapper.init())

    @staticmethod
    def create(server_address: str = "127.0.0.1:8188"):
        return ComfyClient(server_address)

    @property
    def registry(self) -> SchemaRegistry:
        return self.wrapper.registry

    @property
    def checkpoints(self) -> List[str]:
        return self.wrapper.checkpoints

    @property
    def loras(self) -> List[str]:
        return self.wrapper.loras

    @property
    def vaes(self) -> List[str]:
        return self.wrapper.vaes

    @property
    def diffusion_models(self) -> List[str]:
        return self.wrapper.diffusion_models

    def run(self, workflow):
        async def run_and_yield():
            async for node_id, image in self.wrapper.run(workflow):
                yield node_id, image

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gen = run_and_yield()
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
