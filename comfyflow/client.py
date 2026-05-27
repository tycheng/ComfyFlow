import os
import json
import uuid
import httpx
import struct
import asyncio
import mimetypes
import tempfile
import websockets
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union, Any, Tuple
from .registry import SchemaRegistry
from .media import HAS_MOVIEPY, VideoClip

# async client
class AsyncComfyClient:
    """
    Asynchronous client for interacting with a ComfyUI server.
    Handles model discovery, schema loading, file uploads, and workflow execution.
    """

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.registry = SchemaRegistry({})
        self.models: Dict[str, List[str]] = {}

    @staticmethod
    async def create(server_address: str = "127.0.0.1:8188"):
        """Factory method to create and initialize an AsyncComfyClient."""
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
    def vae(self) -> List[str]:
        return self.models.get("vae", [])

    @property
    def diffusion_models(self) -> List[str]:
        return self.models.get("diffusion_models", [])

    @property
    def text_encoders(self) -> List[str]:
        return self.models.get("text_encoders", [])

    @staticmethod
    def decode_comfy_image(binary_data):
        """Decodes binary image data received via WebSocket (PREVIEW_IMAGE)."""
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
        """Initializes the client by fetching available models and the node schema registry."""
        # pre-load models
        model_types = ["checkpoints", "loras", "vae", "diffusion_models", "text_encoders"]
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

    async def run(self, workflow, on_progress=None):
        """
        Executes the workflow on the server.
        Yields (node_id, media) pairs as outputs are produced.
        """
        await self.ensure_media_uploaded(workflow)
        prompt = workflow.to_api_json()
        total_nodes = len(prompt)
        nodes_executed = 0
        node_types = {node.id: node.schema.name for node in workflow.nodes}

        async def get_prompt_id(client):
            response = await client.post(
                f"http://{self.server_address}/prompt",
                json={"prompt": prompt, "client_id": self.client_id}
            )
            if response.status_code != 200:
                raise RuntimeError(f"ComfyUI Error: {response.json()}")
            return response.json()["prompt_id"]

        def _call_progress(node_id, current, total, is_step):
            if on_progress:
                import inspect
                node_type = node_types.get(str(node_id)) if node_id else None
                if inspect.iscoroutinefunction(on_progress):
                    return on_progress(str(node_id), node_type, current, total, is_step)
                else:
                    on_progress(str(node_id), node_type, current, total, is_step)
            return None

        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        async with httpx.AsyncClient() as client, websockets.connect(ws_url) as ws:
            prompt_id = await get_prompt_id(client)
            current_node_id = None

            async for message in ws:
                if not isinstance(message, str):
                    # binary message: likely a PreviewImage output
                    if current_node_id and node_types.get(str(current_node_id)) == "PreviewImage":
                        image = AsyncComfyClient.decode_comfy_image(message)
                        if image:
                            yield str(current_node_id), image
                    continue

                msg = json.loads(message)
                if msg["type"] == "execution_cached":
                    data = msg["data"]
                    if data["prompt_id"] == prompt_id:
                        cached_nodes = data.get("nodes", [])
                        total_nodes = max(1, total_nodes - len(cached_nodes))

                if msg["type"] == "executing":
                    data = msg["data"]
                    if data["prompt_id"] != prompt_id:
                        continue

                    current_node_id = data["node"]
                    if current_node_id is None:
                        break # execution finished

                    nodes_executed += 1
                    res = _call_progress(current_node_id, nodes_executed, total_nodes, is_step=False)
                    if res: await res

                if msg["type"] == "progress":
                    data = msg["data"]
                    if data["prompt_id"] == prompt_id:
                        res = _call_progress(current_node_id, data["value"], data["max"], is_step=True)
                        if res: await res

                if msg["type"] == "executed" and msg["data"]["prompt_id"] == prompt_id:
                    node_id = msg["data"]["node"]
                    output = msg["data"]["output"]
                    if output:
                        async for media in self.fetch_media(client, output):
                            yield str(node_id), media

                if msg["type"] == "execution_error":
                    if msg["data"]["prompt_id"] == prompt_id:
                        raise RuntimeError(f"ComfyUI Execution Error: {msg['data']}")

                if msg["type"] == "execution_interrupted":
                    if msg["data"]["prompt_id"] == prompt_id:
                        raise RuntimeError("ComfyUI Execution Interrupted")

    async def ensure_media_uploaded(self, workflow):
        """Uploads all local media resources referenced in the workflow to the server."""
        for node, key, value in workflow.iter_uploads():
            result = await self.upload_media(value)
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
        """Uploads an image (path, bytes, or PIL Image) to the server."""
        filename = None
        if isinstance(image, (str, Path)):
            path = Path(image)
            filename = path.name
            content = open(path, "rb")
            mime_type = mimetypes.guess_type(filename)[0] or "image/png"
        elif isinstance(image, Image.Image):
            filename = f"upload_{uuid.uuid4()}.png"
            buf = BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            content = buf
            mime_type = "image/png"
        else:
            filename = f"upload_{uuid.uuid4()}.png"
            content = BytesIO(image)
            mime_type = "image/png"

        try:
            return await self.upload_file(filename, content, mime_type, subfolder, type)
        finally:
            if hasattr(content, "close"):
                content.close()

    async def upload_video(
        self,
        video: Union[str, Path, 'VideoClip'],
        subfolder: str = "comfyflow",
        type: str = "input"
    ) -> Dict[str, Any]:
        """Uploads a video (path or MoviePy VideoClip) to the server."""
        filename = None
        cleanup_temp = None
        if isinstance(video, (str, Path)):
            path = Path(video)
            filename = path.name
            content = open(path, "rb")
            mime_type = mimetypes.guess_type(filename)[0] or "video/mp4"
        elif HAS_MOVIEPY and isinstance(video, VideoClip):
            # For VideoClips, we write to a temporary file before uploading
            filename = f"upload_{uuid.uuid4()}.mp4"
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            video.write_videofile(temp_path, logger=None)
            content = open(temp_path, "rb")
            mime_type = "video/mp4"
            cleanup_temp = temp_path
        else:
            raise ValueError(f"Unsupported video type: {type(video)}")

        try:
            return await self.upload_file(filename, content, mime_type, subfolder, type)
        finally:
            if hasattr(content, "close"):
                content.close()
            if cleanup_temp and os.path.exists(cleanup_temp):
                os.remove(cleanup_temp)

    async def upload_file(
        self,
        filename: str,
        content: Any,
        mime_type: str,
        subfolder: str,
        type: str
    ) -> Dict[str, Any]:
        """Internal helper to perform the actual HTTP upload to ComfyUI."""
        url = f"http://{self.server_address}/upload/image"
        files = {"image": (filename, content, mime_type)}
        data = {"overwrite": "true", "type": type, "subfolder": subfolder}

        async with httpx.AsyncClient() as client:
            # ComfyUI uses the same /upload/image endpoint for both images and videos
            response = await client.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()

    async def upload_media(
        self,
        media: Union[str, Path, bytes, Image.Image, 'VideoClip'],
        subfolder: str = "comfyflow",
        type: str = "input"
    ) -> Dict[str, Any]:
        """
        High-level caller that dispatches to upload_image or upload_video
        based on the input type and MIME type.
        """
        if isinstance(media, Image.Image):
            return await self.upload_image(media, subfolder, type)

        if HAS_MOVIEPY and isinstance(media, VideoClip):
            return await self.upload_video(media, subfolder, type)

        if isinstance(media, (str, Path)):
            mime_type = mimetypes.guess_type(str(media))[0]
            if mime_type and mime_type.startswith("video"):
                return await self.upload_video(media, subfolder, type)
            else:
                return await self.upload_image(media, subfolder, type)

        # Default to image for bytes or unknown types
        return await self.upload_image(media, subfolder, type)

    async def fetch_media(self, client, output_data):
        """
        Downloads media outputs from the server.
        Yields PIL.Image for images and moviepy.VideoFileClip for videos/gifs.
        """
        # handle images key (may contain images or videos)
        for m_info in output_data.get("images", []):
            try:
                data = await self._view_file(client, m_info["filename"], m_info["subfolder"], m_info["type"])
            except Exception as e:
                print(f"Error fetching image {m_info['filename']}: {e}")
                continue

            # Detect if it's a video hidden in the images key
            is_video = any(m_info["filename"].lower().endswith(ext) for ext in [".mp4", ".webm", ".gif"])

            if is_video:
                yield self._bytes_to_video(data, m_info["filename"])
            else:
                # Try image first
                try:
                    yield Image.open(BytesIO(data))
                except Exception:
                    # Final attempt as video if it failed as image
                    if HAS_MOVIEPY:
                        yield self._bytes_to_video(data, m_info["filename"])
                    else:
                        yield BytesIO(data)

        # handle explicit videos and gifs keys
        for key in ["videos", "gifs"]:
            for m_info in output_data.get(key, []):
                try:
                    data = await self._view_file(client, m_info["filename"], m_info["subfolder"], m_info["type"])
                    yield self._bytes_to_video(data, m_info["filename"])
                except Exception as e:
                    print(f"Error fetching {key[:-1]} {m_info['filename']}: {e}")

    async def _view_file(self, client, filename: str, subfolder: str, type: str) -> bytes:
        """Internal helper to download a file from the server's view endpoint."""
        url = f"http://{self.server_address}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": type}
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.content

    def _bytes_to_image(self, data: bytes, filename: str) -> Union[Image.Image, BytesIO]:
        """Converts raw bytes to a PIL Image or fallback BytesIO."""
        try:
            return Image.open(BytesIO(data))
        except Exception as e:
            print(f"Warning: Failed to open image {filename}: {e}")
            return BytesIO(data)

    def _bytes_to_video(self, data: bytes, filename: str) -> Union['VideoFileClip', BytesIO]:
        """Converts raw bytes to a MoviePy VideoFileClip or fallback BytesIO."""
        if not HAS_MOVIEPY:
            return BytesIO(data)

        try:
            try:
                from moviepy.editor import VideoFileClip
            except ImportError:
                from moviepy.video.io.VideoFileClip import VideoFileClip

            suffix = Path(filename).suffix or ".mp4"
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            return VideoFileClip(temp_path)
        except Exception as e:
            print(f"Warning: Failed to open video {filename}: {e}")
            return BytesIO(data)

# sync client
class ComfyClient:
    """Synchronous wrapper around AsyncComfyClient for easier use in non-async scripts."""

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

    def run(self, workflow, on_progress=None):
        """Runs the workflow synchronously and yields outputs."""
        async def run_and_yield():
            async for node_id, media in self.wrapper.run(workflow, on_progress=on_progress):
                yield node_id, media

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
