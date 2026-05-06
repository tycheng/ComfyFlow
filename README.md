# ComfyFlow

ComfyFlow is a Python package for building and executing ComfyUI node graphs programmatically. It provides a fluent, type-safe API that is dynamically generated from your actual ComfyUI instance's schema.

## Features

1. **Fluent API**: Build ComfyUI workflows using a natural, method-call syntax. No more manual JSON construction.
2. **Live Previews**: The `run()` method is an async generator that yields images as they are ready, including intermediate `PreviewImage` snapshots.
3. **Model Discovery**: Easily access lists of available `checkpoints`, `loras`, `vaes`, and `diffusion_models` directly from the client.
4. **Smart Validation**:
   - **Strict Keyword Check**: Ensures all keyword arguments match the node schema.
   - **Default Value Support**: Automatically fills in missing arguments with schema defaults.
   - **Range Validation**: Validates `min`/`max` constraints for numeric inputs.
   - **Enum Validation**: Ensures list-based options (like samplers or schedulers) are valid.
5. **Dynamic Schema**: Queries your local ComfyUI instance to automatically support all installed custom nodes and extensions.
6. **Export Options**:
   - Export to API JSON format for direct ComfyUI execution.
   - Export to UI JSON format with basic auto-layout for manual editing in the ComfyUI web interface.

## Installation

### From PyPI

[pypi](https://pypi.org/project/comfyflow/)
```
uv add comfyflow
```

### From Git (Latest)
To install the latest development version directly from GitHub using `uv`:
```bash
uv add git+https://github.com/TianyuCheng/ComfyFlow.git
```

## Usage Example

```python
from comfyflow import ComfyClient, Workflow

async def main():
    # 1. Initialize the client (connects to ComfyUI)
    cli = ComfyClient("127.0.0.1:8188")
    await cli.init()

    # 2. Access available models
    print(f"Checkpoints: {cli.checkpoints}")

    # 3. Create a workflow
    wf = Workflow(cli)

    # Nodes are dynamically available based on your ComfyUI instance
    ckpt = wf.CheckpointLoaderSimple(
        ckpt_name=cli.checkpoints[0]
    )

    pos = wf.CLIPTextEncode(
        text="a high-tech laboratory with glowing blue lights",
        clip=ckpt.CLIP
    )

    neg = wf.CLIPTextEncode(
        text="blurry, low quality",
        clip=ckpt.CLIP
    )

    latent = wf.EmptyLatentImage(
        width=512,
        height=512,
        batch_size=1
    )

    sample = wf.KSampler(
        model=ckpt.MODEL,
        positive=pos.CONDITIONING,
        negative=neg.CONDITIONING,
        latent_image=latent.LATENT,
        steps=20,
        cfg=7.5,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0
    )

    image = wf.VAEDecode(
        samples=sample.LATENT,
        vae=ckpt.VAE
    )

    # Optional: Add a preview node to get live updates during sampling
    wf.PreviewImage(images=image.IMAGE)

    # 4. Run the workflow and receive images as they are ready
    async for node_id, image in wf.run():
        # 'node_id' is the string ID of the node that produced the image
        # 'image' is a PIL.Image object
        print(f"Received image from node {node_id}")
        image.show()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Advanced Usage

### Accessing Node Outputs
Each node call returns a `NodeOutputs` object. You can access specific output slots by name (uppercase, e.g., `ckpt.MODEL`) or by index (e.g., `ckpt[0]`).

### Exporting Workflows
```python
# Export to API format (for /prompt endpoint)
api_json = wf.to_api_json()

# Export to UI format (for loading into ComfyUI browser)
ui_json = wf.to_ui_json()
```

## How it works

ComfyFlow queries `http://127.0.0.1:8188/object_info` to understand the schema of all available nodes. It then uses this information to provide a dynamic API where each node is a method on the `Workflow` class.

The `run()` method establishes a WebSocket connection to ComfyUI, allowing it to stream live progress and binary image data directly to your Python application.
