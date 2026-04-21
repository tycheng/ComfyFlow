import os
import argparse
from PIL import Image
from random import randint
from comfyflow import AsyncComfyClient, ComfyClient, Workflow

# support directly passing in a path on disk,
# images are automatically uploaded to ComfyUI instance
def load_image():
    path = os.path.abspath(__file__)
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "to-upload.png")

# also support passing in a PIL Image
def load_image_object():
    return Image.open(load_image())

# common workflow builder
def build_workflow(cli):
    wf = Workflow(cli)

    # nodes are dynamically available based on your ComfyUI instance's schema
    ckpt   = wf.CheckpointLoaderSimple(ckpt_name=cli.checkpoints[0])
    pos    = wf.CLIPTextEncode(text="a girl", clip=ckpt.CLIP)
    neg    = wf.CLIPTextEncode(text="bad quality", clip=ckpt.CLIP)
    image  = wf.LoadImage(image=load_image())
    latent = wf.VAEEncode(pixels=image.IMAGE, vae=ckpt.VAE)
    sample = wf.KSampler(
        model=ckpt.MODEL,
        positive=pos,
        negative=neg,
        latent_image=latent,
        steps=20,
        cfg=5.5,
        seed=randint(0, 0xffffffff),
        sampler_name="euler",
        scheduler="simple",
        denoise=0.9
    )
    image = wf.VAEDecode(samples=sample, vae=ckpt.VAE)
    wf.PreviewImage(images=image)
    return wf

def sync_main():
    cli = ComfyClient.create()
    wf = build_workflow(cli)
    for node_id, image in cli.run(wf):
        print(f"Received image from node {node_id}")
        image.show()

async def async_main():
    cli = await AsyncComfyClient.create()
    wf = build_workflow(cli)
    async for node_id, image in cli.run(wf):
        print(f"Received image from node {node_id}")
        image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", dest="async_mode", action="store_true", help="run in async mode")
    args = parser.parse_args()
    if args.async_mode:
        import asyncio
        asyncio.run(async_main())
    else:
        sync_main()
