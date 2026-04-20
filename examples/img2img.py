import os
from PIL import Image
from random import randint
from comfyflow import ComfyClient, Workflow

# support directly passing in a path on disk,
# images are automatically uploaded to ComfyUI instance
def load_image():
    path = os.path.abspath(__file__)
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "to-upload.png")

# also support passing in a PIL Image
def load_image_object():
    return Image.open(load_image())

async def main():
    cli = ComfyClient()

    await cli.init()

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

    # run the workflow and wait for the result
    async for node_id, image in wf.run():
        print(f"Received image from node {node_id}")
        image.show()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
