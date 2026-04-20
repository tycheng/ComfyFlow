from random import randint
from comfyflow import ComfyClient, Workflow

async def main():
    cli = ComfyClient()

    await cli.init()

    wf = Workflow(cli)

    # nodes are dynamically available based on your ComfyUI instance's schema
    ckpt   = wf.CheckpointLoaderSimple(ckpt_name=cli.checkpoints[0])
    pos    = wf.CLIPTextEncode(text="a cat", clip=ckpt.CLIP)
    neg    = wf.CLIPTextEncode(text="bad quality", clip=ckpt.CLIP)
    latent = wf.EmptyLatentImage(width=512, height=512, batch_size=1)
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
        denoise=1.0
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
