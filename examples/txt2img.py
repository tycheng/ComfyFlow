import argparse
from random import randint
from comfyflow import AsyncComfyClient, ComfyClient, Workflow

def build_workflow(cli):
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
