import argparse
from random import randint
from comfyflow import ComfyClient, Workflow

def build_workflow(cli, args):
    wf = Workflow(cli)

    # 1. Load Models
    # DualCLIPLoader
    clip = wf.DualCLIPLoader(
        clip_name1=args.clip_l,
        clip_name2=args.t5xxl,
        type="flux",
        device="default"
    )

    # VAELoader
    vae = wf.VAELoader(vae_name=args.vae_name)

    # UNETLoader
    model = wf.UNETLoader(
        unet_name=args.unet_name,
        weight_dtype="default"
    )

    # 2. Load and Prepare Image
    # LoadImage handles local paths and uploads them automatically
    image = wf.LoadImage(image=args.image)

    # ImageStitch
    stitch = wf.ImageStitch(
        direction="right",
        match_image_size=True,
        spacing_width=0,
        spacing_color="white",
        image1=image.IMAGE
    )

    # FluxKontextImageScale
    scale = wf.FluxKontextImageScale(image=stitch.IMAGE)

    # 3. Conditioning
    # CLIPTextEncode (Positive Prompt)
    pos = wf.CLIPTextEncode(
        text=args.prompt,
        clip=clip.CLIP
    )

    # ConditioningZeroOut (Negative Prompt)
    neg = wf.ConditioningZeroOut(conditioning=pos.CONDITIONING)

    # VAEEncode
    latent = wf.VAEEncode(pixels=scale.IMAGE, vae=vae.VAE)

    # ReferenceLatent
    ref_latent = wf.ReferenceLatent(
        conditioning=pos.CONDITIONING,
        latent=latent.LATENT
    )

    # FluxGuidance
    guidance = wf.FluxGuidance(
        guidance=args.guidance,
        conditioning=ref_latent.CONDITIONING
    )

    # 4. Sampling
    sampler = wf.KSampler(
        seed=args.seed if args.seed is not None else randint(0, 0xffffffff),
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        denoise=args.denoise,
        model=model.MODEL,
        positive=guidance.CONDITIONING,
        negative=neg.CONDITIONING,
        latent_image=latent.LATENT
    )

    # 5. Decode and Preview
    decoded = wf.VAEDecode(samples=sampler.LATENT, vae=vae.VAE)
    wf.PreviewImage(images=decoded.IMAGE)

    return wf

def on_progress(node_id, node_type, current, total, is_step):
    if is_step:
        print(f"  - Node {node_id} progress: {current}/{total}")
    else:
        print(f"[{current}/{total}] Executing: {node_type} (ID: {node_id})")

def sync_main(args):
    cli = ComfyClient.create(args.server)
    wf = build_workflow(cli, args)
    print("Running workflow...")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    for node_id, image in cli.run(wf, on_progress=on_progress):
        print(f"Received image from node {node_id}")
        image.show()
        if args.output:
            import uuid
            path = os.path.join(args.output, f"flux_kontext_{uuid.uuid4().hex[:8]}.png")
            image.save(path)
            print(f"Saved image to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Kontext example using ComfyFlow")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--output", help="Directory to save output images")

    # Model arguments
    parser.add_argument("--clip_l", default="clip_l.safetensors", help="CLIP-L model name")
    parser.add_argument("--t5xxl", default="t5xxl_fp8_e4m3fn_scaled.safetensors", help="T5XXL model name")
    parser.add_argument("--vae_name", default="flux-kontext-ae.safetensors", help="VAE model name")
    parser.add_argument("--unet_name", default="redcraftCADSUpdatedJUN29_redKKingOfHearts.safetensors", help="UNET model name")

    # Generation arguments
    parser.add_argument("--image", default="examples/data/to-upload.png", help="Path to input image")
    parser.add_argument("--prompt", default="remove her forehair (bangs) and show her forehead.", help="Positive prompt")
    parser.add_argument("--guidance", type=float, default=2.5, help="Flux guidance scale")
    parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--sampler", default="euler", help="Sampler name")
    parser.add_argument("--scheduler", default="simple", help="Scheduler name")
    parser.add_argument("--denoise", type=float, default=1.0, help="Denoise strength")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Ensure data directory exists if using default image
    import os
    if args.image == "examples/data/to-upload.png" and not os.path.exists(args.image):
        print(f"Warning: Default image {args.image} not found. Please provide an image with --image.")

    sync_main(args)
