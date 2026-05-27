import os
import argparse
import uuid
from random import randint
from comfyflow import ComfyClient, Workflow
from comfyflow.media import VideoClip

# LTX-2.3 Image-to-Video Workflow Example.
#
# This script demonstrates animating an input image into a high-quality upscaled video:
# 1. Image Preprocessing: Resizes and compresses the source image for model compatibility.
# 2. Base Pass (I2V): Generates a low-resolution video following the image content.
# 3. Upsampling: Scales the video latent using a spatial upscaler model.
# 4. Refinement Pass: Performs a secondary sampling pass on the upscaled latent.
#
# Example usage:
# python examples/ltx23_i2v.py --server 127.0.0.1:8188 --image examples/data/to-upload.png --prompt "A queen walking forward" --output ./outputs

def build_workflow(cli, args):
    wf = Workflow(cli)

    # --- 1. Model Loading ---
    # Main LTX-2.3 checkpoint.
    ckpt = wf.CheckpointLoaderSimple(ckpt_name=args.ckpt_name)

    # CLIP Loader for text conditioning (Gemma 3).
    clip = wf.LTXAVTextEncoderLoader(
        text_encoder=args.gemma_path,
        ckpt_name=args.ckpt_name,
        device="default"
    )
    
    # Distilled LoRA for high-quality fast sampling.
    lora = wf.LoraLoaderModelOnly(
        model=ckpt.MODEL,
        lora_name=args.lora_name,
        strength_model=0.5
    )
    
    # Spatial upscaler model for Pass 2.
    upscale_model = wf.LatentUpscaleModelLoader(model_name=args.upscale_model)

    # Audio VAE loader for sound generation.
    audio_vae = wf.LTXVAudioVAELoader(ckpt_name=args.ckpt_name)

    # --- 2. Input Image & Preprocessing ---
    # ComfyFlow handles local paths and uploads them automatically.
    image_input = wf.LoadImage(image=args.image)
    
    # Resize Image using dynamic dot-notation keys (e.g. resize_type.width).
    resized_image = wf.ResizeImageMaskNode(
        resize_type="scale dimensions",
        **{
            "resize_type.width": args.width,
            "resize_type.height": args.height,
            "resize_type.crop": "center"
        },
        scale_method="lanczos",
        input=image_input.IMAGE
    )
    
    # Standardize image size for the model.
    resized_longer = wf.ResizeImagesByLongerEdge(
        longer_edge=1536,
        images=resized_image.resized
    )
    
    # Compression preprocessing specific to LTXV.
    preprocessed = wf.LTXVPreprocess(
        img_compression=18,
        image=resized_longer.images
    )

    # --- 3. Conditioning ---
    # Generate text embeddings for prompts.
    pos_encoded = wf.CLIPTextEncode(text=args.prompt, clip=clip.CLIP)
    neg_encoded = wf.CLIPTextEncode(text=args.negative_prompt, clip=clip.CLIP)

    # Combine text conditioning with frame rate metadata.
    cond = wf.LTXVConditioning(
        frame_rate=args.fps,
        positive=pos_encoded.CONDITIONING,
        negative=neg_encoded.CONDITIONING
    )

    # --- 4. Latent Initialization ---
    # Create empty latents for both video and audio.
    latent_video = wf.EmptyLTXVLatentVideo(
        width=args.width // 2,
        height=args.height // 2,
        length=args.length,
        batch_size=1
    )

    latent_audio = wf.LTXVEmptyLatentAudio(
        frames_number=args.length,
        frame_rate=int(args.fps),
        batch_size=1,
        audio_vae=getattr(audio_vae, "Audio VAE")
    )
    
    # Pass 1 Latent Setup (I2V mode: bypass=False).
    pass1_inplace = wf.LTXVImgToVideoInplace(
        vae=ckpt.VAE,
        image=preprocessed.output_image,
        latent=latent_video.LATENT,
        strength=0.7,
        bypass=False # Active I2V mode.
    )

    # Merge AV latents for multimodal sampling.
    pass1_concat = wf.LTXVConcatAVLatent(
        video_latent=pass1_inplace.latent,
        audio_latent=latent_audio.Latent
    )
    
    # Setup crop guides for image-aware conditioning.
    pass1_guides = wf.LTXVCropGuides(
        positive=cond.positive,
        negative=cond.negative,
        latent=pass1_inplace.latent
    )

    # --- 5. Pass 1: Base Sampling ---
    sampler_p1_select = wf.KSamplerSelect(sampler_name="euler_ancestral_cfg_pp")
    sigmas_pass1 = wf.ManualSigmas(sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
    noise_pass1 = wf.RandomNoise(noise_seed=args.seed if args.seed is not None else 519681071352364)

    # Guider manages the model and conditioning.
    guider_pass1 = wf.CFGGuider(
        model=lora.MODEL,
        positive=pass1_guides.positive,
        negative=pass1_guides.negative,
        cfg=args.cfg
    )

    # Execute first sampling pass.
    sampled_pass1 = wf.SamplerCustomAdvanced(
        noise=noise_pass1.NOISE,
        guider=guider_pass1.GUIDER,
        sampler=sampler_p1_select.SAMPLER,
        sigmas=sigmas_pass1.SIGMAS,
        latent_image=pass1_concat.latent
    )

    # --- 6. Upsampling ---
    separate_pass1 = wf.LTXVSeparateAVLatent(av_latent=sampled_pass1.output)
    
    # Perform spatial upscaling.
    upsampled_video = wf.LTXVLatentUpsampler(
        samples=separate_pass1.video_latent,
        upscale_model=upscale_model.LATENT_UPSCALE_MODEL,
        vae=ckpt.VAE
    )
    
    # --- 7. Pass 2: Refinement Sampling ---
    pass2_inplace = wf.LTXVImgToVideoInplace(
        vae=ckpt.VAE,
        image=preprocessed.output_image,
        latent=upsampled_video.LATENT,
        strength=1.0,
        bypass=False
    )
    
    # Re-merge with original audio latent.
    pass2_concat = wf.LTXVConcatAVLatent(
        video_latent=pass2_inplace.latent,
        audio_latent=separate_pass1.audio_latent
    )

    sampler_p2_select = wf.KSamplerSelect(sampler_name="euler_cfg_pp")
    sigmas_pass2 = wf.ManualSigmas(sigmas="0.85, 0.7250, 0.4219, 0.0")
    noise_pass2 = wf.RandomNoise(noise_seed=42)

    guider_pass2 = wf.CFGGuider(
        model=lora.MODEL,
        positive=pass1_guides.positive,
        negative=pass1_guides.negative,
        cfg=args.cfg
    )

    # Execute refinement pass.
    sampled_pass2 = wf.SamplerCustomAdvanced(
        noise=noise_pass2.NOISE,
        guider=guider_pass2.GUIDER,
        sampler=sampler_p2_select.SAMPLER,
        sigmas=sigmas_pass2.SIGMAS,
        latent_image=pass2_concat.latent
    )

    # --- 8. Final Decoding ---
    separate_final = wf.LTXVSeparateAVLatent(av_latent=sampled_pass2.output)

    # High-resolution memory-efficient decoding.
    decoded_video = wf.VAEDecodeTiled(
        samples=separate_final.video_latent,
        vae=ckpt.VAE,
        tile_size=768,
        overlap=64,
        temporal_size=4096,
        temporal_overlap=4
    )
    
    # Decode final audio stream.
    decoded_audio = wf.LTXVAudioVAEDecode(
        samples=separate_final.audio_latent,
        audio_vae=getattr(audio_vae, "Audio VAE")
    )

    # --- 9. Video ASSEMBLY ---
    video_out = wf.CreateVideo(
        images=decoded_video.IMAGE,
        audio=decoded_audio.Audio,
        fps=args.fps
    )
    
    # Standard terminal node for file generation.
    save_node = wf.SaveVideo(
        video=video_out.VIDEO,
        filename_prefix="video/LTX_2.3_i2v",
        format="auto",
        codec="auto"
    )

    return wf, save_node

def on_progress(node_id, node_type, current, total, is_step):
    """Standard progress callback."""
    if is_step:
        print(f"  - Node {node_id} progress: {current}/{total}")
    else:
        print(f"[{current}/{total}] Executing: {node_type} (ID: {node_id})")

def main(args):
    """Initializes client and runs the animation pipeline."""
    print(f"Connecting to ComfyUI at {args.server}...")
    cli = ComfyClient.create(args.server)
    wf, save_node = build_workflow(cli, args)

    print("Running LTX-2.3 Image-to-Video Workflow...")
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # cli.run() yields (node_id, media) results.
    for node_id, media in cli.run(wf, on_progress=on_progress):
        if node_id != save_node._node.id:
            continue

        print(f"Received video from node {node_id}")
        if args.output:
            path = os.path.join(args.output, f"ltx23_i2v_{uuid.uuid4().hex[:8]}.mp4")
            # media is a moviepy VideoFileClip object (or BytesIO if moviepy not installed)
            if hasattr(media, "write_videofile"):
                media.write_videofile(path, codec="libx264")
            else:
                with open(path, "wb") as f:
                    f.write(media.read())
            print(f"Saved video to {path}")
        else:
            print("Video received, use --output to save it.")

        # Close clip to free resources.
        if hasattr(media, "close"):
            media.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2.3 Image-to-Video Strict Workflow Replication")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--output", help="Directory to save output files")

    # Input image path
    parser.add_argument("--image", default="examples/data/to-upload.png", help="Source image")

    # Model parameters
    parser.add_argument("--ckpt-name", default="LTX2.3\\sulphur2Base_dev.safetensors", help="Checkpoint name")
    parser.add_argument("--gemma-path", default="gemma_3_12B_it_fp4_mixed.safetensors", help="Gemma path")
    parser.add_argument("--lora-name", default="ltx-2.3-22b-distilled-lora-384.safetensors", help="LoRA name")
    parser.add_argument("--upscale-model", default="ltx-2.3-spatial-upscaler-x2-1.1.safetensors", help="Upscaler name")

    # Generation parameters
    parser.add_argument("--prompt", default="Egyptian royal in blue-and-gold headdress...", help="Positive prompt")
    parser.add_argument("--negative-prompt", default="pc game, console game, video game, cartoon, childish, ugly", help="Negative prompt")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--length", type=int, default=126, help="Total frames")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler_cfg_pp")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()
    main(args)
