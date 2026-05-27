import os
import argparse
import uuid
from random import randint
from comfyflow import ComfyClient, Workflow
from comfyflow.media import VideoClip

# LTX-2.3 Text-to-Video Workflow Example.
#
# This script demonstrates a high-quality two-pass video generation process:
# 1. Base Pass: Generates a low-resolution video latent using a 8-step distilled sequence.
# 2. Upsampling: Scales the video latent spatially using a dedicated upscale model.
# 3. Refinement Pass: Performs a secondary 3-step sampling on the upscaled latent to add detail.
# 
# Example usage:
# python examples/ltx23_t2v.py --server 127.0.0.1:8188 --prompt "A cinematic flight over mountains" --output ./outputs

def build_workflow(cli, args):
    wf = Workflow(cli)

    # --- 1. Model Loading ---
    # Standard Checkpoint loader for the main LTX-2.3 model.
    ckpt = wf.CheckpointLoaderSimple(ckpt_name=args.ckpt_name)

    # LTXV Audio-Text Encoder Loader (Gemma 3).
    # This node is specific to the LTX-2.3 architecture and handles the multimodal CLIP encoding.
    clip = wf.LTXAVTextEncoderLoader(
        text_encoder=args.gemma_path,
        ckpt_name=args.ckpt_name,
        device="default"
    )
    
    # LoRA Loader for the Distilled LoRA.
    # Distilled LoRAs are essential for high-quality results in very few steps (3-8 steps).
    # Strength 0.5 is used as per the standard distilled workflow.
    lora = wf.LoraLoaderModelOnly(
        model=ckpt.MODEL,
        lora_name=args.lora_name,
        strength_model=0.5
    )
    
    # Loader for the spatial upscaler model used in the second pass.
    upscale_model = wf.LatentUpscaleModelLoader(model_name=args.upscale_model)

    # Dedicated loader for the LTXV Audio VAE.
    audio_vae = wf.LTXVAudioVAELoader(ckpt_name=args.ckpt_name)

    # --- 2. Conditioning ---
    # Standard CLIP Text Encoding for prompt and negative prompt.
    pos_encoded = wf.CLIPTextEncode(text=args.prompt, clip=clip.CLIP)
    neg_encoded = wf.CLIPTextEncode(text=args.negative_prompt, clip=clip.CLIP)

    # LTXV-specific conditioning that incorporates the target frame rate.
    cond = wf.LTXVConditioning(
        frame_rate=args.fps,
        positive=pos_encoded.CONDITIONING,
        negative=neg_encoded.CONDITIONING
    )

    # --- 3. Latent Initialization ---
    # Create the initial empty video latent.
    # For Pass 1, we start at half-resolution (e.g. 640x360 for a 720p target).
    latent_video = wf.EmptyLTXVLatentVideo(
        width=args.width // 2,
        height=args.height // 2,
        length=args.length
    )

    # Create the initial empty audio latent.
    latent_audio = wf.LTXVEmptyLatentAudio(
        frames_number=args.length,
        frame_rate=int(args.fps),
        audio_vae=getattr(audio_vae, "Audio VAE")
    )
    
    # Initialize the video latent context.
    # bypass=True is used in T2V mode as there is no source image to blend.
    pass1_inplace = wf.LTXVImgToVideoInplace(
        vae=ckpt.VAE,
        image=None,
        latent=latent_video.LATENT,
        strength=0.7,
        bypass=True
    )

    # Concatenate video and audio latents for multimodal sampling.
    pass1_concat = wf.LTXVConcatAVLatent(
        video_latent=pass1_inplace.latent,
        audio_latent=latent_audio.Latent
    )

    # --- 4. Pass 1: Base Sampling ---
    # Select the sampler (typically euler_ancestral for LTXV base pass).
    sampler_p1 = wf.KSamplerSelect(sampler_name="euler_ancestral_cfg_pp")
    
    # Define the 8-step sigma sequence for initial generation.
    # This sequence follows the distilled model's optimized schedule.
    sigmas_pass1 = wf.ManualSigmas(sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
    
    noise_pass1 = wf.RandomNoise(noise_seed=args.seed if args.seed is not None else 316435097547460)

    # CFGGuider manages the model guidance at CFG 1.0.
    guider_pass1 = wf.CFGGuider(
        model=lora.MODEL,
        positive=cond.positive,
        negative=cond.negative,
        cfg=1.0
    )

    # Execute the first sampling pass to generate the base video.
    sampled_pass1 = wf.SamplerCustomAdvanced(
        noise=noise_pass1.NOISE,
        guider=guider_pass1.GUIDER,
        sampler=sampler_p1.SAMPLER,
        sigmas=sigmas_pass1.SIGMAS,
        latent_image=pass1_concat.latent
    )

    # --- 5. Upsampling ---
    # Separate the multimodal latent to extract the video component for upscaling.
    separate_pass1 = wf.LTXVSeparateAVLatent(av_latent=sampled_pass1.output)
    
    # Upscale the video latent to the target resolution.
    upsampled_video = wf.LTXVLatentUpsampler(
        samples=separate_pass1.video_latent,
        upscale_model=upscale_model.LATENT_UPSCALE_MODEL,
        vae=ckpt.VAE
    )
    
    # --- 6. Pass 2: Refinement Sampling ---
    # Setup the refinement context using the upsampled latent.
    pass2_inplace = wf.LTXVImgToVideoInplace(
        vae=ckpt.VAE,
        image=None,
        latent=upsampled_video.LATENT,
        strength=1.0,
        bypass=True
    )
    
    # Re-concatenate with the original audio latent component.
    pass2_concat = wf.LTXVConcatAVLatent(
        video_latent=pass2_inplace.latent,
        audio_latent=separate_pass1.audio_latent
    )

    # Select the refinement sampler.
    sampler_p2 = wf.KSamplerSelect(sampler_name="euler_cfg_pp")

    # Define the 3-step sigma sequence for high-res refinement.
    sigmas_pass2 = wf.ManualSigmas(sigmas="0.85, 0.7250, 0.4219, 0.0")
    
    noise_pass2 = wf.RandomNoise(noise_seed=42) # Static seed for pass stability.

    guider_pass2 = wf.CFGGuider(
        model=lora.MODEL,
        positive=cond.positive,
        negative=cond.negative,
        cfg=1.0
    )

    # Execute the refinement pass to add high-frequency detail.
    sampled_pass2 = wf.SamplerCustomAdvanced(
        noise=noise_pass2.NOISE,
        guider=guider_pass2.GUIDER,
        sampler=sampler_p2.SAMPLER,
        sigmas=sigmas_pass2.SIGMAS,
        latent_image=pass2_concat.latent
    )

    # --- 7. Final Decoding ---
    # Final separation for decoding.
    separate_final = wf.LTXVSeparateAVLatent(av_latent=sampled_pass2.output)

    # TILE VAE Decoding for high-resolution video (prevents memory issues).
    decoded_video = wf.VAEDecodeTiled(
        samples=separate_final.video_latent,
        vae=ckpt.VAE,
        tile_size=768,
        overlap=64
    )
    
    # Final audio decoding using the Audio VAE.
    decoded_audio = wf.LTXVAudioVAEDecode(
        samples=separate_final.audio_latent,
        audio_vae=getattr(audio_vae, "Audio VAE")
    )

    # --- 8. Video Assembly ---
    # Combine decoded images and audio into a single video stream.
    video = wf.CreateVideo(
        images=decoded_video.IMAGE,
        audio=decoded_audio.Audio,
        fps=args.fps
    )
    
    # Terminal node to save the video file on the server and emit it to the client.
    save_node = wf.SaveVideo(
        video=video.VIDEO,
        filename_prefix="video/LTX_2.3_t2v",
        format="auto",
        codec="auto"
    )

    return wf, save_node

def on_progress(node_id, node_type, current, total, is_step):
    """Prints standard execution and step-level progress from ComfyUI."""
    if is_step:
        print(f"  - Node {node_id} progress: {current}/{total}")
    else:
        print(f"[{current}/{total}] Executing: {node_type} (ID: {node_id})")

def main(args):
    """Client lifecycle: connects, builds, runs, and saves resulting media."""
    print(f"Connecting to ComfyUI at {args.server}...")
    cli = ComfyClient.create(args.server)
    wf, save_node = build_workflow(cli, args)

    print("Running LTX-2.3 Identical JSON Two-Pass Video Workflow...")
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # run() yields (node_id, media) results.
    for node_id, media in cli.run(wf, on_progress=on_progress):
        if node_id != save_node._node.id:
            continue

        print(f"Received video from node {node_id}")
        if args.output:
            path = os.path.join(args.output, f"ltx23_t2v_{uuid.uuid4().hex[:8]}.mp4")
            # media is a moviepy VideoFileClip object (or BytesIO if moviepy not installed)
            if hasattr(media, "write_videofile"):
                media.write_videofile(path, codec="libx264")
            else:
                with open(path, "wb") as f:
                    f.write(media.read())
            print(f"Saved video to {path}")
        else:
            print("Video received, use --output to save it.")
        
        if hasattr(media, "close"):
            media.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2.3 Text-to-Video Strict JSON Replication")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--output", help="Directory to save output files")

    # Model Selection
    parser.add_argument("--ckpt-name", default="LTX2.3\\sulphur2Base_dev.safetensors", help="Main checkpoint name")
    parser.add_argument("--gemma-path", default="gemma_3_12B_it_fp4_mixed.safetensors", help="Gemma model path")
    parser.add_argument("--lora-name", default="ltx-2.3-22b-distilled-lora-384.safetensors", help="LoRA name")
    parser.add_argument("--upscale-model", default="ltx-2.3-spatial-upscaler-x2-1.1.safetensors", help="Upscale model name")

    # Generation Parameters
    parser.add_argument("--prompt", default="Dynamic cinematic close-up of high-tech modular machinery self-assembling in midair...", help="Positive prompt")
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
