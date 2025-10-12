from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)
import torch
from PIL import Image


def run(model: str, image_file_prefix: str) -> None:
    prompt = "a large fat tabby cat sitting on a wooden table. realistic photo. natural lighting. 50mm"
    negative = (
        "text, watermark, low quality, hdr, high contrast, distorted, deformity, cgi"
    )
    base_w = 1024
    base_h = 512
    up_w = 1536
    up_h = 768

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Pass 1: low-res text2img ----
    t2i_pipe = AutoPipelineForText2Image.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    t2i_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        t2i_pipe.scheduler.config, use_karras_sigmas=True
    )
    t2i_pipe.enable_vae_tiling()
    t2i_pipe.enable_vae_slicing()
    t2i_pipe.enable_attention_slicing()
    t2i_pipe.enable_model_cpu_offload()
    gen = torch.Generator(device="cuda").manual_seed(328878620293472)

    img_lo = t2i_pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=base_w,
        height=base_h,
        strength=1.0,
        num_inference_steps=12,
        guidance_scale=8.0,
        generator=gen,
    ).images[0]

    # ---- Upscale to target (pixel space) ----
    img_up = img_lo.resize((up_w, up_h), Image.BICUBIC)

    # ---- Pass 2: light img2img to restore detail (“hires fix”) ----
    i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    i2i_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        i2i_pipe.scheduler.config, use_karras_sigmas=True
    )
    i2i_pipe.enable_vae_tiling()
    i2i_pipe.enable_vae_slicing()
    i2i_pipe.enable_attention_slicing()
    i2i_pipe.enable_model_cpu_offload()

    img_hi = i2i_pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=img_up,
        strength=0.5,
        num_inference_steps=14,
        guidance_scale=8.0,
        generator=gen,
    ).images[0]

    img_lo.save(f"{image_file_prefix}-lowres.png")
    img_up.save(f"{image_file_prefix}-upscaled.png")
    img_hi.save(f"{image_file_prefix}-hiresfix.png")


def test_hiresfix_realvisxlv40():
    run("SG161222/RealVisXL_V4.0", "hiresfix-realvisxlv40-cat")


def test_hiresfix_juggernautxlv9():
    run("RunDiffusion/Juggernaut-XL-v9", "hiresfix-juggernautxlv9-cat")
