from diffusers import AutoPipelineForText2Image
from diffusers import EulerAncestralDiscreteScheduler
import torch
import gc


def run(model: str, image_file: str) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = AutoPipelineForText2Image.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )

    pipe = pipe.to("cuda")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    # Clear cache before generation
    torch.cuda.empty_cache()
    gc.collect()

    prompt = "a large fat tabby cat sitting on a wooden table. realistic photo. natural lighting. 50mm"
    negative_prompt = "text, watermarklow quality, medium quality, hdr, high contrast, distorted, deformity, cgi"

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=512,
        num_inference_steps=12,
        guidance_scale=8.0,
        generator=torch.Generator("cuda").manual_seed(328878620293472),
    ).images[0]

    image.save(image_file)


def test_realvisxlv40():
    run("SG161222/RealVisXL_V4.0", "realvisxlv40-cat.png")


def test_juggernautxlv9():
    run("RunDiffusion/Juggernaut-XL-v9", "juggernautxlv9-cat.png")
