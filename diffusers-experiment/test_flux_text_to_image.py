from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
import torch
import gc

def run_flux(model: str, image_file: str) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.cuda.empty_cache()
    gc.collect()

    pipe = FluxPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )

    pipe.enable_sequential_cpu_offload()

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift=1.0
    )

    pipe.enable_vae_tiling()
    pipe.vae.enable_slicing()

    torch.cuda.empty_cache()
    gc.collect()

    prompt = "a large fat tabby cat sitting on a wooden table. realistic photo. natural lighting. 50mm"

    image = pipe(
        prompt,
        width=1024,
        height=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=torch.Generator("cuda").manual_seed(328878620293472),
        max_sequence_length=512,
    ).images[0]

    image.save(image_file)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

def test_flux_dev():
    run_flux("black-forest-labs/FLUX.1-dev", "flux1dev-cat.png")
