from diffusers import AutoPipelineForText2Image
from diffusers import EulerAncestralDiscreteScheduler
import torch
import gc

def test_realvisxl_v40():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    pipe = pipe.to("cuda")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # Memory optimizations to avoid OOM
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    # Enable CPU offloading to manage memory better
    pipe.enable_model_cpu_offload()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Monitor GPU memory before generation
    def print_gpu_memory():
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Free: {memory_free:.2f}GB")

    print("After loading model to GPU:")
    print_gpu_memory()

    # Clear cache before generation
    torch.cuda.empty_cache()
    gc.collect()

    print("Before generation:")
    print_gpu_memory()

    prompt = "a large fat tabby cat sitting on a wooden table. realistic photo. natural lighting. 50mm"
    negative_prompt = "text, watermarklow quality, medium quality, hdr, high contrast, distorted, deformity, cgi"

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=1024, height=512,
        num_inference_steps=12,
        guidance_scale=8.0,
        generator=torch.Generator("cuda").manual_seed(328878620293472),
    ).images[0]

    print("After generation:")
    print_gpu_memory()

    image.save("cat.png")
