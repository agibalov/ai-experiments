from comfy_script.runtime import load, nodes, ImageBatchResult

load()

def make_sd_t2i_workflow(prompt: str, negative_prompt: str) -> nodes.SaveImage:
    model, clip, vae = nodes.CheckpointLoaderSimple('v1-5-pruned-emaonly-fp16.safetensors')
    conditioning = nodes.CLIPTextEncode(prompt, clip)
    conditioning2 = nodes.CLIPTextEncode(negative_prompt, clip)
    latent = nodes.EmptyLatentImage(512, 512, 1)
    latent = nodes.KSampler(model, 1092480794965720, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
    image = nodes.VAEDecode(latent, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def make_sdxl_t2i_upscale_workflow(prompt: str, negative_prompt: str) -> nodes.SaveImage:
    model, clip, vae = nodes.CheckpointLoaderSimple('RealVisXL_V4.0.safetensors')
    conditioning = nodes.CLIPTextEncode(prompt, clip)
    conditioning2 = nodes.CLIPTextEncode(negative_prompt, clip)
    latent = nodes.EmptyLatentImage(512, 512, 1)
    latent = nodes.KSampler(model, 873141143997374, 12, 8, 'dpmpp_sde', 'normal', conditioning, conditioning2, latent, 1)
    latent2 = nodes.LatentUpscale(latent, 'nearest-exact', 768, 768, 'disabled')
    latent2 = nodes.KSampler(model, 32484370206415, 14, 8, 'dpmpp_2m', 'simple', conditioning, conditioning2, latent2, 0.5)
    image = nodes.VAEDecode(latent2, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def make_flux_t2i_workflow(prompt: str) -> nodes.SaveImage:
    model, clip, vae = nodes.CheckpointLoaderSimple('flux1-dev-fp8.safetensors')
    clip_text_encode_positive_prompt_conditioning = nodes.CLIPTextEncode(prompt, clip)
    clip_text_encode_positive_prompt_conditioning = nodes.FluxGuidance(clip_text_encode_positive_prompt_conditioning, None)
    clip_text_encode_negative_prompt_conditioning = nodes.CLIPTextEncode('', clip)
    latent = nodes.EmptySD3LatentImage(1024, 1024, 1)
    latent = nodes.KSampler(model, 972054013131368, 20, 1, 'euler', 'simple', clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, latent, 1)
    image = nodes.VAEDecode(latent, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def make_qwen_t2i_workflow(prompt: str) -> nodes.SaveImage:
    model = nodes.UNETLoader('qwen_image_fp8_e4m3fn.safetensors', 'default')
    model = nodes.ModelSamplingAuraFlow(model, 3.1000000000000005)
    clip = nodes.CLIPLoader('qwen_2.5_vl_7b_fp8_scaled.safetensors', 'qwen_image', 'default')
    clip_text_encode_positive_prompt_conditioning = nodes.CLIPTextEncode(prompt, clip)
    clip_text_encode_negative_prompt_conditioning = nodes.CLIPTextEncode('', clip)
    latent = nodes.EmptySD3LatentImage(1328, 1328, 1)
    latent = nodes.KSampler(model, 1125488487853217, 20, 2.5, 'euler', 'simple', clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, latent, 1)
    vae = nodes.VAELoader('qwen_image_vae.safetensors')
    image = nodes.VAEDecode(latent, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def save_images(result: ImageBatchResult, prefix: str):
    for i, image in enumerate(result.wait()):
        image.save(f"{prefix}{i+1:03d}.png")

# if True:
#     save = make_sd_t2i_workflow(
#         prompt="Huge fat tabby cat sitting on a wooden table", 
#         negative_prompt="text, watermark")
#     result = save.wait()
#     save_images(result=result, prefix="sd_cat_")

# if True:
#     save = make_sdxl_t2i_upscale_workflow(
#         prompt="Huge fat tabby cat sitting on a wooden table", 
#         negative_prompt="text, watermark")
#     result = save.wait()
#     save_images(result=result, prefix="sdxl_cat_")

# if True:
#     save = make_flux_t2i_workflow(prompt="Huge fat tabby cat sitting on a wooden table")
#     result = save.wait()
#     save_images(result=result, prefix="flux_cat_")

# if True:
#     save = make_qwen_t2i_workflow(prompt="Huge fat tabby cat sitting on a wooden table")
#     result = save.wait()
#     save_images(result=result, prefix="qwen_cat_")
