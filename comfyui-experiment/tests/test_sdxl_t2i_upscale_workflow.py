from comfy_script.runtime import load, nodes

from .tools import save_images


def make_sdxl_t2i_upscale_workflow(prompt: str, negative_prompt: str):
    model, clip, vae = nodes.CheckpointLoaderSimple('RealVisXL_V4.0.safetensors')
    conditioning = nodes.CLIPTextEncode(prompt, clip)
    conditioning2 = nodes.CLIPTextEncode(negative_prompt, clip)
    latent = nodes.EmptyLatentImage(512, 512, 1)
    latent = nodes.KSampler(model, 873141143997374, 12, 8, 'dpmpp_sde', 'normal', conditioning, conditioning2, latent, 1)
    latent2 = nodes.LatentUpscale(latent, 'nearest-exact', 768, 768, 'disabled')
    latent2 = nodes.KSampler(model, 32484370206415, 14, 8, 'dpmpp_2m', 'simple', conditioning, conditioning2, latent2, 0.5)
    image = nodes.VAEDecode(latent2, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def test_it_works():
    load()
    save = make_sdxl_t2i_upscale_workflow(
        prompt="Huge fat tabby cat sitting on a wooden table", 
        negative_prompt="text, watermark")
    result = save.wait()
    save_images(result=result, prefix="cat_sdxl_")
