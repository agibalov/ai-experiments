from comfy_script.runtime import load, nodes

from .tools import save_images


def make_flux_t2i_workflow(prompt: str):
    model, clip, vae = nodes.CheckpointLoaderSimple('flux1-dev-fp8.safetensors')
    clip_text_encode_positive_prompt_conditioning = nodes.CLIPTextEncode(prompt, clip)
    clip_text_encode_positive_prompt_conditioning = nodes.FluxGuidance(clip_text_encode_positive_prompt_conditioning, None)
    clip_text_encode_negative_prompt_conditioning = nodes.CLIPTextEncode('', clip)
    latent = nodes.EmptySD3LatentImage(1024, 1024, 1)
    latent = nodes.KSampler(model, 972054013131368, 20, 1, 'euler', 'simple', clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, latent, 1)
    image = nodes.VAEDecode(latent, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def test_it_works():
    load()
    save = make_flux_t2i_workflow(prompt="Huge fat tabby cat sitting on a wooden table")
    result = save.wait()
    save_images(result=result, prefix="cat_flux_")
