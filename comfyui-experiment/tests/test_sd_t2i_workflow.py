from comfy_script.runtime import load, nodes

from tests.judge import check_image_statement
from tests.comfyui_client import comfyui_free_memory


def make_sd_t2i_workflow(prompt: str, negative_prompt: str):
    model, clip, vae = nodes.CheckpointLoaderSimple('v1-5-pruned-emaonly-fp16.safetensors')
    conditioning = nodes.CLIPTextEncode(prompt, clip)
    conditioning2 = nodes.CLIPTextEncode(negative_prompt, clip)
    latent = nodes.EmptyLatentImage(512, 512, 1)
    latent = nodes.KSampler(model, 1092480794965720, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
    image = nodes.VAEDecode(latent, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def test_it_works():
    load()
    save = make_sd_t2i_workflow(
        prompt="Huge fat tabby cat sitting on a wooden table", 
        negative_prompt="text, watermark")
    image = save.wait().wait()[0]
    image.save("cat_sd.png")

    comfyui_free_memory()

    result = check_image_statement(image, "There is a huge fat tabby cat sitting on a wooden table.")
    assert result.correct, result.correct_reasoning
