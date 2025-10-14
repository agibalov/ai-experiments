from comfy_script.runtime import load, nodes

from tests.judge import check_image_statement
from tests.tools import comfyui_free_memory


def make_qwen_t2i_workflow(prompt: str):
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

def test_it_works():
    load()
    save = make_qwen_t2i_workflow(prompt="Huge fat tabby cat sitting on a wooden table")
    image = save.wait().wait()[0]
    image.save("cat_sd.png")

    comfyui_free_memory()

    result = check_image_statement(image, "There is a huge fat tabby cat sitting on a wooden table.")
    assert result.correct
