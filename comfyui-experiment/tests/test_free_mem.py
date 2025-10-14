from comfy_script.runtime import load, nodes
import requests
import time

from .tools import save_images


def make_sd_t2i_workflow(prompt: str, negative_prompt: str):
    model, clip, vae = nodes.CheckpointLoaderSimple('v1-5-pruned-emaonly-fp16.safetensors')
    conditioning = nodes.CLIPTextEncode(prompt, clip)
    conditioning2 = nodes.CLIPTextEncode(negative_prompt, clip)
    latent = nodes.EmptyLatentImage(512, 512, 1)
    latent = nodes.KSampler(model, 1092480794965720, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
    image = nodes.VAEDecode(latent, vae)
    return nodes.SaveImage(image, 'ComfyUI')

def comfyui_free_memory():
    initial_vram = comfyui_get_free_vram()
    
    requests.post('http://127.0.0.1:8188/api/free', 
                  json={"unload_models": True, "free_memory": True})
    
    start_time = time.time()
    best_vram = initial_vram
    stable_readings = 0
    required_stable_readings = 3
    
    while time.time() - start_time < 10:
        current_vram = comfyui_get_free_vram()
        
        if current_vram > best_vram:
            best_vram = current_vram
            stable_readings = 0
        elif abs(current_vram - best_vram) < 1024 * 1024:
            stable_readings += 1
        else:
            stable_readings = 0
        
        if stable_readings >= required_stable_readings:
            break
            
        time.sleep(0.5)

def comfyui_get_free_vram():
    try:
        response = requests.get('http://127.0.0.1:8188/system_stats')
        response.raise_for_status()
        stats = response.json()
        
        if 'devices' in stats and len(stats['devices']) > 0:
            device = stats['devices'][0]
            vram_free = device.get('vram_free', 0)
            return vram_free
        else:
            return 0
            
    except Exception as e:
        return 0

def test_it_works():
    load()
    
    initial_vram = comfyui_get_free_vram()
    
    save = make_sd_t2i_workflow(
        prompt="Huge fat tabby cat sitting on a wooden table", 
        negative_prompt="text, watermark")
    result = save.wait()
    save_images(result=result, prefix="cat_sd_")
        
    comfyui_free_memory()
    
    final_vram = comfyui_get_free_vram()

    assert abs(initial_vram - final_vram) < 10 * 1024 * 1024
