import time
from comfy_script.runtime import ImageBatchResult
import requests


def save_images(result: ImageBatchResult, prefix: str):
    for i, image in enumerate(result.wait()):
        image.save(f"{prefix}{i+1:03d}.png")

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
