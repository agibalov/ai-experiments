import time
import requests
import json
import os


def download_comfyui_output(filename, subfolder="", output_dir=".", local_filename=None, base_url="http://127.0.0.1:8188"):
    if subfolder:
        url = f"{base_url}/view?filename={filename}&subfolder={subfolder}"
    else:
        url = f"{base_url}/view?filename={filename}"
    
    response = requests.get(url)
    response.raise_for_status()
    
    if local_filename is None:
        local_filename = filename
    local_path = os.path.join(output_dir, local_filename)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    
    return local_path

def parameterize_workflow(template_file_name: str, **params):
    from jinja2 import Template

    with open(template_file_name, 'r') as f:
        template_content = f.read()
    
    template = Template(template_content)
    rendered_content = template.render(**params)
    
    workflow = json.loads(rendered_content)
    return workflow

def run_video_workflow(workflow_prompt, target_filename, max_wait=600, base_url="http://127.0.0.1:8188"):
    import time
    
    response = requests.post(f"{base_url}/prompt", 
                             json={"prompt": workflow_prompt})
    response.raise_for_status()
    prompt_data = response.json()
    prompt_id = prompt_data['prompt_id']
    
    start_time = time.time()
    video_found = False
    
    while time.time() - start_time < max_wait:
        history_response = requests.get(f"{base_url}/history/{prompt_id}")
        if history_response.status_code == 200:
            history_data = history_response.json()
            if prompt_id in history_data:
                history_item = history_data[prompt_id]
                
                for node_id, node_output in history_item.get('outputs', {}).items():
                    if 'videos' in node_output:
                        for video_info in node_output['videos']:
                            filename = video_info.get('filename')
                            subfolder = video_info.get('subfolder', '')
                            if filename:
                                download_comfyui_output(filename, subfolder, ".", target_filename, base_url)
                                video_found = True
                                break
                    elif 'images' in node_output:
                        for image_info in node_output['images']:
                            filename = image_info.get('filename')
                            if filename and filename.endswith('.mp4'):
                                subfolder = image_info.get('subfolder', '')
                                download_comfyui_output(filename, subfolder, ".", target_filename, base_url)
                                video_found = True
                                break
                    
                    if video_found:
                        break
                
                if video_found:
                    break
        
        time.sleep(2)
    else:
        raise TimeoutError(f"Workflow {prompt_id} timed out after {max_wait} seconds")
    
    if not video_found:
        raise RuntimeError(f"Workflow {prompt_id} completed but no video file was produced")


def comfyui_free_memory(base_url="http://127.0.0.1:8188"):
    initial_vram = comfyui_get_free_vram(base_url)
    
    requests.post(f'{base_url}/api/free', 
                  json={"unload_models": True, "free_memory": True})
    
    start_time = time.time()
    best_vram = initial_vram
    stable_readings = 0
    required_stable_readings = 3
    
    while time.time() - start_time < 10:
        current_vram = comfyui_get_free_vram(base_url)
        
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

def comfyui_get_free_vram(base_url="http://127.0.0.1:8188"):
    try:
        response = requests.get(f'{base_url}/system_stats')
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
