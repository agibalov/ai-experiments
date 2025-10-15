from comfy_script.runtime import load
from PIL import Image

from tests.comfyui_client import comfyui_free_memory, parameterize_workflow, run_video_workflow, upload_image_to_comfyui, delete_comfyui_image
from tests.test_sd_t2i_workflow import make_sd_t2i_workflow


def test_it_works():
    load()
    
    save = make_sd_t2i_workflow(
        prompt="Huge fat tabby cat sitting on a wooden table", 
        negative_prompt="text, watermark")
    image = save.wait().wait()[0]
    
    resized_img = image.resize((320, 320), Image.Resampling.LANCZOS)
    uploaded_image_name = upload_image_to_comfyui(resized_img)
    
    workflow_prompt = parameterize_workflow(
        "video_wan2_2_14B_i2v.json.jinja",
        image=uploaded_image_name,
        prompt="Huge fat tabby cat sitting on a wooden table",
        negative_prompt="text, watermark",
        width=320,
        height=320,
        frames=101,
        fps=20
    )

    run_video_workflow(
        workflow_prompt, 
        "cat_wan_i2v.mp4"
    )    
    
    delete_comfyui_image(uploaded_image_name)
    comfyui_free_memory()
