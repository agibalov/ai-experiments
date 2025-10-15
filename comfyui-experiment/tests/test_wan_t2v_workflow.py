from comfy_script.runtime import load

from tests.tools import comfyui_free_memory, parameterize_workflow, run_video_workflow


def test_it_works():
    load()
    
    workflow_prompt = parameterize_workflow(
        "video_wan2_2_14B_t2v.json.jinja",
        prompt="Huge fat tabby cat sitting on a wooden table",
        negative_prompt="text, watermark",
        width=640,
        height=640,
        frames=81,
        fps=16
    )
    
    run_video_workflow(
        workflow_prompt, 
        "cat_wan_t2i.mp4"
    )    
    
    comfyui_free_memory()
