# comfyui-experiment

Figuring out ComfyUI automation.

## Prerequisites

### ComfyUI

Install ComfyUI:

```
cd ~
mkdir ComfyUI-Workspace
cd ComfyUI-Workspace
uv venv --seed --python 3.12
uv pip install comfy-cli
uv run comfy --here --skip-prompt install --nvidia 
uv run comfy tracking disable
```

Download all relevant models manually.

### ComfyScript custom node

Install ComfyScript custom node

```
cd ~/ComfyUI-Workspace
git clone https://github.com/Chaoses-Ib/ComfyScript.git ./ComfyUI/custom_nodes/ComfyScript
uv pip install -e "./ComfyUI/custom_nodes/ComfyScript[default]"
```

### ComfyUI API Tools

Install ComfyUI API Tools

```
cd ~/ComfyUI-Workspace
git clone https://github.com/brantje/ComfyUI-api-tools ./ComfyUI/custom_nodes/ComfyUI-api-tools
```

## How to do things

* `cd ~/ComfyUI-Workspace && uv run comfy --here launch` to launch ComfyUI.
* `uv run pytest` to run the app.

## Notes

* Image-producing workflow scripts get printed by ComfyScript when it's installed to ComfyUI custom nodes. Just open the example in the UI and run it once, the script should appear in the console (e.g. `test_sd_t2i_workflow.py`). This, however, doesn't work for video-producing workflows - ComfyUI doesn't seem to support them as of October 2025.
* For video-producing workflows, save the workflow as JSON from the UI, replace hardcoded parameters with Jinja placeholders (e.g. `test_wan_t2v_workflow.py`), and use ComfyUI's HTTP API directly.
