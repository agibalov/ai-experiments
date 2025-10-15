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
