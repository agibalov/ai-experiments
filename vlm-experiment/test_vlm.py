import gc
import pytest
from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, logging
from PIL import Image
import torch

from semantics import match_semantics

logging.set_verbosity_error()
logging.disable_progress_bar()

@pytest.fixture(autouse=True)
def run_before_each():
    from ollama import Client
    c = Client(host="http://localhost:11434")
    for m in c.ps()["models"]:
        c.generate(model=m["name"], prompt="", keep_alive=0)

def run_smolvlm(prompt: str, image: str, max_new_tokens: int) -> str:
    print(f"Prompt: \"{prompt}\", Image: \"{image}\"")

    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)

    prompt = processor.apply_chat_template([{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    }], add_generation_prompt=True)
    inputs = processor(text=prompt, images=[Image.open(image)], return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = processor.decode(ids[0], skip_special_tokens=True).split("Assistant:", 1)[-1].strip()
    print(f"SmolVLM: \"{result}\"")

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return result

def run_minicpm(prompt: str, image: str, max_new_tokens: int) -> str:
    print(f"Prompt: \"{prompt}\", Image: \"{image}\"")

    model_id = "openbmb/MiniCPM-V-2_6-int4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device).eval()

    with torch.inference_mode():
        result = model.chat(image=None, msgs=[{
            "role": "user",
            "content": [Image.open(image), prompt]
        }], tokenizer=tokenizer, max_new_tokens=max_new_tokens, temperature=0.1)
    print(f"MiniCPM: \"{result}\"")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result

@pytest.fixture(params=[run_smolvlm, run_minicpm])
def vlm(request):
    return request.param

def test_stoic_subject(vlm):
    result = vlm("What is the subject of the image?", "stoic.jpg", max_new_tokens=100)
    assert match_semantics(result, "a bicycle").match is True

def test_stoic_json_true(vlm):
    result = vlm("Is there a bicycle in the image? Respond with a JSON object "
                 "{\"answer\": true} or {\"answer\": false}. No explanations.", "stoic.jpg", max_new_tokens=100)
    assert result == "{\"answer\": true}"

def test_stoic_json_false(vlm):
    result = vlm("Is there a unicorn in the image? Respond with a JSON object "
                 "{\"answer\": true} or {\"answer\": false}. No explanations.", "stoic.jpg", max_new_tokens=100)
    assert result == "{\"answer\": false}"

def test_stoic_bicycle_color(vlm):
    result = vlm("What is the color of the bicycle in the image?", "stoic.jpg", max_new_tokens=100)
    assert match_semantics(result, "green").match is True

def test_stoic_behind(vlm):
    result = vlm("What's behind the bicycle?", "stoic.jpg", max_new_tokens=100)
    assert match_semantics(result, "a tree").match is True

def test_stoic_location(vlm):
    result = vlm("Where has this picture been taken?", "stoic.jpg", max_new_tokens=100)
    assert match_semantics(result, "woods").match is True

def test_stoic_season(vlm):
    result = vlm("What season is depicted in the image?", "stoic.jpg", max_new_tokens=100)
    assert match_semantics(result, "late spring or early summer").match is True

def test_stoic_fooling_around(vlm):
    result = vlm("Is bicycle on the photo pleasant to ride?", "stoic.jpg", max_new_tokens=100)
    assert match_semantics(result, "yes").match is True
