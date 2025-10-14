from dataclasses import dataclass
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

def run_smolvlm(prompt: str, image: Image.Image, max_new_tokens: int) -> str:
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
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = processor.decode(ids[0], skip_special_tokens=True).split("Assistant:", 1)[-1].strip()
    print(f"SmolVLM: \"{result}\"")

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return result

@dataclass
class CheckImageStatementResult:
    correct: bool
    correct_reasoning: str

def check_image_statement(image: Image.Image, statement: str) -> CheckImageStatementResult:
    reasoning = run_smolvlm(
        prompt=f"Does the following statement correctly describe the image? Answer with 'Yes' or 'No' and explain why.\nStatement: \"{statement}\"",
        image=image,
        max_new_tokens=256
    )
    correct = reasoning.lower().startswith("yes")
    return CheckImageStatementResult(correct=correct, correct_reasoning=reasoning)
