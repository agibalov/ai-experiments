from comfy_script.runtime import ImageBatchResult


def save_images(result: ImageBatchResult, prefix: str):
    for i, image in enumerate(result.wait()):
        image.save(f"{prefix}{i+1:03d}.png")
