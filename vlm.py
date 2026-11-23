from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Load once when this file is imported (slow ~20s), then reuse for every crop
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=1024*28*28)


def read_crop(img: Image.Image, prompt: str) -> str:
    """Send one image + prompt to Qwen, return its text answer."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, video = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=video,
                       return_tensors="pt", padding=True).to(model.device)

    output_id = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    new_ids = output_id[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True)


# Only runs when you do `py vlm.py`, NOT when another file imports read_crop
if __name__ == "__main__":
    img = Image.open("screenshot.png").convert("RGB")
    print(read_crop(img, "Describe this image."))