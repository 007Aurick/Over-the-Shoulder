from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")


MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'

#loads the model once, downloads the weights, compresses them to 4-bit so they fit in 6 GB of GPU
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map='auto')

#model only understands numbers, converts your text and images into numbers 
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=1024*28*28)

#loads the image
img = Image.open('plate.jpg').convert('RGB')

#prompt the chatbot
messages = [{'role': 'user', 'content': [
    {'type': 'image', 'image': img},
    {'type': 'text',  'text': 'What is the license plate number?'},
]}]

#converts the prompt into the exact text Qwen can understand
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#pulls the image back out of the message and resizes it the way Qwen expects it
images, video = process_vision_info(messages)

#text becomes token IDs, the image becomes pixel tensors. Now the model can process it
inputs = processor(text=[text], images=images, videos=video, return_tensors='pt', padding=True).to(model.device)


output_id = model.generate(**inputs, max_new_tokens=256, do_sample=False)


new_ids = output_id[0, inputs['input_ids'].shape[1]:]
answer = processor.decode(new_ids, skip_special_tokens=True)
print(answer)