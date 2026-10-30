from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch


model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype='auto', device_map='auto')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
img = Image.open('plate.jpg')

messages = [{'role': 'user', 'content': [
    {'type': 'image', 'image': img},
    {'type': 'text',  'text': 'What is the license plate number?'},
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

images, video = process_vision_info(messages)

inputs = processor(text=[text], images=images, videos=video, return_tensors='pt', padding=True).to(model.device)

output_id = model.generate(**inputs, max_new_tokens=256, do_sample=False)


new_ids = output_id[0, inputs['input_ids'].shape[1]:]
answer = processor.decode(new_ids, skip_special_tokens=True)
print(answer)