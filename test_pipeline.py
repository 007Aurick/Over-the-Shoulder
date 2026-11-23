import json, re
import cv2
from PIL import Image
from crops import detect, get_question_crops
from vlm import read_crop

PROMPT_QUESTION = """You are an OCR system. Copy what is printed on the page.

The image shows ONE printed exam question. A student may have drawn a circle around one option letter.

Return ONLY this JSON, nothing else:
{"number": <printed question number as an integer, or null>,
 "type": "mc" if there are printed options A-E, otherwise "short",
 "question": "<printed question text, copied exactly>",
 "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."} or null,
 "circled": "A", "B", "C", "D" or "E" if one option letter has a hand-drawn circle or mark around it, otherwise null}

Ignore all other handwriting. Do not solve the question."""

def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group()) if m else None

frame = cv2.imread("screenshot.png")
det = detect(frame)

for n, q in enumerate(get_question_crops(frame, det), start=1):
    info = parse_json(read_crop(to_pil(q["crop"]), PROMPT_QUESTION))
    print(f"--- crop {n} ---")
    print(info)