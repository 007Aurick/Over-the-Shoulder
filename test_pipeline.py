import json, re
import cv2
from PIL import Image
from crops import detect, get_question_crops
from vlm import read_crop
from check import check

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
canvas = frame.copy()

for n, q in enumerate(get_question_crops(frame, det), start=1):
    info = parse_json(read_crop(to_pil(q["crop"]), PROMPT_QUESTION))
    result = check(info, [])
    print(f"--- crop {n} ---")
    print("read: ", info)
    print("checked: ", result)

    x1, y1, x2, y2 = q["box"]
    mark_x = max(10, x1-70)
    mark_y = y1 + 55

    if result["correct"]:
        cv2.putText(canvas, "CORRECT", (mark_x, mark_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    else:
        cv2.putText(canvas, "X", (mark_x, mark_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        hint_y = min(y2 + 30, canvas.shape[0] - 15)
        cv2.putText(canvas, result["hint"], (x1, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite("result.png", canvas)
print("result.png saved")