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

PROMPT_SHORT = """You are an OCR system, not a tutor. Copy exactly what is on the page.

The image shows ONE printed exam problem with parts (A, B, C) and a student's handwritten work under each part.

- Copy the PRINTED problem and each part's question exactly.
- Copy the HANDWRITING under each part line by line, exactly as written, even if it is wrong.
- NEVER solve, correct or complete the handwriting. Never add numbers that are not written.
- If a handwritten character is unreadable, write "?".

Return ONLY this JSON:
{"number": <problem number>,
 "type": "short",
 "question": "<printed problem text>",
 "parts": {"A": {"question": "...", "work": ["line 1", "line 2"]},
           "B": {"question": "...", "work": ["line 1", "line 2"]}}}"""

def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group()) if m else None

frame = cv2.imread("screenshot_sa.png")
det = detect(frame)
canvas = frame.copy()
crops = get_question_crops(frame, det)
if not any(q["is_mc"] for q in crops):          # no circles → short-answer page
    crops = [{"crop": frame, "box": (0, 0, frame.shape[1], frame.shape[0]), "is_mc": False}]

for n, q in enumerate(crops, start=1):                          # ← use crops
    prompt = PROMPT_QUESTION if q["is_mc"] else PROMPT_SHORT     # ← pick the prompt
    info = parse_json(read_crop(to_pil(q["crop"]), prompt))
    if info is None:
        print(f"Q{n}: couldn't parse Qwen's output")
        continue
    result = check(info, [])
    print(f"--- crop {n} ---")
    print("read: ", info)
    print("checked: ", result)

    if not result.get("readable", True):
        print(f"Q{n}: couldn't read it clearly, try again")
        continue                                                  # ← skip instead of crashing

    x1, y1, x2, y2 = q["box"]
    mark_x = max(10, x1)
    mark_y = max(40, y1 + 10)
    h, w = canvas.shape[:2]

    if not q["is_mc"] and result.get("parts"):
        # short answer: one line in a bottom bar, e.g. "A: OK   B: X   |   B: hint"
        parts = {p: r for p, r in result["parts"].items()
                 if info.get("parts", {}).get(p, {}).get("work")}      # only parts with work
        status = "   ".join(f"{p}: {'OK' if r['correct'] else 'X'}" for p, r in parts.items())
        wrong = [(p, r) for p, r in parts.items() if not r["correct"]]
        line = status + (f"   |   {wrong[0][0]}: {wrong[0][1]['hint']}" if wrong else "")
        line = line.encode("ascii", "ignore").decode()
        cv2.rectangle(canvas, (0, h - 50), (w, h), (40, 40, 40), -1)
        cv2.putText(canvas, line, (15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif result["correct"]:
        cv2.putText(canvas, "OK", (mark_x, mark_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

    else:
        cv2.putText(canvas, "X", (mark_x, mark_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        hint = (result.get("hint") or "Check your answer").encode("ascii", "ignore").decode()
        hint_y = min(y2, h - 45)
        cv2.rectangle(canvas, (mark_x, hint_y), (mark_x + 900, hint_y + 40), (0, 0, 180), -1)
        cv2.putText(canvas, hint, (mark_x + 10, hint_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imwrite("result_sa.png", canvas)
print("result_sa.png saved")