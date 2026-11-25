import base64
import json
import re

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-6.2-astra"

MC_PROMPT = """You are a patient physics/math tutor looking at a photo of ONE multiple choice exam question.
The student circled one option with a pen. Any other handwriting is rough work.

1. Read the question and options. Solve it yourself, privately.
2. Find which option the student circled.
3. Grade it.

Return JSON only:
{"number": "question number as printed", "readable": true, "circled": "A", "correct": true, "hint": null}

Rules:
- If wrong, "hint" is at most 20 words and points at the most fundamental mistake first.
- NEVER state or hint at which letter is correct, or the final numeric answer.
- Each hint must be more specific than the previous hints listed below.
- If correct, "hint" is null.
- If you can't read the question or can't tell what was circled, set "readable" to false.
- Plain ASCII only, no special symbols."""


SA_PROMPT = """You are a patient physics/math tutor looking at a photo of a short answer exam question
with the student's handwritten work.

1. Read the question and each part (A, B, ...). Solve them yourself, privately.
2. Read the student's handwritten work for each part.
3. Grade each FINISHED part. Leave out parts with no work.
   The student may still be writing. A part counts as finished if the student has started a
   later part, OR it has a clear final answer (a value with units, boxed or underlined).
   If the last part with work looks unfinished, leave it out - don't grade work in progress.

Return JSON only:
{"number": "question number as printed", "readable": true,
 "parts": {"A": {"correct": true, "hint": null, "x": 0.85, "y": 0.40},
           "B": {"correct": false, "hint": "...", "x": 0.80, "y": 0.90}}}

"x" and "y" say where that part's FINAL ANSWER is in the image, as fractions from 0 to 1
(x: 0 = left edge, 1 = right edge; y: 0 = top, 1 = bottom). Point at the right end of the
line with the student's final answer for that part, so a mark can be drawn next to it.

Rules:
- For a wrong part, "hint" is at most 20 words and points at the most fundamental mistake first.
- NEVER state the final answer or the correct number.
- Each hint must be more specific than the previous hints listed below.
- Ignore faint, erased or ghosted pencil marks. Only grade dark, current writing.
- If the work is too messy to read, set "readable" to false.
- Plain ASCII only, no special symbols."""


INK_DARKNESS = 45   # how much darker than the paper a mark must be to count (raise to drop more ghosts)


def clean_ink(img_bgr):
    """White out faint/erased pencil: keep only marks clearly darker than the paper around them."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = max(31, (min(gray.shape[:2]) // 8) | 1)          # big odd blur = local paper brightness
    paper = cv2.GaussianBlur(gray, (k, k), 0)
    darker = cv2.subtract(paper, gray)                   # how much darker each pixel is than the paper
    out = gray.copy()
    out[darker < INK_DARKNESS] = 255
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _to_data_url(img_bgr, max_w=1024):
    h, w = img_bgr.shape[:2]
    if w > max_w:
        img_bgr = cv2.resize(img_bgr, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def check_mc_image(img_bgr, previous_hints):
    return _ask(MC_PROMPT, img_bgr, previous_hints, max_w=1024)


FINAL_NOTE = """
The student says they are DONE: grade every part that has work, including the last one."""


def check_sa_image(img_bgr, previous_hints, final=False):
    # handwriting needs a bit more resolution than printed options
    prompt = SA_PROMPT + (FINAL_NOTE if final else "")
    cleaned = clean_ink(img_bgr)
    cv2.imwrite("sa_sent.jpg", cleaned)     # open this to see which handwriting GPT will grade
    prompt += """
You get TWO images of the same page:
- Image 1 is the original photo. Use it to read the printed question text.
- Image 2 has faint and erased pencil removed. Only grade handwriting that is visible in Image 2.
  Handwriting that appears only in Image 1 was erased: ignore it completely."""
    return _ask(prompt, [img_bgr, cleaned], previous_hints, max_w=1600)


def _ask(prompt, img_bgr, previous_hints, max_w):
    prev = "\n".join(f"- {h}" for h in previous_hints) or "(none)"
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Previous hints:\n{prev}"},
                *[{"type": "image_url", "image_url": {"url": _to_data_url(im, max_w)}}
                  for im in (img_bgr if isinstance(img_bgr, list) else [img_bgr])],
            ]},
        ],
    )
    text = resp.choices[0].message.content
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {"readable": False}


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "crop_Q1.jpg")
    print(check_mc_image(img, []))