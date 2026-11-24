"""
Over the Shoulder - live app.

Webcam -> RF-DETR (detect thread) -> crops -> Qwen (read) -> GPT (grade + hint) -> screen

Keys:
  Space  grade the short answer on the page
  F      toggle fullscreen (for the projector)
  B      toggle detection boxes
  R      reset all marks and hints
  Q      quit
"""

import itertools
import json
import re
import textwrap
import threading
import time
from queue import Queue, Empty

import cv2
import supervision as sv
from PIL import Image

from crops import detect, get_question_crops
from vlm import read_crop
from check import check

# ---------------- settings ----------------
CAMERA = 0
STILL_SECONDS = 1.5     # circle must stay still this long before grading
MOVE_PX = 30            # circle moving more than this = new answer
MAX_HINTS = 3
FORGET_SECONDS = 2.0    # forget a question that's been out of view this long
SAME_QUESTION_PX = 150  # centers closer than this = same question
WINDOW = "Over the Shoulder"

# ---------------- prompts ----------------
# If your old main.py had tuned prompts, paste them over these two.
PROMPT_QUESTION = """You are an OCR tool, not a tutor. Transcribe this multiple choice exam question exactly.
Return JSON only, in this format:
{"number": "question number", "type": "multiple_choice", "question": "question text",
 "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
 "circled": "the letter the student circled, or null"}
Ignore any other handwriting or rough work on the page. Do not solve the question."""

PROMPT_SHORT = """You are an OCR tool, not a tutor. Transcribe this short answer exam question and the student's handwritten work exactly.
Return JSON only, in this format:
{"number": "question number", "type": "short_answer", "question": "question text",
 "parts": {"A": {"question": "part question", "work": "student's handwritten work, copied exactly"}}}
Only include parts that have handwritten work. Copy the work exactly as written, even if it is wrong.
Never correct or finish the work. Use "?" for any character you can't read."""


# ---------------- helpers ----------------
def to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def parse_json(text):
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def ascii_only(s):
    return str(s).encode("ascii", "ignore").decode()


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def wrap(text, width=70):
    return textwrap.wrap(ascii_only(text), width) or [""]


# ---------------- shared state (always touch under lock) ----------------
lock = threading.Lock()
latest_frame = None
latest_crops = []
latest_det = None
questions = {}          # key -> state dict
message = "Circle an answer and hold still, or press Space for short answer."
jobs = Queue()
running = True

tracked = {}            # key -> (center, last_seen_time)
_ids = itertools.count(1)


def new_question():
    return {
        "status": "idle",       # idle / grading / correct / wrong / done
        "hints": [],
        "label": "Q",
        "circle": None,         # current circle center
        "still_since": 0.0,
        "graded_circle": None,  # circle center we last graded
    }


def key_for(box, create=True):
    """Give each question a stable name while it stays in view.
    Forgets questions that haven't been seen for FORGET_SECONDS.
    Call with lock held."""
    now = time.time()
    for k in [k for k, (_, seen) in tracked.items() if now - seen > FORGET_SECONDS]:
        del tracked[k]
        questions.pop(k, None)

    c = center(box)
    for k, (kc, _) in tracked.items():
        if dist(c, kc) < SAME_QUESTION_PX:
            tracked[k] = (c, now)
            return k
    if not create:
        return None
    k = f"Q{next(_ids)}"        # unique, never reused
    tracked[k] = (c, now)
    return k


# ---------------- threads ----------------
def detect_loop():
    global latest_crops, latest_det
    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            det = detect(frame)
            crops = get_question_crops(frame, det)
            with lock:
                latest_det = det
                latest_crops = crops
        except Exception as e:
            print("detect error:", e)
            time.sleep(0.5)


def grade_loop():
    global message
    while running:
        try:
            key, img, is_mc = jobs.get(timeout=0.2)
        except Empty:
            continue

        try:
            t0 = time.time()
            prompt = PROMPT_QUESTION if is_mc else PROMPT_SHORT
            info = parse_json(read_crop(to_pil(img), prompt))
            t1 = time.time()

            if info is None:
                with lock:
                    if key in questions:
                        questions[key]["status"] = "idle"
                        questions[key]["still_since"] = time.time()
                    message = "Couldn't read that - hold the page still."
                continue

            with lock:
                prev = list(questions[key]["hints"]) if key in questions else []
            result = check(info, prev)
            t2 = time.time()
            print(f"[{key}] Qwen {t1-t0:.1f}s, GPT {t2-t1:.1f}s")
            print(json.dumps(result, indent=2))

            with lock:
                if key not in questions:      # page moved away while grading
                    continue
                st = questions[key]
                st["label"] = f"Q{info.get('number', '')}" if info.get("number") else st["label"]

                if not result.get("readable", True):
                    st["status"] = "idle"
                    st["graded_circle"] = None
                    st["still_since"] = time.time()
                    message = f"{st['label']}: couldn't read it clearly - try again."
                    continue

                hint = result.get("hint")

                if is_mc:
                    if result.get("correct"):
                        st["status"] = "correct"
                        message = f"{st['label']}: correct!"
                    else:
                        if hint:
                            st["hints"].append(hint)
                        st["status"] = "done" if len(st["hints"]) >= MAX_HINTS else "wrong"
                        message = f"{st['label']}: {hint or 'not quite'}"
                        if st["status"] == "done":
                            message += "  (out of hints)"
                else:
                    parts = result.get("parts") or {}
                    marks, part_hint = [], None
                    for name, p in parts.items():
                        ok = p.get("correct") if isinstance(p, dict) else bool(p)
                        marks.append(f"{name}: {'OK' if ok else 'X'}")
                        if not ok and part_hint is None and isinstance(p, dict) and p.get("hint"):
                            part_hint = f"{name}: {p['hint']}"
                    part_hint = part_hint or (hint and f"{hint}")
                    all_ok = parts and all(
                        (p.get("correct") if isinstance(p, dict) else bool(p)) for p in parts.values()
                    )
                    if all_ok or (not parts and result.get("correct")):
                        st["status"] = "correct"
                        message = "Short answer: " + ("  ".join(marks) or "OK")
                    else:
                        if part_hint:
                            st["hints"].append(part_hint)
                        st["status"] = "done" if len(st["hints"]) >= MAX_HINTS else "wrong"
                        message = "  ".join(marks) + (f" | {part_hint}" if part_hint else "")
        except Exception as e:
            print(f"[{key}] grade error:", e)
            with lock:
                if key in questions:
                    questions[key]["status"] = "idle"
                message = "Something went wrong grading - try again."


# ---------------- triggers ----------------
def check_triggers(now):
    """Multiple choice: grade when a circle has been still for STILL_SECONDS."""
    global message
    with lock:
        crops = list(latest_crops)
        for q in crops:
            key = key_for(q["box"])
            st = questions.setdefault(key, new_question())
            if not q["is_mc"] or q["circle_box"] is None:
                continue

            c = center(q["circle_box"])
            if st["circle"] is None or dist(c, st["circle"]) > MOVE_PX:
                st["circle"] = c            # circle moved -> restart the timer
                st["still_since"] = now
                continue

            if st["status"] in ("grading", "correct", "done"):
                continue
            already_graded = st["graded_circle"] is not None and dist(c, st["graded_circle"]) <= MOVE_PX
            if already_graded:
                continue                    # same answer as last time, wait for a new one
            if now - st["still_since"] >= STILL_SECONDS:
                st["status"] = "grading"
                st["graded_circle"] = c
                message = "Checking..."
                jobs.put((key, q["crop"].copy(), True))


def grade_short_answer():
    global message
    with lock:
        sa = [q for q in latest_crops if not q["is_mc"]]
        if sa:
            img = sa[0]["crop"].copy()
        elif latest_frame is not None:
            img = latest_frame.copy()       # no box found -> use the whole frame
        else:
            return
        st = questions.setdefault("SA", new_question())
        if st["status"] == "grading":
            return
        st["status"] = "grading"
        st["label"] = "Short answer"
        message = "Checking short answer..."
    jobs.put(("SA", img, False))


# ---------------- drawing ----------------
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)


def draw(frame, show_boxes):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    with lock:
        det = latest_det
        crops = list(latest_crops)
        marks = []
        for q in crops:
            k = key_for(q["box"], create=False)
            qs = questions.get(k) if k else None
            if qs:
                marks.append((q["box"], qs["status"]))
        sa = questions.get("SA")
        sa_status = sa["status"] if sa else None
        msg = message

    if show_boxes and det is not None and len(det) > 0:
        canvas = box_annotator.annotate(canvas, det)

    for (x1, y1, x2, y2), status in marks:
        text, color = {
            "correct": ("OK", GREEN),
            "wrong": ("X", RED),
            "done": ("X", RED),
            "grading": ("...", YELLOW),
        }.get(status, (None, None))
        if text:
            cv2.putText(canvas, text, (max(10, int(x1)), max(40, int(y1) + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

    if sa_status and sa_status != "idle":
        text, color = {
            "correct": ("Short answer: OK", GREEN),
            "wrong": ("Short answer: X", RED),
            "done": ("Short answer: X", RED),
            "grading": ("Short answer: ...", YELLOW),
        }[sa_status]
        cv2.putText(canvas, text, (w - 520, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # bottom bar with the latest message / hint
    lines = wrap(msg, width=max(30, w // 22))[:3]
    bar_h = 20 + 40 * len(lines)
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (20, h - bar_h + 45 + 40 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    return canvas


# ---------------- main ----------------
def main():
    global latest_frame, running, message

    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=grade_loop, daemon=True).start()

    show_boxes = True
    fullscreen = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break

        with lock:
            latest_frame = frame

        check_triggers(time.time())
        cv2.imshow(WINDOW, draw(frame, show_boxes))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            grade_short_answer()
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("r"):
            with lock:
                questions.clear()
                tracked.clear()
                message = "Reset. Circle an answer and hold still, or press Space."

    running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()