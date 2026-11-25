import itertools
import json
import re
import textwrap
import threading
import time
from queue import Queue, Empty

import cv2
import supervision as sv

from crops import detect, get_question_crops
from check_image import check_mc_image, check_sa_image

CAMERA = 0
STILL_SECONDS = 1.5     # circle must stay still this long before grading
MOVE_PX = 30            # circle jitter smaller than this is ignored
REGRADE_PX = 60         # circle must move this far from the graded spot to count as a new answer
MAX_HINTS = 3
FORGET_SECONDS = 4.0    # forget a question that's been out of view this long
SAME_QUESTION_PX = 120  # centers closer than this = same question (after motion tracking)
FLOW_W = 320            # motion tracking works on a tiny copy of the frame this wide
FLOW_SIGN = 1           # if marks slide the WRONG way when you move the page, set to -1
GRADE_WORKERS = 3       # how many questions can be graded at once
SA_PAUSE = 2.5          # short answer: auto-grade after the pen/hand is still this long...
SA_MIN_GAP = 6.0        # ...but not more often than this
INK_PIXELS = 150        # ...and only if at least this many pixels of new ink appeared
ACTIVITY = 0.005        # fraction of pixels changing per frame above this = someone is writing/moving
WINDOW = "Over the Shoulder"

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



lock = threading.Lock()
latest_frame = None
latest_crops = []
latest_det = None
questions = {}          # key -> state dict
tracked = {}            # key -> {"center", "box", "seen"}
order = []              # keys in the order they were graded (for the bottom bar)
jobs = Queue()
running = True
_ids = itertools.count(1)
shift = [0.0, 0.0]          # total page movement so far (px), from motion tracking
latest_crops_shift = (0.0, 0.0)   # value of `shift` when latest_crops' frame was captured
frame_size = (1920, 1080)
last_active = 0.0           # last time the frame was changing (hand writing / page moving)
sa_snapshot = None          # grayscale frame at the last short answer grade
sa_snapshot_shift = (0.0, 0.0)


def new_question():
    return {
        "status": "idle",       # idle / grading / correct / wrong / done
        "hints": [],
        "label": "Q",
        "circle": None,         # current (smoothed) circle center
        "still_since": 0.0,
        "graded_circle": None,  # circle center we last graded
    }


def forget_stale(now):
    w, h = frame_size
    for k in list(tracked):
        cx, cy = tracked[k]["center"]
        off_screen = cx < 0 or cy < 0 or cx > w or cy > h
        if off_screen or now - tracked[k]["seen"] > FORGET_SECONDS:
            del tracked[k]
            questions.pop(k, None)
            if k in order:
                order.remove(k)


def key_for(box, now, used):
    """Stable name for a question while it stays in view. Call with lock held.
    `used` = keys already claimed this frame, so two questions never share one."""
    c = center(box)
    best, best_d = None, SAME_QUESTION_PX
    for k, t in tracked.items():
        if k in used:
            continue
        d = dist(c, t["center"])
        if d < best_d:
            best, best_d = k, d
    if best is None:
        best = f"Q{next(_ids)}"
    tracked[best] = {"center": c, "box": box, "seen": now}
    used.add(best)
    return best


def shift_box(box, d):
    x1, y1, x2, y2 = box
    return (x1 + d[0], y1 + d[1], x2 + d[0], y2 + d[1])


def apply_motion(dx, dy):
    """The page moved by (dx, dy) px: move everything we remember with it. Call with lock held."""
    shift[0] += dx
    shift[1] += dy
    for t in tracked.values():
        t["center"] = (t["center"][0] + dx, t["center"][1] + dy)
        t["box"] = shift_box(t["box"], (dx, dy))
    for st in questions.values():
        for name in ("circle", "graded_circle"):
            if st[name] is not None:
                st[name] = (st[name][0] + dx, st[name][1] + dy)
        if st.get("part_pos"):
            st["part_pos"] = {n: (x + dx, y + dy) for n, (x, y) in st["part_pos"].items()}


def detect_loop():
    global latest_crops, latest_det, latest_crops_shift
    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
            s0 = (shift[0], shift[1])
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            det = detect(frame)
            crops = get_question_crops(frame, det)
            with lock:
                latest_det = det
                latest_crops = crops
                latest_crops_shift = s0
        except Exception as e:
            print("detect error:", e)
            time.sleep(0.5)


def grade_mc(key, img, prev):
    t0 = time.time()
    result = check_mc_image(img, prev)
    print(f"[{key}] GPT (image) {time.time()-t0:.1f}s ->", json.dumps(result))
    return result


def grade_sa(key, img, prev, final=False):
    t0 = time.time()
    result = check_sa_image(img, prev, final)
    print(f"[{key}] GPT (image, short answer) {time.time()-t0:.1f}s ->", json.dumps(result))
    return result


def grade_loop():
    while running:
        try:
            key, img, is_mc, final = jobs.get(timeout=0.2)
        except Empty:
            continue
        try:
            with lock:
                prev = list(questions[key]["hints"]) if key in questions else []
            result = grade_mc(key, img, prev) if is_mc else grade_sa(key, img, prev, final)

            with lock:
                if key not in questions:      # page moved away while grading
                    continue
                st = questions[key]
                if result.get("number"):
                    st["label"] = f"Q{ascii_only(result['number']).strip('Q. ')}"
                if key in order:
                    order.remove(key)
                order.append(key)

                if not result.get("readable", True):
                    st["status"] = "idle"
                    st["graded_circle"] = None
                    st["still_since"] = time.time()
                    st["note"] = "couldn't read it - hold still"
                    continue

                if is_mc:
                    ok = bool(result.get("correct"))
                    hint = result.get("hint")
                else:
                    # merge with earlier grades: part A stays graded while you work on B
                    # only parts GPT sees NOW count (erased/unfinished parts drop off);
                    # a part that was right before stays right
                    old_parts = st.get("parts", {})
                    merged, hint = {}, None
                    for name, p in (result.get("parts") or {}).items():
                        p_ok = bool(p.get("correct")) if isinstance(p, dict) else bool(p)
                        merged[name] = True if old_parts.get(name) is True else p_ok
                        if not merged[name] and hint is None and isinstance(p, dict) and p.get("hint"):
                            hint = f"{name}: {p['hint']}"
                    st["parts"] = merged
                    # where to draw each part's mark: GPT gives x/y as fractions of the crop
                    bx1, by1, bx2, by2 = st["crop_box"]
                    ddx, ddy = shift[0] - st["crop_shift"][0], shift[1] - st["crop_shift"][1]
                    pos = {}
                    for name, p in (result.get("parts") or {}).items():
                        if isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None:
                            try:
                                fx = min(max(float(p["x"]), 0.0), 1.0)
                                fy = min(max(float(p["y"]), 0.0), 1.0)
                            except (TypeError, ValueError):
                                continue
                            pos[name] = (bx1 + fx * (bx2 - bx1) + ddx, by1 + fy * (by2 - by1) + ddy)
                    st["part_pos"] = pos
                    st["marks"] = "  ".join(f"{n}: {'OK' if ok_ else 'X'}" for n, ok_ in sorted(merged.items()))
                    if not merged:
                        st["status"] = "idle"
                        st["marks"] = ""
                        st["note"] = "keep going - nothing finished to grade yet"
                        continue
                    if all(merged.values()):
                        st["status"] = "correct"
                        st["note"] = "all done parts correct!"
                    else:
                        st["status"] = "wrong"
                        if hint:
                            st["hints"] = (st["hints"] + [hint])[-3:]   # keep last 3 for context
                            st["note"] = hint
                        else:
                            st["note"] = st.get("note") or "check your work"
                    continue

                if ok:
                    st["status"] = "correct"
                    st["note"] = "correct!"
                else:
                    if hint:
                        st["hints"].append(hint)
                    st["status"] = "done" if len(st["hints"]) >= MAX_HINTS else "wrong"
                    st["note"] = hint or "not quite"
        except Exception as e:
            print(f"[{key}] grade error:", e)
            with lock:
                if key in questions:
                    questions[key]["status"] = "idle"
                    questions[key]["graded_circle"] = None
                    questions[key]["note"] = "error grading - try again"


def check_triggers(now):
    """Multiple choice: grade every question whose circle has been still for STILL_SECONDS."""
    with lock:
        forget_stale(now)
        # detection is a bit behind the camera: move its boxes by how far the page moved since
        d = (shift[0] - latest_crops_shift[0], shift[1] - latest_crops_shift[1])
        used = set()
        for q in latest_crops:
            key = key_for(shift_box(q["box"], d), now, used)
            st = questions.setdefault(key, new_question())
            if not q["is_mc"] or q["circle_box"] is None:
                continue                    # circle missed this frame: don't reset anything

            cx, cy = center(q["circle_box"])
            c = (cx + d[0], cy + d[1])
            if st["circle"] is None or dist(c, st["circle"]) > MOVE_PX:
                st["circle"] = c            # circle really moved -> restart the timer
                st["still_since"] = now
                continue

            if st["status"] in ("grading", "correct", "done"):
                continue
            if st["graded_circle"] is not None and dist(c, st["graded_circle"]) < REGRADE_PX:
                continue                    # same answer as last time
            if now - st["still_since"] >= STILL_SECONDS:
                st["status"] = "grading"
                st["graded_circle"] = c
                jobs.put((key, q["crop"].copy(), True, False))


def find_short_answer(frame):
    """Current short answer crop (from the live frame) and its box, or (None, None).
    A short answer = a question with handwritten Answer boxes but no circled choice.
    Call with lock held."""
    d = (shift[0] - latest_crops_shift[0], shift[1] - latest_crops_shift[1])
    sa = [q for q in latest_crops if not q["is_mc"] and q["n_answers"] > 0]
    h, w = frame.shape[:2]
    if not sa:
        # the model sometimes finds the handwritten Answer box but no Question box on short
        # answer pages: if there are Answer boxes and no circled choices, use the whole frame
        det = latest_det
        if det is not None and len(det) > 0:
            names = list(det.data["class_name"])
            if "Answer" in names and "selected_choice" not in names:
                return frame.copy(), (0, 0, w, h)
        return None, None
    x1, y1, x2, y2 = shift_box(sa[0]["box"], d)
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 50 or y2 - y1 < 50:
        return None, None
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def start_sa_grade(img, gray, now, box, final=False):
    """Queue a short answer grade. Call with lock held. `box` = where img sits in the frame."""
    global sa_snapshot, sa_snapshot_shift
    st = questions.setdefault("SA", new_question())
    if st["status"] == "grading":
        return
    st["status"] = "grading"
    st["crop_box"], st["crop_shift"] = box, (shift[0], shift[1])
    st["label"] = "Short answer"
    st["last_graded"] = now
    sa_snapshot, sa_snapshot_shift = gray, (shift[0], shift[1])
    jobs.put(("SA", img, False, final))


def check_sa_trigger(now, frame, gray):
    """Short answer: when the student pauses after writing new work, grade what's finished.
    GPT only grades finished parts, so finishing A and starting B gets A marked."""
    with lock:
        img, box = find_short_answer(frame)
        if img is None:
            return
        st = questions.get("SA")
        if st and st["status"] == "grading":
            return
        if now - last_active < SA_PAUSE:
            return                          # still writing
        if st and now - st.get("last_graded", 0) < SA_MIN_GAP:
            return

        if sa_snapshot is None:
            changed = True
        elif dist(shift, sa_snapshot_shift) > 40:
            changed = True                  # page moved a lot since last grade: just recheck
        else:
            sh, sw = gray.shape
            fh, fw = frame.shape[:2]
            gx1, gy1 = int(box[0] * sw / fw), int(box[1] * sh / fh)
            gx2, gy2 = int(box[2] * sw / fw), int(box[3] * sh / fh)
            diff = cv2.absdiff(gray[gy1:gy2, gx1:gx2], sa_snapshot[gy1:gy2, gx1:gx2])
            changed = int((diff > 40).sum()) > INK_PIXELS
        if changed:
            start_sa_grade(img, gray, now, box)


def grade_short_answer(frame, gray):
    with lock:
        img, box = find_short_answer(frame)
        if img is None:
            img = frame.copy()              # no box found -> use the whole frame
            box = (0, 0, frame.shape[1], frame.shape[0])
        start_sa_grade(img, gray, time.time(), box, final=True)


box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)
MARK = {"correct": ("OK", GREEN), "wrong": ("X", RED), "done": ("X", RED), "grading": ("...", YELLOW)}


def draw(frame, show_boxes):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    with lock:
        det = latest_det
        # draw marks at the LAST KNOWN position, so a missed detection doesn't make them blink
        marks = [(tracked[k]["box"], questions[k]["status"])
                 for k in tracked if k in questions]
        lines = []
        for k in order[-3:]:
            st = questions.get(k)
            if st and st.get("note"):
                prefix = st["label"] + (f" ({st['marks']})" if st.get("marks") else "")
                lines.append(f"{prefix}: {st['note']}")
        grading = [questions[k]["label"] for k in questions if questions[k]["status"] == "grading"]
        sa = questions.get("SA")
        sa_status = sa["status"] if sa else None
        sa_marks = []
        if sa:
            for name, ok_ in sa.get("parts", {}).items():
                if name in sa.get("part_pos", {}):
                    sa_marks.append((sa["part_pos"][name], ok_))

    if show_boxes and det is not None and len(det) > 0:
        canvas = box_annotator.annotate(canvas, det)

    for (x1, y1, x2, y2), status in marks:
        if status in MARK:
            text, color = MARK[status]
            cv2.putText(canvas, text, (max(10, int(x1)), max(40, int(y1) + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

    if sa_status == "grading":
        cv2.putText(canvas, "Short answer: ...", (w - 520, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 3)
    for (px, py), ok_ in sa_marks:
        text, color = ("OK", GREEN) if ok_ else ("X", RED)
        x = int(min(max(px + 30, 10), w - 110))        # just right of the answer
        y = int(min(max(py + 20, 50), h - 10))
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)

    # bottom bar: one line per recently graded question
    if grading:
        lines.append("Checking " + ", ".join(grading) + "...")
    if not lines:
        lines = ["Circle an answer and hold still, or press Space for short answer."]
    out = []
    for line in lines:
        out += wrap(line, width=max(30, w // 22))
    out = out[-4:]
    bar_h = 20 + 40 * len(out)
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(out):
        cv2.putText(canvas, line, (20, h - bar_h + 45 + 40 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    return canvas


def main():
    global latest_frame, running, last_active, sa_snapshot

    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    threading.Thread(target=detect_loop, daemon=True).start()
    for _ in range(GRADE_WORKERS):
        threading.Thread(target=grade_loop, daemon=True).start()

    global frame_size
    show_boxes = True
    fullscreen = False
    prev_small = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break

        # motion tracking: how far did the page slide since the last frame?
        h, w = frame.shape[:2]
        small = cv2.cvtColor(cv2.resize(frame, (FLOW_W, FLOW_W * h // w)), cv2.COLOR_BGR2GRAY)
        small = small.astype("float32")
        dx = dy = 0.0
        if prev_small is not None:
            (sx, sy), response = cv2.phaseCorrelate(prev_small, small)
            scale = w / FLOW_W
            if response > 0.1 and abs(sx) < FLOW_W / 4 and abs(sy) < FLOW_W / 4:
                dx, dy = sx * scale * FLOW_SIGN, sy * scale * FLOW_SIGN
        now = time.time()
        if prev_small is not None and float((cv2.absdiff(prev_small, small) > 25).mean()) > ACTIVITY:
            last_active = now               # hand writing or page moving
        prev_small = small
        ink_gray = cv2.cvtColor(cv2.resize(frame, (960, 960 * h // w)), cv2.COLOR_BGR2GRAY)

        with lock:
            latest_frame = frame
            frame_size = (w, h)
            if abs(dx) > 1 or abs(dy) > 1:      # ignore sensor noise
                apply_motion(dx, dy)

        check_triggers(now)
        check_sa_trigger(now, frame, ink_gray)
        cv2.imshow(WINDOW, draw(frame, show_boxes))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            grade_short_answer(frame, ink_gray)
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
                order.clear()
                latest_crops.clear()
                sa_snapshot = None

    running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()