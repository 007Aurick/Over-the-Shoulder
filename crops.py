import os

import cv2
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

MODEL_ID = "over-the-shoulder/1"
IMAGE = "screenshot.png"
CONFIDENCE = 0.5
CIRCLE_CONFIDENCE = 0.5
PAD = 20

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)


def inside(box, outer):
    """True if the center of box is inside outer."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def detect(frame):
    result = client.infer(frame, model_id=MODEL_ID)
    det = sv.Detections.from_inference(result).with_nms(0.5)
    names = det.data["class_name"]
    keep = (det.confidence >= CONFIDENCE) & (
        (names != "selected_choice") | (det.confidence >= CIRCLE_CONFIDENCE)
    )
    return det[keep]


def get_question_crops(frame, det):
    h, w = frame.shape[:2]
    names = det.data["class_name"]
    qboxes = det.xyxy[names == "Question"]
    answers = det.xyxy[names == "Answer"]
    circles = det.xyxy[names == "selected_choice"]
    circle_confs = det.confidence[names == "selected_choice"]

    qboxes = sorted(qboxes, key=lambda b: b[1])   # top to bottom
    out = []
    for qbox in qboxes:
        x1, y1, x2, y2 = qbox

        my_answers = [a for a in answers if inside(a, qbox)]
        for a in my_answers:                       # grow box to cover the answer
            x1, y1 = min(x1, a[0]), min(y1, a[1])
            x2, y2 = max(x2, a[2]), max(y2, a[3])

        x1, y1 = max(0, int(x1) - PAD), max(0, int(y1) - PAD)
        x2, y2 = min(w, int(x2) + PAD), min(h, int(y2) + PAD)

        best_circle, best_conf = None, 0.0
        for c, conf in zip(circles, circle_confs):
            if inside(c, (x1, y1, x2, y2)) and conf > best_conf:
                best_circle, best_conf = c, float(conf)

        out.append({
            "crop": frame[y1:y2, x1:x2],
            "box": (x1, y1, x2, y2),
            "is_mc": best_circle is not None,
            "circle_box": None if best_circle is None else tuple(int(v) for v in best_circle),
            "circle_conf": best_conf,
            "n_answers": len(my_answers),
        })
    return out


if __name__ == "__main__":
    frame = cv2.imread(IMAGE)
    det = detect(frame)
    crops = get_question_crops(frame, det)
    for n, q in enumerate(crops, 1):
        cv2.imwrite(f"crop_Q{n}.jpg", q["crop"])
        print(f"Q{n}: mc={q['is_mc']} circle_conf={q['circle_conf']:.2f} answers={q['n_answers']}")
    annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS).annotate(frame.copy(), det)
    cv2.imwrite("detections.jpg", annotated)