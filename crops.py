"""
Over the Shoulder - step 2: detect questions on a saved photo and crop them.

Run:  py crops.py
Needs: screenshot.png next to this file, ROBOFLOW_API_KEY in .env
Output: crop_Q1.jpg, crop_Q2.jpg, ... + a printout of what was found
"""

import os

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

MODEL_ID = "over-the-shoulder/1"
IMAGE = "screenshot.png"
CONFIDENCE = 0.5          # general cut-off
CIRCLE_CONFIDENCE = 0.7   # stricter for circles (erased marks score lower)
PAD = 20                  # extra pixels around each crop

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)


def inside(box, outer):#checks if the small box is inside the outer box
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 #calculates the center of the small box
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
    #Every box is [x1, y1, x2, y2], so for a multiple choice question, we check the center.
    #We use center instead of requiring the whole box to fit inside, because the circle box sometimes pokes slightly outside the question box.



def detect(frame):
    """Run RF-DETR once and clean up the detections."""
    result = client.infer(frame, model_id=MODEL_ID)#send image to RF-DETR, get predictions back
    det = sv.Detections.from_inference(result)#convert to a Detections object
    det = det.with_nms(threshold=0.5)#if two boxes of the same class overlap by more than 50%, it keeps the more confident one.

    #builds a true/false for every box
    names = det.data["class_name"]
    #it keeps a box if the cofidence is at least 0.5 and it's not a circle box or the confidence is at least 0.7
    keep = (det.confidence >= CONFIDENCE) & (
        (names != "selected_choice") | (det.confidence >= CIRCLE_CONFIDENCE)
    )
    return det[keep]#returns only the boxes that are questions or circle boxes with confidence >= 0.5 or >= 0.7


def get_question_crops(frame, det):
    """One crop per Question box, stretched to include its answers.
    Returns a list of dicts sorted top to bottom."""
    h, w = frame.shape[:2] #image height and width
    names = det.data["class_name"]
    questions = det[names == "Question"] #only question rows
    answers = det[names == "Answer"] #only answer rows
    circles = det[names == "selected_choice"] #only circle boxes

    out = []
    for i in np.argsort(questions.xyxy[:, 1]):              # top to bottom
        q = questions.xyxy[i]

        my_answers = [a for a in answers.xyxy if inside(a, q)]
        my_circles = [(c, conf) for c, conf in zip(circles.xyxy, circles.confidence) if inside(c, q)]

        # crop region = question box, grown to cover its answers
        x1, y1, x2, y2 = q
        for a in my_answers:
            x1, y1 = min(x1, a[0]), min(y1, a[1])
            x2, y2 = max(x2, a[2]), max(y2, a[3])

        # pad and keep inside the frame
        x1, y1 = int(max(0, x1 - PAD)), int(max(0, y1 - PAD))
        x2, y2 = int(min(w, x2 + PAD)), int(min(h, y2 + PAD))

        best_circle = max(my_circles, key=lambda t: t[1]) if my_circles else None

        out.append({
            "crop": frame[y1:y2, x1:x2],
            "box": (x1, y1, x2, y2),                        # where it is on the page (for drawing later)
            "is_mc": best_circle is not None,
            "circle_box": best_circle[0] if best_circle else None,
            "circle_conf": float(best_circle[1]) if best_circle else None,
            "n_answers": len(my_answers),
        })
    return out


if __name__ == "__main__":
    frame = cv2.imread(IMAGE)
    if frame is None:
        raise SystemExit(f"Couldn't open {IMAGE}")

    det = detect(frame)
    print("Detected:", dict(zip(*np.unique(det.data["class_name"], return_counts=True))))

    crops = get_question_crops(frame, det)
    for n, q in enumerate(crops, start=1):
        cv2.imwrite(f"crop_Q{n}.jpg", q["crop"])
        kind = f"multiple choice (circle conf {q['circle_conf']:.2f})" if q["is_mc"] else "short answer / unanswered"
        print(f"Q{n}: {kind}, {q['n_answers']} answer box(es) -> crop_Q{n}.jpg")

    # also save the full frame with boxes drawn, to compare against the crops
    annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS).annotate(frame.copy(), det)
    cv2.imwrite("detections.jpg", annotated)
    print("Saved detections.jpg")