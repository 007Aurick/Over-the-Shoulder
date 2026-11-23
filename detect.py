import cv2

import supervision as sv
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
import time

load_dotenv()



MODEL_ID = "over-the-shoulder/1"

client = InferenceHTTPClient(
    api_url = "https://serverless.roboflow.com",
    api_key = os.environ["ROBOFLOW_API_KEY"],
)

CONFIDENCE = 0.5
SEND_EVERY = 0.5
last_sent = 0
detections = sv.Detections.empty()

cap = cv2.VideoCapture(0)
box_annotator = sv.BoxAnnotator()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections", 960, 540)


while True:
    ref, frame = cap.read()
    if not ref:
        break
    now = time.time()

    if now - last_sent > SEND_EVERY:
        result = client.infer(frame, model_id=MODEL_ID)
        detections = sv.Detections.from_inference(result)
        detections = detections[detections.confidence >= CONFIDENCE]
        last_sent = now

    
    annotated = box_annotator.annotate(frame.copy(), detections)
    cv2.imshow("Detections", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite("screenshot.png", frame)
        print("Screenshot saved")
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()