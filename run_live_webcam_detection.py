"""
Engineering Calculator Detection - Real-Time Webcam Runner
Edhitha UAS Recruitment - Task 1 (ICS Sub-team)
Author: Aman Raj (rsguptaaman@gmail.com)

Features:
- YOLOv8-Nano object detection
- Context-Aware Adaptive Environmental Thresholding (dynamic confidence scaling based on luminance)
- Live HUD telemetry (FPS, Ambient Luminance, Dynamic Threshold, Target Status)
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

MODEL_PATH = "AmanRaj_1_ICS_weights.pt"

def compute_adaptive_threshold(frame):
    """
    Computes an adaptive confidence threshold based on ambient luminance (L).
    Standard luminance in dark/dim settings requires lower threshold (~0.60),
    whereas glare/high lighting scales smoothly up to 0.80 to reject false positives.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(np.mean(gray))
    
    # Scale confidence threshold between 0.60 and 0.80
    norm_lum = np.clip(mean_luminance / 255.0, 0.0, 1.0)
    adaptive_conf = 0.60 + (0.20 * norm_lum)
    return adaptive_conf, mean_luminance

def main():
    print(f"Loading YOLOv8 model weights from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam. Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: No accessible webcam found.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting live detection loop. Press 'q' or 'ESC' to exit.")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30.0
        prev_time = current_time

        # Compute dynamic threshold
        conf_thresh, luminance = compute_adaptive_threshold(frame)

        # Run inference
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        detections = results[0].boxes

        # Draw detections
        num_targets = 0
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            num_targets += 1
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # HUD Telemetry Banner
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), ( frame.shape[1], 45), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

        status_color = (0, 255, 0) if num_targets > 0 else (0, 165, 255)
        status_text = f"TRACKING ({num_targets})" if num_targets > 0 else "SEARCHING"

        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Luminance: {luminance:.1f}", (160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Adaptive Thresh: {conf_thresh:.2f}", (380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (680, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("Engineering Calculator Detection - YOLOv8 Live HUD", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam session closed successfully.")

if __name__ == "__main__":
    main()
