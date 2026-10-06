"""
bubble_reader.py
────────────────────────────────────────────────
Reads bubble answers from a warped exam paper image.
Used by app.py for live scanning.

Usage:
    from bubble_reader import read_bubble_answers
    result = read_bubble_answers(warped_image_opencv)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

MODEL_PATH     = "models/best.pt"
MIN_DIST       = 20
PARAM1         = 50
PARAM2         = 28
MIN_RADIUS     = 8
MAX_RADIUS     = 20
ROW_TOLERANCE  = 18

COL1    = list(range(1,  18))   # Q1  to Q17
COL2    = list(range(18, 35))   # Q18 to Q34
COL3    = list(range(35, 51))   # Q35 to Q50
LETTERS = ["A", "B", "C", "D"]

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
        print("✅ YOLOv8 model loaded")
    return _model


# ─────────────────────────────────────────────
# READ ONE QUESTION (4 bubbles)
# ─────────────────────────────────────────────

def read_group(group, gray, q_num):
    """
    Read one question's 4 bubbles.
    Picks the darkest bubble as the answer.
    Returns ? if no bubble is clearly filled.
    """
    mean_vals = []
    for (x, y, r) in group:
        mask   = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r - 2, 255, -1)
        pixels = gray[mask == 255]
        if len(pixels) == 0:
            mean_vals.append(255.0)
            continue
        mean_vals.append(float(np.mean(pixels)))

    answer_idx  = mean_vals.index(min(mean_vals))
    min_val     = mean_vals[answer_idx]
    sorted_vals = sorted(mean_vals)
    difference  = sorted_vals[1] - sorted_vals[0]

    # Only count as answered if:
    # 1. Darkest bubble is clearly dark (< 180)
    # 2. It is clearly darker than the next bubble (difference > 15)
    if min_val < 180 and difference > 15:
        return LETTERS[answer_idx]
    return "?"


# ─────────────────────────────────────────────
# DRAW CIRCLES ON IMAGE
# ─────────────────────────────────────────────

def draw_group(group, gray, output, q_num, answer):
    """Draw GREEN on answer bubble, RED on empty bubbles."""
    mean_vals = []
    for (x, y, r) in group:
        mask   = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r - 2, 255, -1)
        pixels = gray[mask == 255]
        mean_vals.append(float(np.mean(pixels)) if len(pixels) > 0 else 255.0)

    answer_idx = mean_vals.index(min(mean_vals))

    for i, (x, y, r) in enumerate(group):
        if i == answer_idx and answer != "?":
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.putText(output, LETTERS[i], (x - 6, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.circle(output, (x, y), r, (0, 0, 255), 1)

    cv2.putText(output, f"Q{q_num}",
                (group[0][0] - 28, group[0][1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def read_bubble_answers(image):
    """
    Main function called by app.py.
    Input:  OpenCV BGR image of the exam paper
    Output: dict with answers, annotated image, detected count, error
    """
    try:
        model  = get_model()
        output = image.copy()

        # ── YOLO detect bubble zone ──
        results    = model(image, conf=0.25, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            return {
                "answers":   {},
                "annotated": output,
                "detected":  0,
                "error":     "No bubble zone detected. Hold paper flat and steady."
            }

        # Pick the largest detected zone
        largest_area = -1
        best_box     = None
        for b in detections:
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            area = (bx2 - bx1) * (by2 - by1)
            if area > largest_area:
                largest_area = area
                best_box     = b

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        conf            = float(best_box.conf[0])

        # Crop + resize zone to standard width
        zone          = image[y1:y2, x1:x2].copy()
        target_width  = 700
        scale         = target_width / zone.shape[1]
        target_height = int(zone.shape[0] * scale)
        zone          = cv2.resize(zone, (target_width, target_height))
        zone_output   = zone.copy()

        # ── Detect circles ──
        gray    = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=MIN_DIST, param1=PARAM1, param2=PARAM2,
            minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
        )

        if circles is None:
            return {
                "answers":   {},
                "annotated": output,
                "detected":  0,
                "error":     "No circles detected. Try better lighting."
            }

        circles = np.round(circles[0, :]).astype("int")

        # ── Group circles into rows ──
        circles     = sorted(circles, key=lambda c: c[1])
        rows        = []
        current_row = [circles[0]]

        for circle in circles[1:]:
            if abs(circle[1] - current_row[0][1]) < ROW_TOLERANCE:
                current_row.append(circle)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [circle]
        rows.append(sorted(current_row, key=lambda c: c[0]))

        # Keep only valid rows
        rows = [r for r in rows if len(r) in [8, 12, 13]]

        # ── Read answers row by row ──
        answers = {}

        for row_idx, row in enumerate(rows):
            # Fix 13-circle rows — remove stray circle on the left
            if len(row) == 13:
                row = row[1:]

            if len(row) == 12:
                groups = [(row[0:4], COL1), (row[4:8], COL2), (row[8:12], COL3)]
            elif len(row) == 8:
                groups = [(row[0:4], COL1), (row[4:8], COL2)]
            else:
                continue

            for group, col in groups:
                if row_idx < len(col):
                    q_num  = col[row_idx]
                    answer = read_group(group, gray, q_num)
                    answers[str(q_num)] = answer
                    draw_group(group, gray, zone_output, q_num, answer)

        # Draw YOLO box on full image
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 230, 80), 3)
        cv2.putText(output, f"bubble_zone {conf:.0%}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 80), 2)

        detected = len([a for a in answers.values() if a != "?"])

        return {
            "answers":   answers,
            "annotated": zone_output,
            "detected":  detected,
            "error":     None
        }

    except Exception as e:
        return {
            "answers":   {},
            "annotated": image,
            "detected":  0,
            "error":     str(e)
        }