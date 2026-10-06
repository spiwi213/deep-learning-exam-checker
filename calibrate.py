"""
calibrate.py
────────────────────────────────────────────────
Standalone bubble detection tester + grading + Google Sheets.
Works best with: light pencil / ballpen shading
                 portrait orientation
                 dark background

Exam layout:
  Column 1: Q1  to Q17  (left)
  Column 2: Q18 to Q34  (middle)
  Column 3: Q35 to Q50  (right)

Colors in output.jpg:
  GREEN  = detected answer
  RED    = empty bubble

Usage:
  python calibrate.py                                  # test_photo.jpg, dry run
  python calibrate.py test_photo.jpg "Juan Dela Cruz"  # with student name
  python calibrate.py test_photo.jpg "Juan Dela Cruz" --save   # save to Google Sheets
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

from answer_key     import ANSWER_KEY
from sheets_service import grade, normalize_answers, append_result

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
SAVE_TO_SHEET = "--save" in sys.argv

IMAGE_PATH      = _args[0] if len(_args) > 0 else "test_photo.jpg"
STUDENT_NAME    = _args[1] if len(_args) > 1 else \
                  os.path.splitext(os.path.basename(IMAGE_PATH))[0].replace("_", " ")
MODEL_PATH      = "models/best.pt"
OUTPUT_PATH     = "output.jpg"
TOTAL_QUESTIONS = 50

# HoughCircles
MIN_DIST       = 20
PARAM1         = 50
PARAM2         = 28
MIN_RADIUS     = 8
MAX_RADIUS     = 20
ROW_TOLERANCE  = 18

# Fill detection
DARK_PIXEL_VAL = 128

# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────

print("=" * 50)
print("BUBBLE DETECTION CALIBRATOR")
print("=" * 50)

if len(ANSWER_KEY) != TOTAL_QUESTIONS:
    print(f"⚠️  ANSWER_KEY has {len(ANSWER_KEY)} items but TOTAL_QUESTIONS is {TOTAL_QUESTIONS}")

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ Could not load {IMAGE_PATH}")
    exit()
print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} px")
print(f"   Student: {STUDENT_NAME}")

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ─────────────────────────────────────────────
# STEP 2 — YOLO DETECT BUBBLE ZONE
# ─────────────────────────────────────────────

results    = model(image, conf=0.25, verbose=False)
detections = results[0].boxes

if len(detections) == 0:
    print("❌ No bubble zone detected!")
    exit()

print(f"✅ YOLO found {len(detections)} zone(s)")

# Pick the largest zone
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
print(f"   Zone: x1={x1} y1={y1} x2={x2} y2={y2} conf={conf:.1%}")

zone = image[y1:y2, x1:x2].copy()
print(f"   Zone size: {zone.shape[1]}x{zone.shape[0]} px")

# ─────────────────────────────────────────────
# STEP 3 — RESIZE
# ─────────────────────────────────────────────

target_width  = 700
scale         = target_width / zone.shape[1]
target_height = int(zone.shape[0] * scale)
zone          = cv2.resize(zone, (target_width, target_height))
print(f"   Resized: {zone.shape[1]}x{zone.shape[0]} px")

# ─────────────────────────────────────────────
# STEP 4 — DETECT CIRCLES
# ─────────────────────────────────────────────

gray    = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp        = 1.2,
    minDist   = MIN_DIST,
    param1    = PARAM1,
    param2    = PARAM2,
    minRadius = MIN_RADIUS,
    maxRadius = MAX_RADIUS
)

if circles is None:
    print("❌ No circles found!")
    exit()

circles = np.round(circles[0, :]).astype("int")
print(f"✅ Circles detected: {len(circles)} (expected ~200)")

# ─────────────────────────────────────────────
# STEP 5 — GROUP INTO ROWS
# ─────────────────────────────────────────────

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
print(f"✅ Valid rows: {len(rows)} (expected 17)")
print(f"   Circles per row: {[len(r) for r in rows]}")

# ─────────────────────────────────────────────
# STEP 6 — QUESTION LAYOUT
# ─────────────────────────────────────────────

col1 = list(range(1,  18))   # Q1  to Q17
col2 = list(range(18, 35))   # Q18 to Q34
col3 = list(range(35, 51))   # Q35 to Q50

# ─────────────────────────────────────────────
# HELPER — read one group of 4 bubbles
# ─────────────────────────────────────────────

def read_group(group, gray, output, q_num, letters):
    """
    Read one question's 4 bubbles.
    Picks the darkest bubble as the answer using mean pixel value.
    GREEN = answer, RED = empty.
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
    
    print(f"Q{q_num} → A={mean_vals[0]:.1f} B={mean_vals[1]:.1f} C={mean_vals[2]:.1f} D={mean_vals[3]:.1f} picked={['A','B','C','D'][mean_vals.index(min(mean_vals))]}")
    
    # Darkest bubble = lowest mean = the answer
    answer_idx = mean_vals.index(min(mean_vals))
    min_val    = mean_vals[answer_idx]

    # Draw circles
    for i, (x, y, r) in enumerate(group):
        if i == answer_idx:
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.putText(output, letters[i], (x - 6, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.circle(output, (x, y), r, (0, 0, 255), 1)

    # Draw question number
    cv2.putText(output, f"Q{q_num}",
                (group[0][0] - 28, group[0][1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)

    # Return ? only if all bubbles are completely white (empty paper)
    if min_val < 180:
        return letters[answer_idx]
    return "?"

# ─────────────────────────────────────────────
# STEP 7 — READ ALL ANSWERS
# ─────────────────────────────────────────────

output  = zone.copy()
answers = {}
letters = ["A", "B", "C", "D"]

for row_idx, row in enumerate(rows):

    # Fix 13-circle rows — remove first stray circle
    if len(row) == 13:
        row = row[1:]

    # Normal row — 3 questions (12 circles)
    if len(row) == 12:
        group1 = row[0:4]
        group2 = row[4:8]
        group3 = row[8:12]

        if row_idx < len(col1):
            answers[str(col1[row_idx])] = read_group(
                group1, gray, output, col1[row_idx], letters)

        if row_idx < len(col2):
            answers[str(col2[row_idx])] = read_group(
                group2, gray, output, col2[row_idx], letters)

        if row_idx < len(col3):
            answers[str(col3[row_idx])] = read_group(
                group3, gray, output, col3[row_idx], letters)

    # Last row — 2 questions only (8 circles)
    elif len(row) == 8:
        group1 = row[0:4]
        group2 = row[4:8]

        if row_idx < len(col1):
            answers[str(col1[row_idx])] = read_group(
                group1, gray, output, col1[row_idx], letters)

        if row_idx < len(col2):
            answers[str(col2[row_idx])] = read_group(
                group2, gray, output, col2[row_idx], letters)

# Draw YOLO box on full image
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 230, 80), 3)
cv2.putText(image, f"bubble_zone {conf:.0%}",
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 80), 2)

# ─────────────────────────────────────────────
# STEP 8 — SAVE IMAGES
# ─────────────────────────────────────────────

cv2.imwrite(OUTPUT_PATH,       output)
cv2.imwrite("output_full.jpg", image)
print(f"\n✅ Saved: {OUTPUT_PATH}")
print(f"✅ Saved: output_full.jpg")

# ─────────────────────────────────────────────
# STEP 9 — PRINT ANSWERS (with answer key check)
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("DETECTED ANSWERS")
print("=" * 50)

unknown = []
for q in range(1, TOTAL_QUESTIONS + 1):
    a   = answers.get(str(q), "?")
    key = ANSWER_KEY[q - 1] if q <= len(ANSWER_KEY) else "-"
    if a == "?":
        mark = "?"
        unknown.append(str(q))
    elif a == key:
        mark = "✓"
    else:
        mark = f"✗ (key: {key})"
    print(f"  Q{q:>2}: {a}  {mark}")

print(f"\nTotal:    {TOTAL_QUESTIONS}")
print(f"Detected: {TOTAL_QUESTIONS - len(unknown)}")
print(f"Unknown:  {len(unknown)}")
if unknown:
    print(f"Unknown Q: {', '.join(unknown)}")

# ─────────────────────────────────────────────
# STEP 10 — GRADE + SAVE TO GOOGLE SHEETS
# ─────────────────────────────────────────────

cleaned = normalize_answers(answers, len(ANSWER_KEY))
summary = grade(cleaned, ANSWER_KEY)

print("\n" + "=" * 50)
print("SCORE")
print("=" * 50)
print(f"  Student: {STUDENT_NAME}")
print(f"  Score:   {summary['score']}/{summary['total']}")
print(f"  Percent: {summary['percent']}%")
print(f"  Status:  {summary['status']}")

if SAVE_TO_SHEET:
    try:
        append_result(STUDENT_NAME, answers, ANSWER_KEY)
        print("  ✅ Saved to Google Sheets")
    except Exception as e:
        print(f"  ❌ Google Sheets error: {e}")
else:
    print("  (dry run — add --save to write to Google Sheets)")

print("\n" + "=" * 50)
print("CALIBRATION TIPS")
print("=" * 50)
print(f"  Circles: {len(circles)} (ideal ~200)")
if len(circles) < 180:
    print("  → Too few: lower PARAM2 or MIN_RADIUS")
elif len(circles) > 220:
    print("  → Too many: raise PARAM2 or MIN_DIST")
else:
    print("  → Circle count is good!")
if len(unknown) > 3:
    print(f"  → Too many ?: paper type needs more training photos")
else:
    print("  → Detection is good!")
