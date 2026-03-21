"""
detector.py
────────────────────────────────────────────────
Detects and straightens exam paper from camera frame.
Called by app.py on every camera frame.

Input:  OpenCV image (BGR)
Output: (annotated_frame, warped_frame or None, detected bool)
"""

import cv2
import numpy as np


def order_points(pts):
    """Order 4 corners: top-left, top-right, bottom-right, bottom-left."""
    rect    = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Warp paper to flat top-down view."""
    rect             = order_points(pts)
    (tl, tr, br, bl) = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst  = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M    = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def find_paper_contour(gray, image_area):
    """Try multiple strategies to find the paper edges."""
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel   = np.ones((3, 3), np.uint8)
    otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edge_maps = [
        cv2.Canny(blurred, 20, 80),
        cv2.Canny(blurred, otsu_val * 0.4, otsu_val),
        cv2.Canny(
            cv2.adaptiveThreshold(blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2), 10, 50),
        cv2.dilate(cv2.Canny(blurred, 20, 80), kernel, iterations=2),
    ]

    for edges in edge_maps:
        cnts, _ = cv2.findContours(edges.copy(),
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < image_area * 0.08 or area > image_area * 0.97:
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx

    return None


def detect_paper(frame):
    """
    Main function called by app.py.
    Input:  OpenCV BGR image
    Output: (annotated_frame, warped_frame or None, detected bool)
    """
    original = frame.copy()
    h, w     = frame.shape[:2]
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    approx   = find_paper_contour(gray, h * w)

    if approx is not None:
        pts = approx.reshape(4, 2).astype("float32")

        # Draw green outline
        cv2.drawContours(frame, [approx], -1, (0, 230, 80), 4)
        for pt in approx.reshape(4, 2):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(frame, (px, py), 12, (0, 230, 80), -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

        # Green status bar
        cv2.rectangle(frame, (0, 0), (w, 52), (0, 170, 60), -1)
        cv2.putText(frame, "PAPER DETECTED", (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        warped = four_point_transform(original, pts)
        return frame, warped, True

    else:
        # Dark scanning bar
        cv2.rectangle(frame, (0, 0), (w, 52), (25, 25, 25), -1)
        cv2.putText(frame, "SCANNING...", (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)

        # Orange guide box
        m = 45
        cv2.rectangle(frame, (m, m + 52), (w - m, h - m),
                      (0, 140, 255), 2)
        cv2.putText(frame, "Aim paper inside this box",
                    (m + 8, h - m - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        return frame, None, False