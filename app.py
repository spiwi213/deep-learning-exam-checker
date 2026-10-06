"""
app.py
────────────────────────────────────────────────
Exam Scanner - FastAPI backend

Routes:
  GET  /          → scan page
  GET  /results   → results page
  POST /scan      → paper detection (camera frame)
  POST /detect    → read bubble answers, grade, save to Google Sheets
"""

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import os

from detector       import detect_paper
from bubble_reader  import read_bubble_answers
from sheets_service import append_result, normalize_answers, grade

# ─────────────────────────────────────────────
# ANSWER KEY  (palitan ng totoong key mo — 50 items)
# ─────────────────────────────────────────────
from answer_key import ANSWER_KEY   # edit answer_key.py para palitan

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

app       = FastAPI(title="Exam Scanner")
templates = Jinja2Templates(directory="templates")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def base64_to_opencv(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def opencv_to_base64(image):
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def scan_page(request: Request):
    return templates.TemplateResponse("scan.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})


# ─────────────────────────────────────────────
# API — PAPER DETECTION
# ─────────────────────────────────────────────

@app.post("/scan")
async def scan(image: str = Form(...)):
    """
    Receives camera frame every 800ms.
    Detects paper outline + returns annotated + warped images.
    """
    frame = base64_to_opencv(image)
    if frame is None:
        return JSONResponse({"error": "Bad image"}, status_code=400)

    annotated, warped, detected = detect_paper(frame)

    _, buf1       = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf1).decode()

    warped_b64 = None
    if warped is not None:
        _, buf2    = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 82])
        warped_b64 = "data:image/jpeg;base64," + base64.b64encode(buf2).decode()

    return JSONResponse({
        "detected":  detected,
        "annotated": annotated_b64,
        "warped":    warped_b64
    })


# ─────────────────────────────────────────────
# API — BUBBLE READING + GRADING + GOOGLE SHEETS
# ─────────────────────────────────────────────

@app.post("/detect")
async def detect(
    warped_image:  str = Form(...),
    student_name:  str = Form(...),
    save_to_sheet: bool = Form(True)   # i-send ang "false" para di mag-save (test mode)
):
    """
    1. Reads bubble answers from the warped (flat) paper image
    2. Grades them against ANSWER_KEY
    3. Appends one row to Google Sheets
    """
    image = base64_to_opencv(warped_image)
    if image is None:
        return JSONResponse({"error": "Bad image"}, status_code=400)

    result = read_bubble_answers(image)

    if result["error"]:
        return JSONResponse({"error": result["error"]}, status_code=500)

    annotated_b64 = opencv_to_base64(result["annotated"])

    # ── Grade ──
    cleaned = normalize_answers(result["answers"], len(ANSWER_KEY))
    summary = grade(cleaned, ANSWER_KEY)

    # ── Save to Google Sheets (hindi mag-fa-fail ang scan kapag may sheet error) ──
    saved       = False
    sheet_error = None
    if save_to_sheet:
        try:
            # run_in_threadpool: blocking ang gspread, para di ma-freeze ang server
            await run_in_threadpool(append_result, student_name, result["answers"], ANSWER_KEY)
            saved = True
        except Exception as e:
            sheet_error = str(e)
            print(f"[Sheets] ERROR: {e}")

    return JSONResponse({
        "student_name": student_name,
        "answers":      result["answers"],
        "annotated":    annotated_b64,
        "detected":     result["detected"],
        "score":        summary["score"],
        "total":        summary["total"],
        "percent":      summary["percent"],
        "status":       summary["status"],
        "saved":        saved,
        "sheet_error":  sheet_error
    })


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host         = "0.0.0.0",
        port         = 5001,
        ssl_keyfile  = "server.key",
        ssl_certfile = "server.crt"
    )
