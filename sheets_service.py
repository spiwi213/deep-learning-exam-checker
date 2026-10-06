"""
sheets_service.py
Grades answers vs answer key and writes ONE row per student to Google Sheets.

Sheet layout (one tab per Subject + Section, auto-created):

  A1: Subject:   B1: Math
  A2: Section:   B2: 10-A
  A3: (blank)
  A4: Student Name | 1 | 2 | 3 | ... | 50 | Score
  A5: Juan Cruz    | ✓ | ✗ | ✓ | ... | ✓  | 42
"""
import os
import re
from typing import Any, List

import gspread
from gspread.exceptions import WorksheetNotFound

import answer_key as _ak

SUBJECT = getattr(_ak, "SUBJECT", "Subject")
SECTION = getattr(_ak, "SECTION", "Section")

SPREADSHEET_ID  = "1_HP8oqmJu_tCzbpy7uc99iH08MgazOD-OnPKFe9L-mQ"
CREDS_FILE      = os.getenv("GOOGLE_CREDS_FILE", "credentials.json")  # service account JSON
PASSING_PERCENT = float(os.getenv("PASSING_PERCENT", 75))

HEADER_ROW     = 4      # 1-indexed row of "Student Name | 1 | 2 ..."
FIRST_DATA_ROW = 5
MAX_ROWS       = 1000

CORRECT, WRONG, BLANK = "✓", "✗", "?"

_ss = None
_ws = None


# ─────────────────────────────────────────────
# ANSWER CLEANING / GRADING
# ─────────────────────────────────────────────

def _clean(value: Any) -> str:
    """Turn one detected answer into a clean string. Blank/None -> ''."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):      # multiple bubbles shaded
        return ",".join(str(v).strip().upper() for v in value if v)
    return str(value).strip().upper()


def normalize_answers(answers: Any, total: int) -> List[str]:
    """
    Accepts dict {1: "A", "2": "C", ...} or list ["A", "C", None, ...].
    Returns a list of exactly `total` strings (blank if no answer).
    """
    if isinstance(answers, dict):
        out = [""] * total
        for k, v in answers.items():
            try:
                idx = int(k) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= idx < total:
                out[idx] = _clean(v)
        return out

    out = [_clean(v) for v in (answers or [])]
    return out[:total] + [""] * max(0, total - len(out))


def grade(student_answers: List[str], answer_key: List[str]) -> dict:
    total = len(answer_key)
    score = sum(
        1 for s, k in zip(student_answers, answer_key)
        if s and s == str(k).strip().upper()
    )
    percent = round(score / total * 100, 2) if total else 0
    return {
        "score":   score,
        "total":   total,
        "percent": percent,
        "status":  "Pass" if percent >= PASSING_PERCENT else "Fail",
    }


def _mark(student: str, key: str) -> str:
    if student in ("", "?"):
        return BLANK                     # walang nakitang sagot -> i-check ang papel
    return CORRECT if student == str(key).strip().upper() else WRONG


# ─────────────────────────────────────────────
# SHEET SETUP (runs once, when the tab doesn't exist yet)
# ─────────────────────────────────────────────

def _tab_title() -> str:
    title = f"{SUBJECT} - {SECTION}"
    return re.sub(r"[\[\]\*\?:/\\]", "-", title)[:100]


def _rgb(r, g, b):
    return {"red": r, "green": g, "blue": b}


def _setup_layout(ss, ws, n: int):
    """Write headings + format the tab so 50 columns fit a long paper."""
    ws.update(values=[["Subject:", SUBJECT], ["Section:", SECTION]],
              range_name="A1", value_input_option="RAW")
    header = ["Student Name", *range(1, n + 1), "Score"]
    ws.update(values=[header], range_name=f"A{HEADER_ROW}", value_input_option="RAW")

    sid = ws.id
    last_col = n + 2          # exclusive end index: name + n questions + score

    def width(start, end, px):
        return {"updateDimensionProperties": {
            "range": {"sheetId": sid, "dimension": "COLUMNS",
                      "startIndex": start, "endIndex": end},
            "properties": {"pixelSize": px}, "fields": "pixelSize"}}

    def fmt(r0, r1, c0, c1, cell_fmt, fields):
        return {"repeatCell": {
            "range": {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1,
                      "startColumnIndex": c0, "endColumnIndex": c1},
            "cell": {"userEnteredFormat": cell_fmt}, "fields": fields}}

    def rule(text, bg, fg):
        return {"addConditionalFormatRule": {"index": 0, "rule": {
            "ranges": [{"sheetId": sid, "startRowIndex": FIRST_DATA_ROW - 1,
                        "endRowIndex": MAX_ROWS, "startColumnIndex": 1,
                        "endColumnIndex": 1 + n}],
            "booleanRule": {
                "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": text}]},
                "format": {"backgroundColor": bg, "textFormat": {"foregroundColor": fg}}}}}}

    requests = [
        # freeze headings + name column
        {"updateSheetProperties": {
            "properties": {"sheetId": sid, "gridProperties": {
                "frozenRowCount": HEADER_ROW, "frozenColumnCount": 1}},
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount"}},
        # widths: wide name, tiny question columns, medium score
        width(0, 1, 220),
        width(1, 1 + n, 26),
        width(1 + n, last_col, 60),
        # Subject / Section labels bold
        fmt(0, 2, 0, 1, {"textFormat": {"bold": True}}, "userEnteredFormat.textFormat"),
        # header row
        fmt(HEADER_ROW - 1, HEADER_ROW, 0, last_col,
            {"textFormat": {"bold": True, "fontSize": 9},
             "horizontalAlignment": "CENTER",
             "backgroundColor": _rgb(0.9, 0.9, 0.9)},
            "userEnteredFormat(textFormat,horizontalAlignment,backgroundColor)"),
        # student rows: small + centered (except name column)
        fmt(FIRST_DATA_ROW - 1, MAX_ROWS, 1, last_col,
            {"textFormat": {"fontSize": 9}, "horizontalAlignment": "CENTER"},
            "userEnteredFormat(textFormat,horizontalAlignment)"),
        # colors: ✓ green, ✗ red, ? yellow
        rule(BLANK,   _rgb(1.0, 0.95, 0.7),  _rgb(0.5, 0.4, 0.0)),
        rule(WRONG,   _rgb(0.96, 0.8, 0.8),  _rgb(0.7, 0.1, 0.1)),
        rule(CORRECT, _rgb(0.85, 0.95, 0.85), _rgb(0.1, 0.5, 0.1)),
    ]
    ss.batch_update({"requests": requests})


def _get_ws(n: int):
    """Connect once; open (or create) the Subject-Section tab."""
    global _ss, _ws
    if _ws is None:
        client = gspread.service_account(filename=CREDS_FILE)
        _ss    = client.open_by_key(SPREADSHEET_ID)
        title  = _tab_title()
        try:
            _ws = _ss.worksheet(title)
        except WorksheetNotFound:
            _ws = _ss.add_worksheet(title=title, rows=MAX_ROWS, cols=n + 2)
            _setup_layout(_ss, _ws, n)
    return _ws


# ─────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────

def append_result(student_name: str, answers: Any, answer_key: List[str]) -> dict:
    """Grade + write one row: Name | ✓/✗ x N | Score. Returns grade summary."""
    n       = len(answer_key)
    cleaned = normalize_answers(answers, n)
    result  = grade(cleaned, answer_key)

    ws    = _get_ws(n)
    marks = [_mark(c, k) for c, k in zip(cleaned, answer_key)]
    row   = [student_name, *marks, result["score"]]

    # Write to an EXACT cell (A5, A6...) so columns never shift
    next_row = max(len(ws.col_values(1)) + 1, FIRST_DATA_ROW)
    ws.update(values=[row], range_name=f"A{next_row}", value_input_option="RAW")
    return result


if __name__ == "__main__":
    # python sheets_service.py  -> creates the tab/layout without saving a student
    ws = _get_ws(len(_ak.ANSWER_KEY))
    print(f"✅ Ready: tab '{ws.title}'")
