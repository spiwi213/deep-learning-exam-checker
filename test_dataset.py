"""
test_dataset.py
Test the full pipeline on dataset images WITHOUT camera/server.

Usage:
  python test_dataset.py dataset/                 # dry run (hindi mag-save sa Sheet)
  python test_dataset.py dataset/ --save          # mag-save sa Google Sheets
  python test_dataset.py dataset/sheet1.jpg       # isang image lang

Student name = filename (e.g. juan_delacruz.jpg -> "juan delacruz")
"""
import sys
import glob
import os
import cv2

from detector       import detect_paper
from bubble_reader  import read_bubble_answers
from sheets_service import append_result, normalize_answers, grade
from answer_key     import ANSWER_KEY

EXTS = ("*.jpg", "*.jpeg", "*.png")


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in EXTS:
        files += glob.glob(os.path.join(path, ext))
    return sorted(files)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]
    save   = "--save" in sys.argv

    images = collect_images(target)
    if not images:
        print("Walang nahanap na images.")
        return

    for path in images:
        name  = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        frame = cv2.imread(path)
        if frame is None:
            print(f"[SKIP] {path}: hindi mabasa")
            continue

        # Kung raw photo, i-warp muna. Kung flat/scanned na, gamitin as-is.
        _, warped, detected = detect_paper(frame)
        image = warped if (detected and warped is not None) else frame

        result = read_bubble_answers(image)
        if result["error"]:
            print(f"[ERROR] {name}: {result['error']}")
            continue

        cleaned = normalize_answers(result["answers"], len(ANSWER_KEY))
        summary = grade(cleaned, ANSWER_KEY)
        print(f"{name:25} {summary['score']}/{summary['total']}  "
              f"{summary['percent']}%  {summary['status']}")

        if save:
            try:
                append_result(name, result["answers"], ANSWER_KEY)
                print("   -> saved to Google Sheets")
            except Exception as e:
                print(f"   -> SHEETS ERROR: {e}")


if __name__ == "__main__":
    main()
