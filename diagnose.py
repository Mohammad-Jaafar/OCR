

import sys

import cv2
import numpy as np

import scan 


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python diagnose.py photo.jpg")

    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit("Could not read " + sys.argv[1])

    ratio = max(image.shape[0] / scan.WORK_HEIGHT, 1.0)
    small = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    area = small.shape[0] * small.shape[1]
    print("photo    : {}x{}".format(image.shape[1], image.shape[0]))
    print("searched : {}x{}".format(small.shape[1], small.shape[0]))
    print("a candidate must have 4 corners AND cover >10% ({} px)\n".format(int(0.10 * area)))
    print("  #  corners   coverage   verdict")

    annotated = small.copy()
    winner = None
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        covered = cv2.contourArea(approx) / area

        if len(approx) != 4:
            verdict = "not a quad ({} corners)".format(len(approx))
        elif covered <= 0.10:
            verdict = "too small"
        else:
            verdict = "PAGE FOUND"
            if winner is None:
                winner = approx

        print("  {}  {:>7}   {:>7.1%}   {}".format(i + 1, len(approx), covered, verdict))
        color = (0, 255, 0) if verdict == "PAGE FOUND" else (0, 0, 255)
        cv2.drawContours(annotated, [approx], -1, color, 2)

    cv2.imwrite("diagnose_edges.png", edges)
    cv2.imwrite("diagnose_candidates.png", annotated)
    print("\nwrote diagnose_edges.png       -- what Canny saw")
    print("wrote diagnose_candidates.png  -- green = accepted, red = rejected")

    if winner is None:
        print(
            "\nNothing passed. Check diagnose_edges.png:\n"
            "  - is the page border an unbroken white loop?  if not, the paper does\n"
            "    not contrast enough with what it is sitting on\n"
            "  - is the loop there but the shape rejected for too many corners?\n"
            "    something is interrupting the border (a stack of sheets, a pen\n"
            "    lying across it, a corner out of frame)\n"
            "  - 'too small' means the page does not fill enough of the photo"
        )


if __name__ == "__main__":
    main()
