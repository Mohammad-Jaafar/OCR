import argparse
import os
import sys

import cv2
import numpy as np

WORK_HEIGHT = 500

DEBUG_DIR = "debug"
_debug_on = False
_debug_step = 0


def debug_save(name, image):
    global _debug_step
    if not _debug_on:
        return
    _debug_step += 1
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, "{:02d}_{}.jpg".format(_debug_step, name))
    cv2.imwrite(path, image)
    print("  [debug] " + path)


# ---------------------------------------------------------------------------
# Step 1 -- find the page
# ---------------------------------------------------------------------------


def find_page(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_save("gray", gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    debug_save("blurred", blurred)

    edges = cv2.Canny(blurred, 75, 200)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    debug_save("edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_area = image.shape[0] * image.shape[1]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 0.10 * image_area:
            return approx.reshape(4, 2).astype("float32")

    return None


# ---------------------------------------------------------------------------
# Step 2 -- flatten it (the homography)
# ---------------------------------------------------------------------------


def order_corners(points):
    rect = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)  # x + y
    rect[0] = points[np.argmin(total)]  # top-left
    rect[2] = points[np.argmax(total)]  # bottom-right
    diff = np.diff(points, axis=1)  # y - x
    rect[1] = points[np.argmin(diff)]  # top-right
    rect[3] = points[np.argmax(diff)]  # bottom-left
    return rect


def flatten(image, corners):
    """Warp the quadrilateral `corners` out of `image` into a straight rectangle."""
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, M, (width, height))

    m = 4
    return warped[m:-m, m:-m]


# ---------------------------------------------------------------------------
# Step 3 -- make it look like a scan
# ---------------------------------------------------------------------------


def enhance(warped, mode):
    """color = leave it alone, gray = desaturate, bw = crisp black on white."""
    if mode == "color":
        return warped

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return gray

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Flatten a photo of a document.")
    parser.add_argument("image", help="input photo")
    parser.add_argument("-o", "--output", default="scanned.jpg", help="output file")
    parser.add_argument(
        "--mode",
        choices=["color", "gray", "bw"],
        default="bw",
        help="look of the final scan (default: bw)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="save each intermediate step to debug/"
    )
    args = parser.parse_args()

    global _debug_on
    _debug_on = args.debug

    image = cv2.imread(args.image)
    if image is None:
        sys.exit("Could not read " + args.image)
    print("Loaded {} ({}x{})".format(args.image, image.shape[1], image.shape[0]))

    # Only ever shrink: max(..., 1.0) stops a small photo being blown UP to 500,
    # which would invent pixels and blur the very border we are looking for.
    ratio = max(image.shape[0] / WORK_HEIGHT, 1.0)
    small = cv2.resize(
        image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio))
    )

    corners = find_page(small)
    if corners is None:
        sys.exit(
            "No 4-sided page found. Try --debug and look at debug/03_edges.jpg:\n"
            "the page outline needs to be an unbroken loop. More contrast\n"
            "between the paper and the background usually fixes it."
        )

    if _debug_on:
        outlined = small.copy()
        cv2.drawContours(outlined, [corners.astype(int)], -1, (0, 255, 0), 2)
        debug_save("page_found", outlined)

    corners *= ratio
    warped = flatten(image, corners)
    debug_save("warped", warped)

    result = enhance(warped, args.mode)
    debug_save("final", result)

    cv2.imwrite(args.output, result)
    print(
        "Wrote {} ({}x{}, mode={})".format(
            args.output, result.shape[1], result.shape[0], args.mode
        )
    )


if __name__ == "__main__":
    main()
