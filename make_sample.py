"""
make_sample.py -- fake a "phone photo of a piece of paper" so you have something
to test scan.py with straight away.

It builds a clean page, then does the OPPOSITE of what scan.py does: it warps
that flat page onto a tilted quadrilateral over a desk-ish background, and adds
blur and noise. Reading this file is a decent way to convince yourself that a
homography really is reversible.

    python make_sample.py
"""

import os
import sys

import cv2
import numpy as np

PAGE_W, PAGE_H = 850, 1100      # a flat page, roughly US Letter proportions
PHOTO_W, PHOTO_H = 1200, 1600   # the "photo" we output


def build_page():
    """A white page with a title and some grey bars standing in for text."""
    page = np.full((PAGE_H, PAGE_W, 3), 250, dtype=np.uint8)

    cv2.putText(page, "QUARTERLY REPORT", (70, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.line(page, (70, 165), (PAGE_W - 70, 165), (30, 30, 30), 2)

    # Paragraphs: rows of grey bars, with a short bar ending each paragraph.
    rng = np.random.default_rng(7)
    y = 230
    for paragraph in range(6):
        if y > PAGE_H - 320:   # leave room for the figure box below
            break
        for line in range(rng.integers(3, 7)):
            width = int(rng.integers(PAGE_W // 2, PAGE_W - 140))
            cv2.rectangle(page, (70, y), (70 + width, y + 12), (90, 90, 90), -1)
            y += 34
        y += 30

    cv2.rectangle(page, (70, y), (PAGE_W - 70, y + 180), (150, 150, 150), 2)
    cv2.putText(page, "figure 1", (90, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2, cv2.LINE_AA)
    return page


def build_background():
    """A noisy brown surface, so the white page has something to contrast with."""
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 12, (PHOTO_H, PHOTO_W, 1))
    desk = np.full((PHOTO_H, PHOTO_W, 3), (60, 85, 115), dtype=np.float64)  # BGR
    desk += noise
    return np.clip(desk, 0, 255).astype(np.uint8)


def main():
    page = build_page()
    photo = build_background()

    # The four corners the page will occupy in the photo: a tilted, tapered
    # quad, the way paper looks when the camera is not square-on to it.
    source = np.array([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]],
                      dtype="float32")
    destination = np.array([[250, 190], [1010, 330], [880, 1440], [130, 1190]],
                           dtype="float32")
    M = cv2.getPerspectiveTransform(source, destination)

    warped_page = cv2.warpPerspective(page, M, (PHOTO_W, PHOTO_H))
    # Warp a solid white rectangle the same way to get a mask of "where the
    # page ended up", so we paste only the page and not the black corners.
    mask = cv2.warpPerspective(np.full((PAGE_H, PAGE_W), 255, np.uint8),
                               M, (PHOTO_W, PHOTO_H))
    photo[mask > 0] = warped_page[mask > 0]

    # Sell the illusion: a soft shadow down one side, a little lens blur, grain.
    shade = np.tile(np.linspace(1.0, 0.75, PHOTO_W), (PHOTO_H, 1))[:, :, None]
    photo = np.clip(photo * shade, 0, 255).astype(np.uint8)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    grain = np.random.default_rng(11).normal(0, 4, photo.shape)
    photo = np.clip(photo.astype(np.float64) + grain, 0, 255).astype(np.uint8)

    # Default to sample.jpg, but never clobber a file that is already there --
    # you probably put your own test photo in it.
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    if os.path.exists(out):
        sys.exit(
            "{} already exists, refusing to overwrite it.\n"
            "Pass another name:  python make_sample.py generated.jpg".format(out)
        )

    cv2.imwrite(out, photo)
    print("Wrote {} ({}x{})".format(out, PHOTO_W, PHOTO_H))


if __name__ == "__main__":
    main()
