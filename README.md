# Document scanner

A phone photo of a piece of paper in, a flat cropped "scan" out.
Two files, no dependencies beyond OpenCV and NumPy.

```bash
python make_sample.py && python scan.py sample.jpg --debug
```

That writes `sample.jpg` (a fake phone photo), `scanned.jpg` (the result), and a
`debug/` folder holding every intermediate image. **Flip through `debug/` in
order — that is the whole lesson.**

```
python scan.py photo.jpg                      # -> scanned.jpg, black & white
python scan.py photo.jpg -o out.png --mode color
python scan.py photo.jpg --mode gray --debug
```

## The pipeline

| # | Step | Call | Why |
|---|------|------|-----|
| 1 | Shrink to 500px tall | `cv2.resize` | Faster, and small text blurs away so only the page border survives as an edge |
| 2 | Drop color | `cv2.cvtColor` | Edge detectors read one channel; color says nothing about *where* an edge is |
| 3 | Blur | `cv2.GaussianBlur` | Noise changes brightness fast, and so do edges — blur kills the noise first |
| 4 | Find edges | `cv2.Canny` | Turns the photo into thin white lines wherever brightness jumps |
| 5 | Seal gaps | `cv2.dilate` | A contour must be a closed loop; one shadow-gap in the border loses the page |
| 6 | Trace shapes | `cv2.findContours` | Every connected white outline becomes a list of points |
| 7 | Simplify to 4 corners | `cv2.approxPolyDP` | The biggest blob that collapses to 4 points is the page |
| 8 | Solve the warp | `cv2.getPerspectiveTransform` | Four corner-pairs pin down the 3×3 matrix that undoes the perspective |
| 9 | Apply it | `cv2.warpPerspective` | Straightens and crops in one shot, at full resolution |
| 10 | Scan look | `cv2.adaptiveThreshold` | Local cutoff → crisp black ink on white paper despite uneven lighting |

## The five ideas, and where they live

**Color spaces** — [scan.py:61](scan.py:61). OpenCV reads images as **BGR**, not
RGB. Grayscale is not "prettier", it is a requirement: Canny wants a single
intensity channel.

**Blurring** — [scan.py:68](scan.py:68). A Gaussian kernel replaces each pixel
with a distance-weighted average of its neighbours. Noise is random and cancels;
a real edge is wider than the kernel and survives. Skip this line and step 4
returns a snowstorm — try it.

**Thresholding** — [scan.py:180](scan.py:180). A *global* threshold ("darker than
128 = ink") blacks out the shadowed half of a phone photo. Adaptive thresholding
computes a cutoff from each pixel's own 21×21 neighbourhood, so it means "darker
than the paper right around me."

**Contours** — [scan.py:84](scan.py:84). A contour is one connected outline as a
list of points. Two habits matter here: sort by `contourArea` so the biggest
candidate is checked first, and run `approxPolyDP` to collapse a wobbly
several-hundred-point trace into exactly 4 corners.

**Homography** — [scan.py:152](scan.py:152). The star of the show. A 3×3 matrix
mapping the tilted page to a flat rectangle. It is *projective*, not affine — it
can turn a trapezoid back into a rectangle, which is exactly the distortion a
camera introduced. Four point-pairs are the minimum, and we have exactly four.
`order_corners` ([scan.py:110](scan.py:110)) exists only because the matrix maps
corner 1→1, 2→2, …: get the order wrong and your scan comes out mirrored.

## Things to try

- Comment out the `GaussianBlur` line and look at `debug/03_edges.jpg`.
- Change Canny's `75, 200` to `30, 100`. More edges, more false candidates.
- Comment out the `dilate` line, then photograph a white page on a pale desk —
  watch the detection fail because the border has gaps.
- Set `blockSize=101` in `adaptiveThreshold` and see it behave more like a global
  threshold.
- Print `M` in `flatten()`. On `sample.jpg` the bottom row comes out as
  `[0.000149, 0.000018, 1.0]`. An affine transform (rotate/scale/shear) is
  forced to have `[0, 0, 1]` there; those two tiny non-zero numbers *are* the
  perspective. They look negligible, but they divide the result, so across 1200
  pixels they stretch the far edge of the page by around 20%.

## Where it breaks

Honest limits of a 120-line scanner:

- **No page border in frame** → nothing to find. The page must be fully visible
  against a contrasting background.
- **Curved or folded paper** → a homography assumes the page is flat. A creased
  receipt stays crooked.
- **Aspect ratio is an estimate.** We measure the quad's sides in the photo, but
  a tilted page is foreshortened, so the output is a few percent off true Letter
  or A4. Recovering it exactly needs the camera's focal length.
- **Adaptive thresholding amplifies grain.** If `--mode bw` looks speckled, add a
  `cv2.medianBlur(gray, 3)` before the threshold, or use `--mode gray`.
"# OCR" 
