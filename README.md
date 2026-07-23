# IDPA Target Detection

A C++17 / OpenCV computer vision system that automatically locates and outlines all scoring zones on IDPA practice targets photographed from real-world shooting ranges — handling perspective distortion, rotation, zoom variation, and multiple target color variants (green, red, black, white) without any machine learning or training data.

---

## Table of Contents

- [Features](#features)
- [Target Variants Supported](#target-variants-supported)
- [Pipeline Overview](#pipeline-overview)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Running](#running)
- [Output](#output)
- [Project Structure](#project-structure)
- [Algorithm Deep-Dive](#algorithm-deep-dive)
- [Tuning Parameters](#tuning-parameters)
- [Known Limitations](#known-limitations)

---

## Features

| Feature                      | Detail                                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| **Rotation correction**      | Detects tilt from the longest polygon edge and rotates to upright                             |
| **Auto-zoom**                | Crops and re-centres if the target occupies less than 30% of the frame                        |
| **Perspective correction**   | Iterative homography warp using the B-zone octagon as a geometric reference                   |
| **Multi-color detection**    | Handles green, red, blue, and dark-filled scoring circles via per-channel ratio masking       |
| **Dark target support**      | Smart polarity inversion + flood-fill boundary recovery for black-on-white targets            |
| **Bullet hole robustness**   | Area-increase test removes dent artifacts from polygon vertices without touching real corners |
| **Resolution normalisation** | All output images standardised to 850 × 1550 px for consistent downstream processing          |
| **Scrollable result viewer** | All results stacked in a single OpenCV window with trackbar scroll and keyboard navigation    |
| **Batch processing**         | Reads an entire folder of JPG/PNG images automatically                                        |
| **Per-image result saving**  | Each side-by-side panel saved as `result_NNN.jpg` in the working directory                    |
| **Execution timing**         | Total wall-clock time printed on completion                                                   |

---

## Target Variants Supported

The system handles all common IDPA target colour schemes:

- **Classic white/tan** — dark outlines on light background (most common)
- **Green zone** — green filled scoring circles (Torres Targets style)
- **Red zone** — red filled scoring circles
- **Black silhouette** — dark body with white outline text (inverted polarity)
- **Blue/pink zone** — blue-dominant filled circles

---

## Pipeline Overview

Each image passes through four sequential stages before detection:

```
Raw Image
    │
    ▼
┌─────────────────────────────────────────┐
│  STAGE A — Rotation Correction          │
│  Finds the longest polygon edge,        │
│  computes tilt angle, applies           │
│  affine rotation if > threshold         │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  STAGE B — Auto Zoom                    │
│  If target fills < 30% of frame,        │
│  crops to a 1.3× padded bounding box    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  STAGE C — Perspective Correction       │
│  Iterative loop (max 4 passes):         │
│  1. Detect B-zone octagon               │
│  2. Recover true 4 rectangle corners    │
│     via alternating-parity edge         │
│     selection + line intersection       │
│  3. Expand quad to cover outer torso    │
│  4. Apply perspective warp              │
│  Stops when score stops improving       │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  STAGE D — Resolution Normalisation     │
│  Resize to 850 × 1550 px                │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  DETECTION                              │
│  On each normalised image:              │
│  • Smart polarity inversion             │
│  • Dark circle pre-pass (Otsu)          │
│  • Adaptive threshold + Otsu merge      │
│  • Per-channel colour zone masking      │
│  • Flood-fill outer boundary recovery   │
│  • Contour filtering & deduplication    │
│  • Circle test (ellipse fitting)        │
│  • Octagon recovery (epsilon sweep +    │
│    area-increase noise removal)         │
│  • Outer torso spike removal            │
│  • Shape classification                 │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  PLOTTING                               │
│  Side-by-side panels (raw + detected)   │
│  Saved to result_NNN.jpg                │
│                                         │
└─────────────────────────────────────────┘
```

---

## Prerequisites

### All Platforms

- CMake 3.16 or newer
- C++17-capable compiler
- OpenCV 4.x (core, imgproc, imgcodecs, highgui, calib3d)

### Windows (Visual Studio)

1. Download the pre-built OpenCV Windows installer from https://opencv.org/releases/
2. Extract to e.g. `C:\opencv`
3. Add `C:\opencv\build\x64\vc16\bin` (or your vc version) to your system `PATH`

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install cmake build-essential libopencv-dev
```

### macOS (Homebrew)

```bash
brew install cmake opencv
```

---

## Building

### Linux / macOS

```bash
# Clone or copy the project files into a directory
cd idpa_target_detection

# Create a build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -- -j$(nproc)
```

### Windows (Visual Studio command prompt)

```cmd
cd idpa_target_detection
mkdir build && cd build

cmake .. -DOpenCV_DIR="C:\opencv\build" -G "Visual Studio 17 2022" -A x64

cmake --build . --config Release
```

The compiled binary will be at:

- Linux/macOS: `build/idpa_detect`
- Windows: `build\Release\idpa_detect.exe`

---

## Running

### Default (uses hardcoded folder path in source)

```bash
./idpa_detect
```

### Custom folder path (recommended)

Pass the path to your images folder as the first argument:

```bash
# Linux / macOS
./idpa_detect /path/to/your/target/images

# Windows
idpa_detect.exe "C:\Users\YourName\Pictures\range_session"
```

The program scans the folder for all `.jpg`, `.jpeg`, and `.png` files (case-insensitive) and processes them in filesystem order.

### Keyboard controls in the result viewer

| Key          | Action                    |
| ------------ | ------------------------- |
| `W` or `↑`   | Scroll up (quarter row)   |
| `S` or `↓`   | Scroll down (quarter row) |
| `Page Up`    | Scroll up one viewport    |
| `Page Down`  | Scroll down one viewport  |
| `F`          | Jump to first image       |
| `L`          | Jump to last image        |
| `Q` or `Esc` | Quit                      |
| Trackbar     | Drag to any position      |

---

## Output

### Console output

For each image, the console prints a detailed processing log:

```
Processing image #1: /path/to/image.jpg
  Deskew: rotation=2.5° → rotated | fill=0.62 → no zoom needed
--- Deskewing Loop Started ---
Pass 1: Score = 14.231 → perspective pass applied (scale=1.83)
Pass 2: Score = 3.104 → perspective pass applied (scale=1.71)
Pass 3: Score = 1.022 -> Improvement plateaued (diff = 0.08). Stopping.
--- Deskewing Finished | Total successful passes: 2 ---
 → resolution normalized to 850×1550

Image #1: 5 shapes detected → saved result_001.jpg
Total execution time: 12.34 seconds
```

### Result files

`result_001.jpg`, `result_002.jpg`, … are written to the **current working directory** (wherever you ran the binary from). Each file contains a side-by-side panel:

- **Left panel** — original raw image scaled to match detection height
- **Right panel** — deskewed image with all detected zone contours drawn in red

### Detected shapes per image

A correctly processed IDPA target typically yields **5 shapes**:

| Shape                            | Zone             |
| -------------------------------- | ---------------- |
| Outer polygon (torso silhouette) | -3 zone boundary |
| Polygon 8 pts (B-zone)           | -1 zone boundary |
| Circle / ellipse (large)         | -0 centre circle |
| Rectangle (head box)             | Head zone        |
| Circle / ellipse (small)         | Head circle      |

---

## Project Structure

```
idpa_target_detection/
├── cpptest.cpp          ← Full source (single-file project)
├── CMakeLists.txt       ← Build configuration
├── README.md            ← This file
└── build/               ← Created by cmake (not committed)
    └── idpa_detect      ← Compiled binary
```

The source is intentionally a single file — all three major components (detection, deskewing, plotting) are self-contained functions that share only the global math helpers and type aliases at the top.

---

## Algorithm Deep-Dive

### Polarity inversion

The centre 50% of the image is sampled to compute mean brightness. If the mean is below 100 (dark target body), the grayscale is bit-inverted before thresholding, turning a black silhouette into a white one so `THRESH_BINARY_INV` works identically on all target colours. This only fires during the final detection pass — deskewing always uses the original polarity to avoid breaking polygon finding.

### Dual-threshold merge

Two thresholds are computed and OR'd together:

- **Adaptive Gaussian** — catches thin dark outlines on varying backgrounds
- **Otsu global** — catches solid dark-filled regions that adaptive misses (black scoring circles)

### Colour zone masking

Each colour channel (B, G, R) is blurred and compared pairwise. Pixels where `R > B` AND `R > G` by at least 40 units are labelled red; `G > R` by 30 is green; `B > G` AND `B > R` by 30 is blue. A brightness floor prevents dark noise from being labelled as colour. The masks are morphologically closed and filtered for solid blobs (solidity > 0.85, fill ratio > 0.35) before being fused with the structural threshold.

### Octagon corner recovery (deskew)

The B-zone octagon is a corner-cut rectangle — its 8 edges alternate between long (real sides) and short (diagonal corner cuts). The 4 real sides are selected by the alternating-parity sum: whichever of the even-indexed or odd-indexed edge groups sums longer is the real sides group. Those 4 edges are classified as top/bottom/left/right by midpoint position, fitted as infinite lines, and intersected in adjacent pairs to find the true TL, TR, BR, BL corners of the underlying rectangle. This is immune to which angle the target is viewed from.

### Perspective warp expansion

After corner recovery, the tight 4-point quad is expanded outward from its own centroid via homothety (uniform scaling) until all points of the outer torso silhouette fit inside it. This preserves edge directions exactly — the expanded quad has identically-angled sides, just pushed further out. A 3% padding margin and a 2.5× maximum scale cap prevent runaway expansion from background outliers.

### Bullet hole noise removal (octagon detection)

During final shape detection, the B-zone octagon contour is approximated by sweeping epsilon values from 0.005 to 0.04 of hull perimeter and selecting whichever result has a vertex count closest to 8. If the count is still above 8, a loop removes the most removable noise vertex — defined as one where both adjacent edges are shorter than 12% of perimeter AND removing it does not decrease the polygon area. A real octagon corner always decreases area when removed (it defines the convex extent); a bullet hole dent always increases area (it fills in a concavity). This is the key discriminator.

---

## Tuning Parameters

All key thresholds are constants inside the code with comments. The most commonly adjusted ones:

| Location      | Variable                         | Default      | Effect                                  |
| ------------- | -------------------------------- | ------------ | --------------------------------------- |
| `deskewing()` | `angle_threshold`                | `10.0°`      | Minimum rotation to trigger Stage A     |
| `deskewing()` | `area_ratio < 0.3`               | `0.3`        | Fill ratio below which Stage B zooms    |
| `deskewing()` | `MAX_ITERS`                      | `4`          | Maximum perspective passes              |
| `deskewing()` | `SCORE_IMPROVE_MIN`              | `0.1`        | Minimum score improvement to continue   |
| `detection()` | `MIN_AREA = IMG_AREA * 0.002`    | `0.2%`       | Smallest contour kept                   |
| `detection()` | `solidity < 0.45`                | `0.45`       | Minimum shape solidity                  |
| `detection()` | `TOLERANCE_PX = IMG_DIAG * 0.03` | `3%`         | Deduplication distance                  |
| `detection()` | `red threshold`                  | `40`         | Channel difference to classify as red   |
| `detection()` | `green threshold`                | `30`         | Channel difference to classify as green |
| `plotting()`  | `TARGET_W / TARGET_H`            | `850 / 1550` | Output resolution                       |
| `plotting()`  | `DISPLAY_W`                      | `900`        | Display panel width in viewer           |
| `plotting()`  | `MAX_ROW_H`                      | `500`        | Max panel height in mosaic              |

---

## Known Limitations

- **Single target per image** — the pipeline assumes one IDPA target is the primary subject. Multiple targets in the same frame are not individually segmented.
- **Extreme angles** — perspective correction works reliably up to roughly 35-40° of viewing angle. Beyond that, the octagon may not be detectable by the alternating-parity method.
- **Heavily obscured targets** — if more than ~40% of the B-zone octagon border is covered (by tape patches, heavy bloodstain markers, or overlapping paper), corner recovery may fail and deskewing is skipped.
- **Non-standard colours** — targets with yellow or orange scoring zones are not currently in the colour mask. Adding a new colour requires one additional `subtract + threshold + bitwise_and` block in the colour zone section.
- **Output path** — result images are always written to the current working directory, not the input folder.
