"""Measure flat-surface noise and temporal stability in Cornell-box captures.

Run with the bundled Python runtime (Pillow and NumPy), after --capture-quality.
This is a regression metric, not a path-traced accuracy or convergence test.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

REGIONS = {"front": (230, 307, 262, 340), "back_wall": (180, 135, 328, 250)}


def read(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    assert image.shape == (480, 640, 3), "metrics require 640x480 captures"
    return image


def measure(image):
    result = {}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        rgb = image[y0:y1, x0:x1]
        # Remove the smooth illumination gradient; measure broad blotches as
        # well as fine noise. Adjacent-pixel differences miss blurred artifacts.
        y, x = np.mgrid[-1:1:complex(y1 - y0), -1:1:complex(x1 - x0)]
        basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1).reshape(-1, 6)
        values = rgb.reshape(-1, 3)
        residual = values - basis @ np.linalg.lstsq(basis, values, rcond=None)[0]
        result[name] = {"mean": float(rgb.mean()), "residual_rms": float(np.sqrt(np.mean(residual ** 2)))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--sequence", required=True, help="PNG filename glob")
    parser.add_argument("--output", default="tmp/quality-metrics.json")
    args = parser.parse_args()
    before, after = measure(read(args.before)), measure(read(args.after))
    paths = sorted(glob.glob(args.sequence))
    assert len(paths) >= 8, "capture at least eight frames"
    frames = [read(path) for path in paths]
    report = {"before": before, "after": after, "frames": len(frames), "temporal_mae": {}}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        changes = [float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean()) for a, b in zip(frames, frames[1:])]
        report["temporal_mae"][name] = {"mean": float(np.mean(changes)), "maximum": max(changes)}
        # Preserve useful lighting: reducing noise by making everything black
        # must fail, as must severe frame-to-frame flicker on a static surface.
        assert after[name]["mean"] > 8, f"{name} became dark"
        assert after[name]["residual_rms"] < before[name]["residual_rms"], f"{name} noise did not improve"
        assert max(changes) < 2.0, f"{name} flickers by more than 2/255"
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
