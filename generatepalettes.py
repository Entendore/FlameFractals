#!/usr/bin/env python3
"""
generate_palettes.py  –  high-speed dramatic palette generator
-----------------------------------------------------------
pip install numpy pillow tqdm
"""
import argparse
import colorsys
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------------ constants
DEFAULT_COUNT = 150
DEFAULT_SIZE  = 256          # colours per palette
OUT_DIR       = Path("palettes")

# ------------------------------------------------------------------ generation
def dramatic_palette(size: int = 256) -> np.ndarray:
    """Return size×3 uint8 array with extreme colour transitions."""
    # 1. way-points
    num_wp = random.randint(4, 8)
    positions = sorted({0, size - 1} | {random.randint(1, size - 2) for _ in range(num_wp - 2)})
    hsv = np.empty((len(positions), 3), dtype=float)
    for i in range(len(positions)):
        hsv[i, 0] = random.random()
        hsv[i, 1] = random.choice([random.uniform(0.0, 0.3), random.uniform(0.7, 1.0)])
        hsv[i, 2] = random.choice([random.uniform(0.1, 0.4), random.uniform(0.7, 1.0)])

    # 2. non-linear interpolation along the whole gradient
    rgb = np.empty((size, 3), dtype=float)
    for ch in range(3):
        col_ch = hsv[:, ch]
        # power curve for non-linear ramp
        exp = random.uniform(0.3, 3.0)
        ramp  = np.linspace(0, 1, len(positions)) ** exp
        # cubic spline would be nicer, but this is fast and good enough
        rgb[:, ch] = np.interp(np.arange(size), positions, col_ch)

    # 3. dual sine distortion
    def twinsine(x):
        a1, f1, p1 = random.uniform(20, 70), random.uniform(0.02, 0.3), random.uniform(0, 2 * math.pi)
        a2, f2, p2 = random.uniform(10, 40), random.uniform(0.05, 0.5), random.uniform(0, 2 * math.pi)
        return a1 * np.sin(f1 * x + p1) + a2 * np.sin(f2 * x + p2)

    for ch in range(3):
        rgb[:, ch] += twinsine(np.arange(size))
        # 4. random jumps
        jump_mask = np.random.rand(size) < 0.05
        rgb[jump_mask, ch] *= random.uniform(0.5, 1.5)

    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

# ------------------------------------------------------------------ exporters
def save_png(pal: np.ndarray, file: Path, height: int = 50):
    """256×1 PNG row visualisation."""
    img = Image.fromarray(pal.reshape(1, -1, 3).repeat(height, axis=0), mode="RGB")
    img.save(file)

def save_map(pal: np.ndarray, file: Path):
    """Classic .map text format (R G B lines)."""
    np.savetxt(file, pal, fmt="%3d")

# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="Generate dramatic flame-fractal palettes")
    parser.add_argument("-n", "--number", type=int, default=DEFAULT_COUNT, help="how many palettes")
    parser.add_argument("-p", "--palette-size", type=int, default=DEFAULT_SIZE, help="colours per palette")
    parser.add_argument("-o", "--out-dir", type=Path, default=OUT_DIR, help="output folder")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    print(f"Generating {args.number} palettes …")
    for idx in tqdm(range(1, args.number + 1), ncols=80):
        pal = dramatic_palette(args.palette_size)
        base = args.out_dir / f"palette_{idx:03d}"
        save_png(pal, base.with_suffix(".png"))
        save_map(pal, base.with_suffix(".map"))

    print("\nDone – palettes saved to", args.out_dir.resolve())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)