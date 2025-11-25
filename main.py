#!/usr/bin/env python3
# flame_master.py – single-file, high-performance flame-fractal generator
# --------------------------------------------------------------------
#  pip install numpy pillow imageio numba tqdm (tqdm optional but nice)
# --------------------------------------------------------------------
import math, os, random, string, sys, time
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageFilter
from numba import njit, prange

# ------------------------------------------------------------------ CONFIG
class Config:
    def __init__(
        self,
        width=800,
        height=800,
        iterations=3_000_000,
        transforms=15,
        zoom=1.2,
        skip=100,
        gamma=2.2,
        layers=3,
        frames=0,               # 0 = single image, >0 = GIF
        symmetry=None,          # None, "x", "y", "kaleidoscope"
        symmetry_segments=6,
        palette=None,           # N×3 numpy array [0-1]
        use_palette=True,
        art_style=None,         # None, "pointillism", "expressionist", "oil", "watercolor"
        seed=None,
        parallel=True,
    ):
        self.__dict__.update(locals())
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

# ------------------------------------------------------------------ VARIATIONS
@njit
def var_linear(x, y): return x, y

@njit
def var_sinusoidal(x, y): return math.sin(x), math.sin(y)

@njit
def var_spherical(x, y):
    r2 = x * x + y * y + 1e-6
    return x / r2, y / r2

@njit
def var_swirl(x, y):
    r2 = x * x + y * y
    return x * math.sin(r2) - y * math.cos(r2), x * math.cos(r2) + y * math.sin(r2)

@njit
def var_horseshoe(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    return (x - y) * (x + y) / r, 2 * x * y / r

@njit
def var_polar(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    return theta / math.pi, r - 1

@njit
def var_handkerchief(x, y):
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r * math.sin(theta + r), r * math.cos(theta - r)

@njit
def var_heart(x, y):
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r * math.sin(theta * r), -r * math.cos(theta * r)

@njit
def var_disk(x, y):
    r = math.sqrt(x * x + y * y) / math.pi
    theta = math.atan2(y, x)
    return theta / math.pi * math.sin(math.pi * r), theta / math.pi * math.cos(math.pi * r)

@njit
def var_blur(x, y):
    angle = random.uniform(0, 2 * math.pi)
    rad = random.uniform(0, 1)
    return x + rad * math.cos(angle), y + rad * math.sin(angle)

@njit
def var_fisheye(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    return x / r, y / r

@njit
def var_julia(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    return r * math.cos(theta / 2), r * math.sin(theta / 2)

@njit
def var_popcorn(x, y):
    return x + 0.1 * math.sin(math.tan(3 * y)), y + 0.1 * math.sin(math.tan(3 * x))

@njit
def var_bent(x, y):
    if x >= 0 and y >= 0:
        return x, y
    if x < 0 and y >= 0:
        return 2 * x, y
    if x >= 0 and y < 0:
        return x, y / 2
    return 2 * x, y / 2

@njit
def var_waves(x, y):
    return x + 0.1 * math.sin(y), y + 0.1 * math.sin(x)

@njit
def var_exponential(x, y):
    r = math.exp(x) - 1
    theta = math.pi * y
    return r * math.cos(theta), r * math.sin(theta)

VAR_FUNCS = [
    var_linear,
    var_sinusoidal,
    var_spherical,
    var_swirl,
    var_horseshoe,
    var_polar,
    var_handkerchief,
    var_heart,
    var_disk,
    var_blur,
    var_fisheye,
    var_julia,
    var_popcorn,
    var_bent,
    var_waves,
    var_exponential,
]

# ------------------------------------------------------------------ PALETTE LOADER
def load_palettes(folder: Path):
    pals = []
    if not folder.exists():
        return pals
    for f in folder.iterdir():
        try:
            if f.suffix.lower() == ".png":
                img = Image.open(f).convert("RGB")
                pals.append(np.array(img)[0, :, :] / 255.0)
            elif f.suffix.lower() == ".txt":
                pals.append(
                    np.loadtxt(f, delimiter=",", dtype=np.float32) / 255.0
                )
        except Exception as e:
            print("Palette skip:", f, e)
    return pals

# ------------------------------------------------------------------ CORE NUMBA ENGINE
@njit()
def iterate_fractal(transforms, iterations, width, height, zoom, skip):
    density = np.zeros((height, width), dtype=np.float32)
    color_acc = np.zeros((height, width, 3), dtype=np.float32)
    x, y = 0.0, 0.0
    n_trans = len(transforms)

    for i in prange(iterations):
        t = transforms[np.random.randint(0, n_trans)]
        a, b, c, d, e, f = t[0:6]
        weight = t[6]
        var_idx = int(t[7])
        col = t[8:11]

        x1 = a * x + b * y + c
        y1 = d * x + e * y + f

        # apply variation
        vx, vy = 0, 0
        if var_idx == 0:
            vx, vy = var_linear(x1, y1)
        elif var_idx == 1:
            vx, vy = var_sinusoidal(x1, y1)
        elif var_idx == 2:
            vx, vy = var_spherical(x1, y1)
        elif var_idx == 3:
            vx, vy = var_swirl(x1, y1)
        elif var_idx == 4:
            vx, vy = var_horseshoe(x1, y1)
        elif var_idx == 5:
            vx, vy = var_polar(x1, y1)
        elif var_idx == 6:
            vx, vy = var_handkerchief(x1, y1)
        elif var_idx == 7:
            vx, vy = var_heart(x1, y1)
        elif var_idx == 8:
            vx, vy = var_disk(x1, y1)
        elif var_idx == 9:
            vx, vy = var_blur(x1, y1)
        elif var_idx == 10:
            vx, vy = var_fisheye(x1, y1)
        elif var_idx == 11:
            vx, vy = var_julia(x1, y1)
        elif var_idx == 12:
            vx, vy = var_popcorn(x1, y1)
        elif var_idx == 13:
            vx, vy = var_bent(x1, y1)
        elif var_idx == 14:
            vx, vy = var_waves(x1, y1)
        else:  # 15
            vx, vy = var_exponential(x1, y1)

        x, y = vx * weight, vy * weight

        ix = int((x * zoom + 1) * width / 2)
        iy = int((y * zoom + 1) * height / 2)

        if 0 <= ix < width and 0 <= iy < height and i > skip:
            density[iy, ix] += 1
            color_acc[iy, ix, 0] += col[0]
            color_acc[iy, ix, 1] += col[1]
            color_acc[iy, ix, 2] += col[2]

    # normalize
    max_d = np.max(density)
    safe_d = np.log(density + 1) / np.log(max_d + 1 + 1e-6)
    norm_c = np.where(
        density[..., None] > 0,
        color_acc / (density[..., None] + 1e-6),
        0,
    )
    return safe_d, np.clip(norm_c, 0, 1)

# ------------------------------------------------------------------ FRACTAL CLASS
class FlameFractal:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def rand_color(self):
        if self.cfg.use_palette and self.cfg.palette is not None:
            return random.choice(self.cfg.palette)
        return np.random.rand(3)

    def build_transforms(self):
        ts = []
        for _ in range(self.cfg.transforms):
            a, b, c = np.random.uniform(-1, 1, 3)
            d, e, f = np.random.uniform(-1, 1, 3)
            weight = np.random.uniform(0.5, 1.5)
            var_idx = np.random.randint(0, len(VAR_FUNCS))
            col = self.rand_color()
            ts.append((a, b, c, d, e, f, weight, var_idx, *col))
        return np.array(ts, dtype=np.float32)

    def apply_symmetry(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.symmetry == "x":
            return np.hstack([img, np.fliplr(img)])
        if self.cfg.symmetry == "y":
            return np.vstack([img, np.flipud(img)])
        if self.cfg.symmetry == "kaleidoscope":
            segs = self.cfg.symmetry_segments
            h, w = img.shape[:2]
            y, x = np.ogrid[:h, :w]
            cx, cy = w // 2, h // 2
            theta = np.arctan2(y - cy, x - cx)
            theta_mod = np.mod(theta, 2 * math.pi / segs)
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            x_new = (r * np.cos(theta_mod) + 1) * w / 2
            y_new = (r * np.sin(theta_mod) + 1) * h / 2
            x_new = np.clip(x_new.astype(int), 0, w - 1)
            y_new = np.clip(y_new.astype(int), 0, h - 1)
            out = np.empty_like(img)
            out[...] = img[y_new, x_new]
            return out
        return img

    def art_filter(self, img: np.ndarray) -> np.ndarray:
        style = self.cfg.art_style
        if style is None:
            return img
        pil = Image.fromarray(img)
        if style == "pointillism":
            dots = Image.new("RGB", pil.size)
            for _ in range(50_000):
                x = random.randint(0, pil.width - 1)
                y = random.randint(0, pil.height - 1)
                dots.putpixel((x, y), tuple(img[y, x]))
            pil = dots
        elif style == "expressionist":
            pil = pil.filter(ImageFilter.GaussianBlur(2))
            img = np.array(pil, float) * 1.5
            return np.clip(img, 0, 255).astype(np.uint8)
        elif style == "oil":
            pil = pil.filter(ImageFilter.ModeFilter(5)).filter(ImageFilter.EDGE_ENHANCE)
        elif style == "watercolor":
            pil = pil.filter(ImageFilter.GaussianBlur(3))
            img = np.array(pil, float) * 0.8
            return np.clip(img, 0, 255).astype(np.uint8)
        return np.array(pil)

    def render_single(self) -> np.ndarray:
        combined = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.float32)
        for _ in range(self.cfg.layers):
            transforms = self.build_transforms()
            safe_d, norm_c = iterate_fractal(
                transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
                self.cfg.zoom, self.cfg.skip
            )
            rgb = (norm_c ** (1 / self.cfg.gamma)) * safe_d[..., None] * 255
            combined += np.clip(rgb, 0, 255)
        combined = (combined / self.cfg.layers).astype(np.uint8)
        combined = self.apply_symmetry(combined)
        combined = self.art_filter(combined)
        return combined

    def render_animation(self, filename: str):
        frames = []
        transforms = self.build_transforms()
        for f in range(self.cfg.frames):
            # jitter weight & color drift
            for t in transforms:
                t[6] *= random.uniform(0.95, 1.05)
                t[8:11] = t[8:11] * 0.9 + self.rand_color() * 0.1
            safe_d, norm_c = iterate_fractal(
                transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
                self.cfg.zoom, self.cfg.skip
            )
            rgb = (norm_c ** (1 / self.cfg.gamma)) * safe_d[..., None] * 255
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = self.apply_symmetry(rgb)
            rgb = self.art_filter(rgb)
            frames.append(rgb)
            print(f"Frame {f+1}/{self.cfg.frames}")
        imageio.mimsave(filename, frames, duration=0.1)
        print("Saved animation:", filename)

# ------------------------------------------------------------------ MAIN / CLI
def main():
    import argparse, textwrap

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""\
        High-performance flame-fractal generator
        ----------------------------------------
        Put palettes in ./palettes/  (PNG top-row  or  R,G,B txt files)
        """)
    )
    ap.add_argument("-W", "--width", type=int, default=800, help="output width")
    ap.add_argument("-H", "--height", type=int, default=800, help="output height")
    ap.add_argument("-i", "--iterations", type=int, default=3_000_000)
    ap.add_argument("-t", "--transforms", type=int, default=15)
    ap.add_argument("-z", "--zoom", type=float, default=1.2)
    ap.add_argument("-s", "--skip", type=int, default=100)
    ap.add_argument("-g", "--gamma", type=float, default=2.2)
    ap.add_argument("-l", "--layers", type=int, default=3)
    ap.add_argument("-f", "--frames", type=int, default=0, help="0=image, >0=GIF")
    ap.add_argument("--symmetry", choices=[None, "x", "y", "kaleidoscope"], default=None)
    ap.add_argument("--segments", type=int, default=6, help="kaleido segments")
    ap.add_argument("--style", choices=[None, "pointillism", "expressionist", "oil", "watercolor"], default=None)
    ap.add_argument("--palette", action="store_true", help="use palette folder")
    ap.add_argument("-o", "--output", default=None, help="file name (auto if none)")
    args = ap.parse_args()

    palettes = load_palettes(Path("palettes")) if args.palette else []
    if args.palette and not palettes:
        print("No palettes found – falling back to random colors")

    cfg = Config(
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        transforms=args.transforms,
        zoom=args.zoom,
        skip=args.skip,
        gamma=args.gamma,
        layers=args.layers,
        frames=args.frames,
        symmetry=args.symmetry,
        symmetry_segments=args.segments,
        palette=random.choice(palettes) if palettes else None,
        use_palette=bool(palettes),
        art_style=args.style,
    )

    fractal = FlameFractal(cfg)
    if cfg.frames:
        out = args.output or f"flame_{int(time.time())}.gif"
        fractal.render_animation(out)
    else:
        out = args.output or f"flame_{int(time.time())}.png"
        img = fractal.render_single()
        Image.fromarray(img).save(out)
        print("Saved", out)

if __name__ == "__main__":
    main()