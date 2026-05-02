import logging
import random
import math
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

def create_dramatic_palette(size: int = 256) -> np.ndarray:
    num_wp = random.randint(4, 8)
    positions = sorted({0, size - 1} | {random.randint(1, size - 2) for _ in range(num_wp - 2)})
    hsv = np.empty((len(positions), 3), dtype=float)
    for i in range(len(positions)):
        hsv[i, 0] = random.random()
        hsv[i, 1] = random.choice([random.uniform(0.0, 0.3), random.uniform(0.7, 1.0)])
        hsv[i, 2] = random.choice([random.uniform(0.1, 0.4), random.uniform(0.7, 1.0)])

    rgb = np.empty((size, 3), dtype=float)
    for ch in range(3):
        col_ch = hsv[:, ch]
        exp = random.uniform(0.3, 3.0)
        ramp  = np.linspace(0, 1, len(positions)) ** exp
        rgb[:, ch] = np.interp(np.arange(size), positions, col_ch)

    def twinsine(x):
        a1, f1, p1 = random.uniform(20, 70), random.uniform(0.02, 0.3), random.uniform(0, 2 * math.pi)
        a2, f2, p2 = random.uniform(10, 40), random.uniform(0.05, 0.5), random.uniform(0, 2 * math.pi)
        return a1 * np.sin(f1 * x + p1) + a2 * np.sin(f2 * x + p2)

    for ch in range(3):
        rgb[:, ch] += twinsine(np.arange(size))
        jump_mask = np.random.rand(size) < 0.05
        rgb[jump_mask, ch] *= random.uniform(0.5, 1.5)

    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

def generate_palette_file(folder_path):
    folder = Path(folder_path)
    folder.mkdir(exist_ok=True)
    count = len(list(folder.glob("*.png")))
    fname = folder / f"palette_{count+1:03d}.png"
    pal = create_dramatic_palette()
    img = Image.fromarray(pal.reshape(1, -1, 3).repeat(50, axis=0), mode="RGB")
    img.save(fname)
    logger.info(f"Generated new palette: {fname}")
    return str(fname)

def load_palettes(folder_path):
    palettes = {}
    p_dir = Path(folder_path)
    if not p_dir.exists():
        logger.warning(f"Palette directory '{folder_path}' not found.")
        return palettes
    
    for f in p_dir.iterdir():
        try:
            if f.suffix.lower() in [".png", ".txt", ".map"]:
                if f.suffix.lower() == ".png":
                    img = Image.open(f).convert("RGB")
                    pixels = np.array(img)
                    h, w = pixels.shape[:2]
                    palette = pixels[h//2, :, :] / 255.0
                else:
                    try:
                        palette = np.loadtxt(f, dtype=np.float32)
                    except ValueError:
                        palette = np.loadtxt(f, delimiter=",", dtype=np.float32)
                    if np.max(palette) > 1.0: palette /= 255.0
                
                if palette.ndim == 1:
                    palette = palette.reshape(-1, 3)
                
                palettes[f.stem] = palette
                logger.info(f"Loaded palette: {f.stem}")
        except Exception as e:
            logger.error(f"Failed to load palette {f.name}: {e}")
    return palettes