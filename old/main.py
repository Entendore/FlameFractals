#!/usr/bin/env python3
"""
main.py – single-file, high-performance flame-fractal generator
--------------------------------------------------------------------
pip install numpy pillow imageio numba tqdm (tqdm optional but nice)
--------------------------------------------------------------------
"""
import math
import os
import random
import sys
import time
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from numba import njit, prange

__version__ = "1.0.0"

# ------------------------------------------------------------------ CONFIG
class Config:
    """
    Configuration class for the flame fractal generator.
    Holds all parameters and validates them upon initialization.
    """
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
        background_color=(0, 0, 0),  # Custom background color
        vibrancy=1.0,           # Color vibrancy control (0.0-1.0)
        final_transform=None,   # Final affine transform coefficients (a,b,c,d,e,f)
    ):
        # Store all parameters
        self.width = width
        self.height = height
        self.iterations = iterations
        self.transforms = transforms
        self.zoom = zoom
        self.skip = skip
        self.gamma = gamma
        self.layers = layers
        self.frames = frames
        self.symmetry = symmetry
        self.symmetry_segments = symmetry_segments
        self.palette = palette
        self.use_palette = use_palette
        self.art_style = art_style
        self.seed = seed
        self.background_color = background_color
        self.vibrancy = vibrancy
        self.final_transform = final_transform

        # Initialize random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # --- Parameter Validation ---
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        if iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")
        if transforms <= 0:
            raise ValueError("Number of transforms must be a positive integer.")
        if zoom <= 0:
            raise ValueError("Zoom must be a positive number.")
        if gamma <= 0:
            raise ValueError("Gamma must be a positive number.")
        if layers <= 0:
            raise ValueError("Number of layers must be a positive integer.")
        if frames < 0:
            raise ValueError("Number of frames must be a non-negative integer.")
        if symmetry not in (None, "x", "y", "kaleidoscope"):
            raise ValueError("Invalid symmetry type. Must be None, 'x', 'y', or 'kaleidoscope'.")
        if symmetry == "kaleidoscope" and (symmetry_segments <= 0 or symmetry_segments > 360):
            raise ValueError("Symmetry segments must be between 1 and 360 for kaleidoscope symmetry.")
        if art_style not in (None, "pointillism", "expressionist", "oil", "watercolor"):
            raise ValueError("Invalid art style. Must be None or one of: pointillism, expressionist, oil, watercolor")
        if not (0.0 <= vibrancy <= 1.0):
            raise ValueError("Vibrancy must be between 0.0 and 1.0")

        # Validate background color
        if not isinstance(background_color, (tuple, list)) or len(background_color) != 3:
            raise ValueError("Background color must be a tuple or list of three integers (R, G, B).")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in background_color):
            raise ValueError("Background color components must be integers between 0 and 255.")
        self.background_color = tuple(int(c) for c in background_color)

        # Validate final transform
        if final_transform is not None:
            if not isinstance(final_transform, (tuple, list)) or len(final_transform) != 6:
                raise ValueError("Final transform must be a tuple or list of six numbers (a, b, c, d, e, f).")
            try:
                self.final_transform = tuple(float(x) for x in final_transform)
            except (TypeError, ValueError):
                raise ValueError("Final transform values must be convertible to floats.")

    def to_dict(self):
        """Convert configuration to a dictionary for saving to JSON."""
        config_dict = {
            'width': self.width,
            'height': self.height,
            'iterations': self.iterations,
            'transforms': self.transforms,
            'zoom': self.zoom,
            'skip': self.skip,
            'gamma': self.gamma,
            'layers': self.layers,
            'frames': self.frames,
            'symmetry': self.symmetry,
            'symmetry_segments': self.symmetry_segments,
            'use_palette': self.use_palette,
            'art_style': self.art_style,
            'seed': self.seed,
            'background_color': self.background_color,
            'vibrancy': self.vibrancy,
            'final_transform': self.final_transform
        }
        
        # Handle palette separately since it's a numpy array
        if self.palette is not None:
            config_dict['palette'] = self.palette.tolist()
        else:
            config_dict['palette'] = None
            
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """Create a Config instance from a dictionary loaded from JSON."""
        # Convert list back to numpy array if palette exists
        if 'palette' in config_dict and config_dict['palette'] is not None:
            config_dict['palette'] = np.array(config_dict['palette'], dtype=np.float32)
        
        # Ensure background_color is a tuple of integers
        if 'background_color' in config_dict:
            config_dict['background_color'] = tuple(int(c) for c in config_dict['background_color'])
        
        # Ensure final_transform is a tuple of floats if it exists
        if 'final_transform' in config_dict and config_dict['final_transform'] is not None:
            config_dict['final_transform'] = tuple(float(x) for x in config_dict['final_transform'])
        
        return cls(**config_dict)

    def save_to_file(self, filepath):
        """Save the current configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")

    def model_copy(self, **kwargs):
        """Create a copy of the config with updated parameters."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return Config.from_dict(config_dict)


# ------------------------------------------------------------------ VARIATIONS
# All variation functions are compiled with Numba for performance.
# The @njit decorator compiles these to efficient machine code.

@njit
def var_linear(x, y): return x, y

@njit
def var_sinusoidal(x, y): return math.sin(x), math.sin(y)

@njit
def var_spherical(x, y):
    r2 = x * x + y * y + np.float32(1e-6)
    return x / r2, y / r2

@njit
def var_swirl(x, y):
    r2 = x * x + y * y
    return x * math.sin(r2) - y * math.cos(r2), x * math.cos(r2) + y * math.sin(r2)

@njit
def var_horseshoe(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    return (x - y) * (x + y) / r, np.float32(2.0) * x * y / r

@njit
def var_polar(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    theta = math.atan2(y, x)
    return theta / math.pi, r - np.float32(1.0)

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
    angle = random.uniform(np.float32(0.0), np.float32(2.0) * math.pi)
    rad = random.uniform(np.float32(0.0), np.float32(1.0))
    return x + rad * math.cos(angle), y + rad * math.sin(angle)

@njit
def var_fisheye(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    return np.float32(2.0) * y / r, np.float32(2.0) * x / r

@njit
def var_julia(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    theta = math.atan2(y, x)
    return r * math.cos(theta / np.float32(2.0) + math.pi/np.float32(2.0)), r * math.sin(theta / np.float32(2.0) + math.pi/np.float32(2.0))

@njit
def var_popcorn(x, y):
    return x + np.float32(0.1) * math.sin(math.tan(np.float32(3.0) * y)), y + np.float32(0.1) * math.sin(math.tan(np.float32(3.0) * x))

@njit
def var_bent(x, y):
    if x >= 0 and y >= 0:
        return x, y
    if x < 0 and y >= 0:
        return np.float32(2.0) * x, y
    if x >= 0 and y < 0:
        return x, y / np.float32(2.0)
    return np.float32(2.0) * x, y / np.float32(2.0)

@njit
def var_waves(x, y):
    return x + np.float32(0.1) * math.sin(y * np.float32(4.0)), y + np.float32(0.1) * math.sin(x * np.float32(4.0))

@njit
def var_exponential(x, y):
    r = math.exp(x) - np.float32(1.0)
    theta = math.pi * y
    return r * math.cos(theta), r * math.sin(theta)

@njit
def var_julian(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    theta = math.atan2(y, x)
    power = np.float32(2.0)
    r = math.pow(r, power)
    theta = theta * power
    return r * math.cos(theta), r * math.sin(theta)

@njit
def var_rings(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    theta = math.atan2(y, x)
    r = (r + np.float32(0.5) * math.sin(theta * np.float32(5.0))) % np.float32(1.0)
    return r * math.cos(theta), r * math.sin(theta)

@njit
def var_fan(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    theta = math.atan2(y, x)
    if theta < 0:
        theta += np.float32(2.0) * math.pi
    theta = (theta + math.pi) % (math.pi / np.float32(2.0)) - math.pi / np.float32(2.0)
    return r * math.cos(theta), r * math.sin(theta)

@njit
def var_blob(x, y):
    r = math.sqrt(x * x + y * y) + np.float32(1e-6)
    theta = math.atan2(y, x)
    r = r * (np.float32(0.5) + np.float32(0.5) * math.sin(np.float32(6.0) * theta))
    return r * math.cos(theta), r * math.sin(theta)

# ------------------------------------------------------------------ PALETTE LOADER
def load_palettes(folder: Path):
    """
    Loads color palettes from a given folder.
    Supports PNG (top row), TXT (comma-separated), and MAP (space-separated) formats.
    """
    pals = []
    if not folder.exists():
        print(f"Warning: Palette folder '{folder}' not found. Using random colors.")
        return pals
    for f in folder.iterdir():
        try:
            if f.suffix.lower() == ".png":
                img = Image.open(f).convert("RGB")
                # Take the first row of pixels as the palette
                palette = np.array(img)[0, :, :] / 255.0
                pals.append(palette)
            elif f.suffix.lower() == ".txt":
                # Load text file with comma-separated RGB values
                data = np.loadtxt(f, delimiter=",", dtype=np.float32)
                # Normalize to [0,1] if values are in [0,255]
                if np.max(data) > 1.0:
                    data = data / 255.0
                pals.append(data)
            elif f.suffix.lower() == ".map":
                # Load map file with space-separated RGB values
                data = np.loadtxt(f, dtype=np.float32)
                # Normalize to [0,1] if values are in [0,255]
                if np.max(data) > 1.0:
                    data = data / 255.0
                pals.append(data)
        except Exception as e:
            print(f"Warning: Could not load palette '{f}': {e}")
    return pals

def create_sample_palettes():
    """Create sample palette files if they don't exist."""
    palettes_dir = Path("palettes")
    if not palettes_dir.exists():
        palettes_dir.mkdir()
        print("Created palettes directory")
    
    # Create sample palettes if they don't exist
    sample_palettes = {
        "sunset.txt": "255,94,77\n255,154,0\n237,117,57\n255,206,84\n255,157,77\n",
        "ocean.txt": "0,119,190\n0,180,216\n144,224,239\n72,202,228\n0,150,199\n",
        "forest.txt": "34,139,34\n107,142,35\n124,252,0\n0,128,0\n85,107,47\n"
    }
    
    for filename, content in sample_palettes.items():
        palette_path = palettes_dir / filename
        if not palette_path.exists():
            with open(palette_path, "w") as f:
                f.write(content)
            print(f"Created sample palette: {palette_path}")

# ------------------------------------------------------------------ CORE NUMBA ENGINE
@njit
def render_layer(transforms, iterations, width, height, zoom, skip, final_transform):
    """
    Renders a single layer of the fractal.
    This is the core of the chaos game algorithm.
    It's a serial process, as each point depends on the previous one.
    """
    density = np.zeros((height, width), dtype=np.float32)
    color_acc = np.zeros((height, width, 3), dtype=np.float32)
    x, y = np.float32(0.0), np.float32(0.0)
    n_trans = len(transforms)

    for i in range(iterations):
        # Select a random transform from the set
        t = transforms[np.random.randint(0, n_trans)]
        
        # Decompose the transform into its components
        # Pre-affine: a,b,c,d,e,f
        a, b, c, d, e, f = t[0:6]
        # Weight: w
        weight = t[6]
        # Variation index: vi
        var_idx = int(t[7])
        # Color: r,g,b
        col = t[8:11]
        # Post-affine: pa,pb,pc,pd,pe,pf
        pa, pb, pc, pd, pe, pf = t[11:17]

        # 1. Apply the pre-affine transformation
        x1 = a * x + b * y + c
        y1 = d * x + e * y + f

        # 2. Apply the non-linear variation function
        if var_idx == 0:      vx, vy = var_linear(x1, y1)
        elif var_idx == 1:    vx, vy = var_sinusoidal(x1, y1)
        elif var_idx == 2:    vx, vy = var_spherical(x1, y1)
        elif var_idx == 3:    vx, vy = var_swirl(x1, y1)
        elif var_idx == 4:    vx, vy = var_horseshoe(x1, y1)
        elif var_idx == 5:    vx, vy = var_polar(x1, y1)
        elif var_idx == 6:    vx, vy = var_handkerchief(x1, y1)
        elif var_idx == 7:    vx, vy = var_heart(x1, y1)
        elif var_idx == 8:    vx, vy = var_disk(x1, y1)
        elif var_idx == 9:    vx, vy = var_blur(x1, y1)
        elif var_idx == 10:   vx, vy = var_fisheye(x1, y1)
        elif var_idx == 11:   vx, vy = var_julia(x1, y1)
        elif var_idx == 12:   vx, vy = var_popcorn(x1, y1)
        elif var_idx == 13:   vx, vy = var_bent(x1, y1)
        elif var_idx == 14:   vx, vy = var_waves(x1, y1)
        elif var_idx == 15:   vx, vy = var_exponential(x1, y1)
        elif var_idx == 16:   vx, vy = var_julian(x1, y1)
        elif var_idx == 17:   vx, vy = var_rings(x1, y1)
        elif var_idx == 18:   vx, vy = var_fan(x1, y1)
        else:                vx, vy = var_blob(x1, y1) # 19

        # 3. Apply the post-affine transformation
        vx1 = pa * vx + pb * vy + pc
        vy1 = pd * vx + pe * vy + pf

        # 4. Apply the weight and update the chaos game coordinates
        x, y = vx1 * weight, vy1 * weight

        # 5. Apply the final (global) transform if provided
        if final_transform is not None:
            fa, fb, fc, fd, fe, ff = final_transform
            x_final = fa * x + fb * y + fc
            y_final = fd * x + fe * y + ff
            x, y = x_final, y_final

        # 6. Map to screen coordinates and plot the point
        ix = int((x * zoom + np.float32(1.0)) * width / np.float32(2.0))
        iy = int((y * zoom + np.float32(1.0)) * height / np.float32(2.0))

        # Skip the first few iterations (FCAF) to let the attractor settle
        if 0 <= ix < width and 0 <= iy < height and i > skip:
            density[iy, ix] += np.float32(1.0)
            color_acc[iy, ix, 0] += col[0]
            color_acc[iy, ix, 1] += col[1]
            color_acc[iy, ix, 2] += col[2]

    # Normalize the density map using a logarithmic filter for better visibility
    max_d = np.max(density)
    if max_d == 0:
        safe_d = density
    else:
        safe_d = np.log(density + np.float32(1.0)) / np.log(max_d + np.float32(1.0))

    # Normalize the color accumulator
    norm_c = np.where(
        density[..., None] > 0,
        color_acc / (density[..., None] + np.float32(1e-6)),
        np.float32(0.0),
    )
    return safe_d, np.clip(norm_c, 0, 1)

@njit(parallel=True)
def render_all_layers_parallel(all_transforms, iterations, width, height, zoom, skip, final_transforms):
    """
    Renders all layers in parallel.
    Each thread renders one layer independently, which is safe as layers do not depend on each other.
    """
    num_layers = all_transforms.shape[0]
    all_safe_d = np.empty((num_layers, height, width), dtype=np.float32)
    all_norm_c = np.empty((num_layers, height, width, 3), dtype=np.float32)

    for i in prange(num_layers):
        safe_d, norm_c = render_layer(
            all_transforms[i], iterations, width, height, zoom, skip, final_transforms[i]
        )
        all_safe_d[i] = safe_d
        all_norm_c[i] = norm_c
        
    return all_safe_d, all_norm_c

# ------------------------------------------------------------------ FRACTAL CLASS
class FlameFractal:
    """
    Main class for generating and rendering a flame fractal.
    Orchestrates the entire process from building transforms to saving the final image.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_variations = 20 # Total number of variation functions available
        # Load palettes if needed
        if cfg.use_palette and cfg.palette is None:
            palettes = load_palettes(Path("palettes"))
            if palettes:
                self.cfg.palette = random.choice(palettes)
            else:
                print("Warning: No palettes found. Disabling palette usage.")
                self.cfg.use_palette = False

    def rand_color(self):
        """Selects a random color, either from a loaded palette or uniformly."""
        if self.cfg.use_palette and self.cfg.palette is not None:
            return self.cfg.palette[np.random.randint(0, len(self.cfg.palette))]
        return np.random.rand(3)

    def build_transforms(self):
        """
        Creates a set of random affine transforms with variations.
        Each transform now includes a post-affine component for more complexity.
        Structure: (pre-affine, weight, var_idx, color, post-affine)
        Total: 6 + 1 + 1 + 3 + 6 = 17 elements
        """
        ts = []
        for _ in range(self.cfg.transforms):
            # Pre-affine coefficients (random values between -1 and 1)
            a, b, c = np.random.uniform(-1, 1, 3)
            d, e, f = np.random.uniform(-1, 1, 3)
            # Weight
            weight = np.random.uniform(0.5, 1.5)
            # Variation index
            var_idx = np.random.randint(0, self.num_variations)
            # Color
            col = self.rand_color()
            # Post-affine coefficients (random values between -1 and 1)
            pa, pb, pc = np.random.uniform(-1, 1, 3)
            pd, pe, pf = np.random.uniform(-1, 1, 3)
            
            ts.append((a, b, c, d, e, f, weight, var_idx, *col, pa, pb, pc, pd, pe, pf))
        return np.array(ts, dtype=np.float32)

    def apply_symmetry(self, img: np.ndarray) -> np.ndarray:
        """Applies post-render symmetry transformations."""
        if self.cfg.symmetry == "x":
            return np.hstack([img, np.fliplr(img)])
        if self.cfg.symmetry == "y":
            return np.vstack([img, np.flipud(img)])
        if self.cfg.symmetry == "kaleidoscope":
            segs = self.cfg.symmetry_segments
            h, w = img.shape[:2]
            # Use float coordinates for center to be more robust
            cx, cy = w / 2.0, h / 2.0
            y, x = np.ogrid[:h, :w]
            theta = np.arctan2(y - cy, x - cx)
            theta_mod = np.mod(theta, 2 * math.pi / segs)
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            x_new = (r * np.cos(theta_mod) + cx).astype(int)
            y_new = (r * np.sin(theta_mod) + cy).astype(int)
            x_new = np.clip(x_new, 0, w - 1)
            y_new = np.clip(y_new, 0, h - 1)
            out = np.empty_like(img)
            out[...] = img[y_new, x_new]
            return out
        return img

    def art_filter(self, img: np.ndarray) -> np.ndarray:
        """Applies artistic filters to the final image."""
        style = self.cfg.art_style
        if style is None:
            return img
        
        # Convert to PIL for filters that use it
        pil = Image.fromarray(img)

        if style == "pointillism":
            # Create a new blank canvas
            dots = Image.new("RGB", pil.size)
            draw = ImageDraw.Draw(dots)
            
            # Define point size and density
            point_size = 2
            num_points = (pil.width * pil.height) // 200  # Reduced density for better performance
            
            # Get colors from the original image
            pixels = np.array(pil)
            h, w = pixels.shape[:2]
            
            # Vectorized approach for better performance
            for _ in range(num_points):
                # Pick a random point
                x, y = random.randint(0, w - 1), random.randint(0, h - 1)
                # Get a small patch color from the original image
                y1, y2 = max(0, y-1), min(h, y+2)
                x1, x2 = max(0, x-1), min(w, x+2)
                patch = pixels[y1:y2, x1:x2]
                if patch.size > 0:
                    color = tuple(np.mean(patch.reshape(-1, 3), axis=0).astype(int))
                    # Draw a small circle
                    draw.ellipse([x-point_size, y-point_size, x+point_size, y+point_size], fill=tuple(color))
            pil = dots
        elif style == "expressionist":
            pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
            img_arr = np.array(pil, float) * 1.5
            return np.clip(img_arr, 0, 255).astype(np.uint8)
        elif style == "oil":
            pil = pil.filter(ImageFilter.ModeFilter(size=6)).filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif style == "watercolor":
            # Add noise for texture before blurring
            img_arr = np.array(pil, float)
            noise = np.random.normal(0, 10, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255)
            pil = Image.fromarray(img_arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
            img_arr = np.array(pil, float) * 0.9
            return np.clip(img_arr, 0, 255).astype(np.uint8)
            
        return np.array(pil)

    def render_single(self, show_progress=False) -> np.ndarray:
        """
        Renders a single static image by compositing multiple layers.
        Layers are rendered in parallel for performance.
        """
        # Pre-generate all transforms and final transforms for each layer
        all_transforms = np.empty((self.cfg.layers, self.cfg.transforms, 17), dtype=np.float32)
        all_final_transforms = np.empty((self.cfg.layers, 6), dtype=np.float32)
        
        for i in range(self.cfg.layers):
            all_transforms[i] = self.build_transforms()
            if self.cfg.final_transform is not None:
                all_final_transforms[i] = np.array(self.cfg.final_transform, dtype=np.float32)
            else:
                # Identity transform
                all_final_transforms[i] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        zoom = np.float32(self.cfg.zoom)
        
        # Render all layers in parallel
        all_safe_d, all_norm_c = render_all_layers_parallel(
            all_transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
            zoom, self.cfg.skip, all_final_transforms
        )
        
        # Combine the rendered layers
        combined = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.float32)
        layer_iterator = range(self.cfg.layers)
        if show_progress:
            try:
                from tqdm import trange
                layer_iterator = trange(self.cfg.layers, desc="Compositing Layers")
            except ImportError:
                pass

        for i in layer_iterator:
            safe_d = all_safe_d[i]
            norm_c = all_norm_c[i]
            
            # Apply vibrancy control
            if self.cfg.vibrancy < 1.0:
                # Mix saturated color with brightness
                luminance = 0.299 * norm_c[..., 0] + 0.587 * norm_c[..., 1] + 0.114 * norm_c[..., 2]
                base_color = norm_c * self.cfg.vibrancy + luminance[..., None] * (1.0 - self.cfg.vibrancy)
            else:
                # Use gamma-corrected saturated color
                base_color = norm_c ** (1.0 / self.cfg.gamma)

            # Apply the density map to the base color
            rgb = base_color * safe_d[..., None] * 255.0
            combined += np.clip(rgb, 0, 255)
        
        combined = (combined / self.cfg.layers).astype(np.uint8)
        
        # Apply post-processing steps
        bg_color = np.array(self.cfg.background_color, dtype=np.uint8)
        if not np.array_equal(bg_color, [0, 0, 0]):
            # Create a mask for non-background pixels
            mask = np.any(combined > 0, axis=2)
            bg = np.tile(bg_color, (self.cfg.height, self.cfg.width, 1))
            combined = np.where(mask[..., None], combined, bg)
        
        combined = self.apply_symmetry(combined)
        combined = self.art_filter(combined)
        return combined

    def render_animation(self, filename: str, show_progress=False):
        """Renders an animated GIF by slightly modifying transforms between frames."""
        frames = []
        base_transforms = self.build_transforms()
        
        final_transform = None
        if self.cfg.final_transform is not None:
            final_transform = tuple(np.float32(x) for x in self.cfg.final_transform)
        else:
            final_transform = (np.float32(1.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(1.0), np.float32(0.0))
        
        frame_iterator = range(self.cfg.frames)
        if show_progress:
            try:
                from tqdm import trange
                frame_iterator = trange(self.cfg.frames, desc="Creating Frames")
            except ImportError:
                pass
        
        for f in frame_iterator:
            # Create a copy of the base transforms
            transforms = base_transforms.copy()
            
            # Jitter weight & color drift for animation
            for t in transforms:
                # Subtle weight change (±2%)
                t[6] *= random.uniform(0.98, 1.02)
                # Subtle color drift (5% new color)
                t[8:11] = t[8:11] * 0.95 + self.rand_color() * 0.05
            
            # Use the serial renderer for each frame
            safe_d, norm_c = render_layer(
                transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
                np.float32(self.cfg.zoom), self.cfg.skip, final_transform
            )
            
            # Apply vibrancy control
            if self.cfg.vibrancy < 1.0:
                luminance = 0.299 * norm_c[..., 0] + 0.587 * norm_c[..., 1] + 0.114 * norm_c[..., 2]
                base_color = norm_c * self.cfg.vibrancy + luminance[..., None] * (1.0 - self.cfg.vibrancy)
            else:
                base_color = norm_c ** (1.0 / self.cfg.gamma)

            rgb = base_color * safe_d[..., None] * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            
            # Apply background color
            bg_color = np.array(self.cfg.background_color, dtype=np.uint8)
            if not np.array_equal(bg_color, [0, 0, 0]):
                mask = np.any(rgb > 0, axis=2)
                bg = np.ones_like(rgb) * bg_color
                rgb = np.where(mask[..., None], rgb, bg)
            
            # Apply symmetry and art filters
            rgb = self.apply_symmetry(rgb)
            rgb = self.art_filter(rgb)
            frames.append(rgb)
            
            if not show_progress:
                print(f"Frame {f+1}/{self.cfg.frames} rendered.")
        
        try:
            imageio.mimsave(filename, frames, duration=0.1, loop=0)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Try to save as individual frames
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(f"frame_{i:03d}.png")
            print("Saved individual frames as fallback")

# ------------------------------------------------------------------ MAIN / CLI
def main():
    import argparse
    import textwrap

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""\
        High-performance flame-fractal generator with post-transforms.
        -------------------------------------------------------
        Place palettes in a './palettes/' directory.
        Supported formats: PNG (uses top row), TXT (comma-separated), or MAP (space-separated).

        Examples:
        # Generate a 1024x1024 image with 5 million iterations
        python main.py -W 1024 -H 1024 -i 5000000 -o my_fractal.png

        # Generate a 30-frame kaleidoscope GIF with a watercolor style
        python main.py -f 30 --symmetry kaleidoscope --style watercolor -o animation.gif

        # Use a specific palette and seed for reproducibility
        python main.py --palette --seed 42 -o reproducible.png

        # Load a preset configuration
        python main.py --preset my_config.json -o from_preset.png

        # Save current configuration to a preset file
        python main.py --save-preset my_config.json
        """)
    )
    ap.add_argument("-W", "--width", type=int, default=800, help="Output width in pixels.")
    ap.add_argument("-H", "--height", type=int, default=800, help="Output height in pixels.")
    ap.add_argument("-i", "--iterations", type=int, default=3_000_000, help="Number of fractal iterations.")
    ap.add_argument("-t", "--transforms", type=int, default=15, help="Number of transforms to generate.")
    ap.add_argument("-z", "--zoom", type=float, default=1.2, help="Zoom level.")
    ap.add_argument("-s", "--skip", type=int, default=100, help="Number of initial iterations to skip (FCAF).")
    ap.add_argument("-g", "--gamma", type=float, default=2.2, help="Gamma correction value.")
    ap.add_argument("-l", "--layers", type=int, default=3, help="Number of layers to composite.")
    ap.add_argument("-f", "--frames", type=int, default=0, help="Number of frames for GIF animation (0 for single image).")
    ap.add_argument("--symmetry", choices=[None, "x", "y", "kaleidoscope"], default=None, help="Apply symmetry.")
    ap.add_argument("--segments", type=int, default=6, help="Number of segments for kaleidoscope symmetry.")
    ap.add_argument("--style", choices=[None, "pointillism", "expressionist", "oil", "watercolor"], default=None, help="Apply an artistic style.")
    ap.add_argument("--palette", action="store_true", help="Use a random palette from the './palettes' folder.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for random number generator for reproducible results.")
    ap.add_argument("-o", "--output", default=None, help="Output filename (auto-generated if not provided).")
    
    ap.add_argument("--progress", action="store_true", help="Show progress bars during rendering.")
    ap.add_argument("--bg-color", nargs=3, type=int, default=[0, 0, 0], metavar=('R', 'G', 'B'), 
                    help="Background color in RGB format (0-255). Default: 0 0 0 (black).")
    ap.add_argument("--preset", default=None, help="Load configuration from a JSON preset file.")
    ap.add_argument("--save-preset", default=None, help="Save current configuration to a JSON preset file.")
    
    ap.add_argument("--vibrancy", type=float, default=1.0, help="Color vibrancy (0.0-1.0). 1.0 is full color, 0.0 is grayscale.")
    ap.add_argument("--final-transform", nargs=6, type=float, default=None, metavar=('a', 'b', 'c', 'd', 'e', 'f'),
                    help="Final affine transform coefficients (a,b,c,d,e,f).")
    
    args = ap.parse_args()

    try:
        # Handle preset loading first
        config_dict = {}
        if args.preset:
            try:
                with open(args.preset, 'r') as f:
                    config_dict = json.load(f)
                print(f"Loaded configuration from {args.preset}")
            except Exception as e:
                print(f"Error loading preset: {e}")
                sys.exit(1)

        # Override with command-line arguments
        config_dict.update({
            'width': args.width,
            'height': args.height,
            'iterations': args.iterations,
            'transforms': args.transforms,
            'zoom': args.zoom,
            'skip': args.skip,
            'gamma': args.gamma,
            'layers': args.layers,
            'frames': args.frames,
            'symmetry': args.symmetry,
            'symmetry_segments': args.segments,
            'use_palette': args.palette,
            'art_style': args.style,
            'seed': args.seed,
            'background_color': tuple(args.bg_color),
            'vibrancy': args.vibrancy,
            'final_transform': args.final_transform,
        })

        # Create config object
        cfg = Config.from_dict(config_dict)

        # Save preset if requested
        if args.save_preset:
            cfg.save_to_file(args.save_preset)
            if not args.output and args.frames == 0:
                print("Preset saved. No output requested.")
                sys.exit(0)

        # Create sample palettes if needed
        if args.palette and not Path("palettes").exists():
            create_sample_palettes()

        # Initialize and render fractal
        fractal = FlameFractal(cfg)

        if cfg.frames > 0:
            out_file = args.output or f"flame_anim_{int(time.time())}.gif"
            fractal.render_animation(out_file, args.progress)
        else:
            out_file = args.output or f"flame_{int(time.time())}.png"
            img = fractal.render_single(args.progress)
            try:
                Image.fromarray(img).save(out_file)
                print(f"Image saved to {out_file}")
            except Exception as e:
                print(f"Error saving image: {e}")
                # Save as PNG if other formats fail
                png_file = os.path.splitext(out_file)[0] + ".png"
                Image.fromarray(img).save(png_file)
                print(f"Saved as PNG fallback: {png_file}")
                sys.exit(1)

    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()