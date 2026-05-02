import sys
import os
import random
import math
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import imageio
from numba import njit, prange

# PySide6 Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, 
                             QPushButton, QComboBox, QCheckBox, QFileDialog,
                             QGroupBox, QFormLayout, QColorDialog, QMessageBox,
                             QLineEdit, QProgressBar, QScrollArea)
from PySide6.QtCore import QThread, Signal, Qt, QCoreApplication
from PySide6.QtGui import QImage, QPixmap, QColor

# ==============================================================================
# 1. CONFIGURATION & CORE LOGIC (Integrated from provided scripts)
# ==============================================================================

class Config:
    """Configuration container for all fractal parameters."""
    def __init__(self,
                 width=800, height=800,
                 iterations=3_000_000, transforms=15,
                 zoom=1.2, skip=100, gamma=2.2,
                 layers=3, frames=0,
                 symmetry=None, symmetry_segments=6,
                 palette=None, use_palette=True,
                 art_style=None, seed=None,
                 background_color=(0, 0, 0),
                 vibrancy=1.0,
                 final_transform=None):
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

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def to_dict(self):
        d = {
            'width': self.width, 'height': self.height,
            'iterations': self.iterations, 'transforms': self.transforms,
            'zoom': self.zoom, 'skip': self.skip, 'gamma': self.gamma,
            'layers': self.layers, 'frames': self.frames,
            'symmetry': self.symmetry, 'symmetry_segments': self.symmetry_segments,
            'use_palette': self.use_palette, 'art_style': self.art_style,
            'seed': self.seed, 'background_color': self.background_color,
            'vibrancy': self.vibrancy, 'final_transform': self.final_transform
        }
        if self.palette is not None:
            d['palette'] = self.palette.tolist()
        else:
            d['palette'] = None
        return d

    @classmethod
    def from_dict(cls, d):
        if 'palette' in d and d['palette'] is not None:
            d['palette'] = np.array(d['palette'], dtype=np.float32)
        return cls(**d)

# ------------------------------------------------------------------------------
# NUMBA OPTIMIZED ENGINE (From flamefractal6.py & main.py)
# ------------------------------------------------------------------------------

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
    return (x - y) * (x + y) / r, 2.0 * x * y / r
@njit
def var_polar(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    return theta / math.pi, r - 1.0
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
    angle = random.uniform(0.0, 2.0 * math.pi)
    rad = random.uniform(0.0, 1.0)
    return x + rad * math.cos(angle), y + rad * math.sin(angle)
@njit
def var_fisheye(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    return 2.0 * y / r, 2.0 * x / r
@njit
def var_julia(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    return r * math.cos(theta / 2.0 + math.pi/2.0), r * math.sin(theta / 2.0 + math.pi/2.0)
@njit
def var_popcorn(x, y):
    return x + 0.1 * math.sin(math.tan(3.0 * y)), y + 0.1 * math.sin(math.tan(3.0 * x))
@njit
def var_bent(x, y):
    if x >= 0 and y >= 0: return x, y
    if x < 0 and y >= 0: return 2.0 * x, y
    if x >= 0 and y < 0: return x, y / 2.0
    return 2.0 * x, y / 2.0
@njit
def var_waves(x, y):
    return x + 0.1 * math.sin(y * 4.0), y + 0.1 * math.sin(x * 4.0)
@njit
def var_exponential(x, y):
    r = math.exp(x) - 1.0
    theta = math.pi * y
    return r * math.cos(theta), r * math.sin(theta)
@njit
def var_julian(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    power = 2.0
    r = math.pow(r, power)
    theta = theta * power
    return r * math.cos(theta), r * math.sin(theta)
@njit
def var_rings(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    r = (r + 0.5 * math.sin(theta * 5.0)) % 1.0
    return r * math.cos(theta), r * math.sin(theta)
@njit
def var_fan(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    if theta < 0: theta += 2.0 * math.pi
    theta = (theta + math.pi) % (math.pi / 2.0) - math.pi / 2.0
    return r * math.cos(theta), r * math.sin(theta)
@njit
def var_blob(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    r = r * (0.5 + 0.5 * math.sin(6.0 * theta))
    return r * math.cos(theta), r * math.sin(theta)

@njit
def render_layer(transforms, iterations, width, height, zoom, skip, final_transform):
    density = np.zeros((height, width), dtype=np.float32)
    color_acc = np.zeros((height, width, 3), dtype=np.float32)
    x, y = 0.0, 0.0
    n_trans = len(transforms)

    for i in range(iterations):
        t = transforms[np.random.randint(0, n_trans)]
        a, b, c, d, e, f = t[0:6]
        weight = t[6]
        var_idx = int(t[7])
        col = t[8:11]
        pa, pb, pc, pd, pe, pf = t[11:17]

        x1 = a * x + b * y + c
        y1 = d * x + e * y + f

        # Variation Dispatch
        if var_idx == 0: vx, vy = var_linear(x1, y1)
        elif var_idx == 1: vx, vy = var_sinusoidal(x1, y1)
        elif var_idx == 2: vx, vy = var_spherical(x1, y1)
        elif var_idx == 3: vx, vy = var_swirl(x1, y1)
        elif var_idx == 4: vx, vy = var_horseshoe(x1, y1)
        elif var_idx == 5: vx, vy = var_polar(x1, y1)
        elif var_idx == 6: vx, vy = var_handkerchief(x1, y1)
        elif var_idx == 7: vx, vy = var_heart(x1, y1)
        elif var_idx == 8: vx, vy = var_disk(x1, y1)
        elif var_idx == 9: vx, vy = var_blur(x1, y1)
        elif var_idx == 10: vx, vy = var_fisheye(x1, y1)
        elif var_idx == 11: vx, vy = var_julia(x1, y1)
        elif var_idx == 12: vx, vy = var_popcorn(x1, y1)
        elif var_idx == 13: vx, vy = var_bent(x1, y1)
        elif var_idx == 14: vx, vy = var_waves(x1, y1)
        elif var_idx == 15: vx, vy = var_exponential(x1, y1)
        elif var_idx == 16: vx, vy = var_julian(x1, y1)
        elif var_idx == 17: vx, vy = var_rings(x1, y1)
        elif var_idx == 18: vx, vy = var_fan(x1, y1)
        else: vx, vy = var_blob(x1, y1)

        # Post-affine
        vx1 = pa * vx + pb * vy + pc
        vy1 = pd * vx + pe * vy + pf
        x, y = vx1 * weight, vy1 * weight

        if final_transform is not None:
            fa, fb, fc, fd, fe, ff = final_transform
            x_final = fa * x + fb * y + fc
            y_final = fd * x + fe * y + ff
            x, y = x_final, y_final

        ix = int((x * zoom + 1.0) * width / 2.0)
        iy = int((y * zoom + 1.0) * height / 2.0)

        if 0 <= ix < width and 0 <= iy < height and i > skip:
            density[iy, ix] += 1.0
            color_acc[iy, ix, 0] += col[0]
            color_acc[iy, ix, 1] += col[1]
            color_acc[iy, ix, 2] += col[2]

    max_d = np.max(density)
    if max_d == 0: safe_d = density
    else: safe_d = np.log(density + 1.0) / np.log(max_d + 1.0)
    
    norm_c = np.where(density[..., None] > 0, color_acc / (density[..., None] + 1e-6), 0.0)
    return safe_d, np.clip(norm_c, 0, 1)

@njit(parallel=True)
def render_all_layers_parallel(all_transforms, iterations, width, height, zoom, skip, final_transforms):
    num_layers = all_transforms.shape[0]
    all_safe_d = np.empty((num_layers, height, width), dtype=np.float32)
    all_norm_c = np.empty((num_layers, height, width, 3), dtype=np.float32)
    for i in prange(num_layers):
        sd, nc = render_layer(all_transforms[i], iterations, width, height, zoom, skip, final_transforms[i])
        all_safe_d[i] = sd
        all_norm_c[i] = nc
    return all_safe_d, all_norm_c

# ------------------------------------------------------------------------------
# FRACTAL GENERATOR CLASS
# ------------------------------------------------------------------------------

class FlameFractal:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_variations = 20
        self.current_palette_name = "Random"

    def rand_color(self):
        if self.cfg.use_palette and self.cfg.palette is not None:
            return self.cfg.palette[np.random.randint(0, len(self.cfg.palette))]
        return np.random.rand(3)

    def build_transforms(self):
        ts = []
        for _ in range(self.cfg.transforms):
            # Pre-affine
            a, b, c = np.random.uniform(-1, 1, 3)
            d, e, f = np.random.uniform(-1, 1, 3)
            weight = np.random.uniform(0.5, 1.5)
            var_idx = np.random.randint(0, self.num_variations)
            col = self.rand_color()
            # Post-affine
            pa, pb, pc = np.random.uniform(-1, 1, 3)
            pd, pe, pf = np.random.uniform(-1, 1, 3)
            ts.append((a, b, c, d, e, f, weight, var_idx, *col, pa, pb, pc, pd, pe, pf))
        return np.array(ts, dtype=np.float32)

    def apply_symmetry(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.symmetry == "x":
            return np.hstack([img, np.fliplr(img)])
        if self.cfg.symmetry == "y":
            return np.vstack([img, np.flipud(img)])
        if self.cfg.symmetry == "kaleidoscope":
            segs = self.cfg.symmetry_segments
            h, w = img.shape[:2]
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
        style = self.cfg.art_style
        if style is None: return img
        pil = Image.fromarray(img)

        if style == "pointillism":
            dots = Image.new("RGB", pil.size)
            draw = ImageDraw.Draw(dots)
            pixels = np.array(pil)
            h, w = pixels.shape[:2]
            for _ in range((w * h) // 200):
                x, y = random.randint(0, w - 1), random.randint(0, h - 1)
                patch = pixels[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)]
                color = tuple(np.mean(patch.reshape(-1, 3), axis=0).astype(int))
                draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
            pil = dots
        elif style == "expressionist":
            pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
            img_arr = np.array(pil, float) * 1.5
            return np.clip(img_arr, 0, 255).astype(np.uint8)
        elif style == "oil":
            pil = pil.filter(ImageFilter.ModeFilter(size=6)).filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif style == "watercolor":
            img_arr = np.array(pil, float)
            noise = np.random.normal(0, 10, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255)
            pil = Image.fromarray(img_arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
            img_arr = np.array(pil, float) * 0.9
            return np.clip(img_arr, 0, 255).astype(np.uint8)
        return np.array(pil)

    def render_single(self):
        all_transforms = np.empty((self.cfg.layers, self.cfg.transforms, 17), dtype=np.float32)
        all_final_transforms = np.empty((self.cfg.layers, 6), dtype=np.float32)
        
        for i in range(self.cfg.layers):
            all_transforms[i] = self.build_transforms()
            if self.cfg.final_transform is not None:
                all_final_transforms[i] = np.array(self.cfg.final_transform, dtype=np.float32)
            else:
                all_final_transforms[i] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        zoom = np.float32(self.cfg.zoom)
        all_safe_d, all_norm_c = render_all_layers_parallel(
            all_transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
            zoom, self.cfg.skip, all_final_transforms
        )
        
        combined = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.float32)
        for i in range(self.cfg.layers):
            safe_d = all_safe_d[i]
            norm_c = all_norm_c[i]
            if self.cfg.vibrancy < 1.0:
                luminance = 0.299 * norm_c[..., 0] + 0.587 * norm_c[..., 1] + 0.114 * norm_c[..., 2]
                base_color = norm_c * self.cfg.vibrancy + luminance[..., None] * (1.0 - self.cfg.vibrancy)
            else:
                base_color = norm_c ** (1.0 / self.cfg.gamma)
            rgb = base_color * safe_d[..., None] * 255.0
            combined += np.clip(rgb, 0, 255)
        
        combined = (combined / self.cfg.layers).astype(np.uint8)
        
        bg_color = np.array(self.cfg.background_color, dtype=np.uint8)
        if not np.array_equal(bg_color, [0, 0, 0]):
            mask = np.any(combined > 0, axis=2)
            bg = np.tile(bg_color, (self.cfg.height, self.cfg.width, 1))
            combined = np.where(mask[..., None], combined, bg)
        
        combined = self.apply_symmetry(combined)
        combined = self.art_filter(combined)
        return combined

    def render_animation(self, filename):
        frames = []
        base_transforms = self.build_transforms()
        final_transform = (np.float32(1.0), np.float32(0.0), np.float32(0.0), 
                           np.float32(0.0), np.float32(1.0), np.float32(0.0))
        if self.cfg.final_transform:
            final_transform = tuple(np.float32(x) for x in self.cfg.final_transform)

        for f in range(self.cfg.frames):
            transforms = base_transforms.copy()
            for t in transforms:
                t[6] *= random.uniform(0.98, 1.02) # Weight jitter
                t[8:11] = t[8:11] * 0.95 + self.rand_color() * 0.05 # Color drift
            
            safe_d, norm_c = render_layer(
                transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
                np.float32(self.cfg.zoom), self.cfg.skip, final_transform
            )
            
            if self.cfg.vibrancy < 1.0:
                luminance = 0.299 * norm_c[..., 0] + 0.587 * norm_c[..., 1] + 0.114 * norm_c[..., 2]
                base_color = norm_c * self.cfg.vibrancy + luminance[..., None] * (1.0 - self.cfg.vibrancy)
            else:
                base_color = norm_c ** (1.0 / self.cfg.gamma)
            
            rgb = base_color * safe_d[..., None] * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = self.apply_symmetry(rgb)
            rgb = self.art_filter(rgb)
            frames.append(rgb)
        
        imageio.mimsave(filename, frames, duration=0.1, loop=0)
        return filename

# ==============================================================================
# 2. GUI WORKER THREAD
# ==============================================================================

class GeneratorThread(QThread):
    progress = Signal(str)
    finished = Signal(object) # QImage or str (filepath)
    error = Signal(str)

    def __init__(self, config, save_path=None):
        super().__init__()
        self.config = config
        self.save_path = save_path
        self.is_anim = config.frames > 0

    def run(self):
        try:
            self.progress.emit("Initializing Engine...")
            fractal = FlameFractal(self.config)
            
            if self.is_anim:
                self.progress.emit(f"Generating {self.config.frames} Frames...")
                fname = self.save_path or "animation.gif"
                fractal.render_animation(fname)
                self.progress.emit("GIF Saved.")
                self.finished.emit(fname)
            else:
                self.progress.emit("Rendering Layers...")
                img_array = fractal.render_single()
                self.progress.emit("Processing Image...")
                
                # Convert to QImage
                h, w, c = img_array.shape
                bytes_per_line = 3 * w
                q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                
                if self.save_path:
                    q_img.save(self.save_path)
                    self.progress.emit("Image Saved.")
                
                self.finished.emit(q_img)

        except Exception as e:
            self.error.emit(str(e))

# ==============================================================================
# 3. MAIN WINDOW GUI
# ==============================================================================

class FlameStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Flame Fractal Studio")
        self.resize(1200, 800)
        
        self.config = Config()
        self.worker = None
        self.last_image = None
        
        self.init_ui()
        self.load_palettes()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # --- Left Panel: Controls ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        left_widget = QWidget()
        form = QVBoxLayout(left_widget)
        
        # -- Dimensions Group --
        grp_dim = QGroupBox("Dimensions & Quality")
        lay_dim = QFormLayout()
        self.spin_w = QSpinBox(); self.spin_w.setRange(100, 4000); self.spin_w.setValue(self.config.width)
        self.spin_h = QSpinBox(); self.spin_h.setRange(100, 4000); self.spin_h.setValue(self.config.height)
        self.spin_iter = QSpinBox(); self.spin_iter.setRange(10000, 100000000); self.spin_iter.setValue(self.config.iterations); self.spin_iter.setSingleStep(100000)
        self.spin_skip = QSpinBox(); self.spin_skip.setRange(0, 1000); self.spin_skip.setValue(self.config.skip)
        lay_dim.addRow("Width:", self.spin_w)
        lay_dim.addRow("Height:", self.spin_h)
        lay_dim.addRow("Iterations:", self.spin_iter)
        lay_dim.addRow("Skip (FCAF):", self.spin_skip)
        grp_dim.setLayout(lay_dim)
        form.addWidget(grp_dim)
        
        # -- Transform Group --
        grp_trans = QGroupBox("Transforms")
        lay_trans = QFormLayout()
        self.spin_num_trans = QSpinBox(); self.spin_num_trans.setRange(1, 50); self.spin_num_trans.setValue(self.config.transforms)
        self.spin_zoom = QDoubleSpinBox(); self.spin_zoom.setRange(0.1, 10.0); self.spin_zoom.setValue(self.config.zoom)
        self.spin_layers = QSpinBox(); self.spin_layers.setRange(1, 10); self.spin_layers.setValue(self.config.layers)
        lay_trans.addRow("Num Transforms:", self.spin_num_trans)
        lay_trans.addRow("Zoom:", self.spin_zoom)
        lay_trans.addRow("Layers:", self.spin_layers)
        grp_trans.setLayout(lay_trans)
        form.addWidget(grp_trans)
        
        # -- Color & Style Group --
        grp_color = QGroupBox("Color & Style")
        lay_color = QFormLayout()
        self.chk_palette = QCheckBox(); self.chk_palette.setChecked(self.config.use_palette)
        self.combo_palette = QComboBox()
        self.combo_style = QComboBox(); self.combo_style.addItems(["None", "pointillism", "expressionist", "oil", "watercolor"])
        self.spin_gamma = QDoubleSpinBox(); self.spin_gamma.setRange(0.1, 5.0); self.spin_gamma.setValue(self.config.gamma)
        self.slide_vibrancy = QDoubleSpinBox(); self.slide_vibrancy.setRange(0.0, 1.0); self.slide_vibrancy.setValue(self.config.vibrancy); self.slide_vibrancy.setSingleStep(0.1)
        
        self.btn_bg_color = QPushButton("Select Color")
        self.bg_color = self.config.background_color
        self.update_color_btn()
        self.btn_bg_color.clicked.connect(self.select_bg_color)
        
        lay_color.addRow("Use Palette:", self.chk_palette)
        lay_color.addRow("Select Palette:", self.combo_palette)
        lay_color.addRow("Art Style:", self.combo_style)
        lay_color.addRow("Gamma:", self.spin_gamma)
        lay_color.addRow("Vibrancy:", self.slide_vibrancy)
        lay_color.addRow("Background:", self.btn_bg_color)
        grp_color.setLayout(lay_color)
        form.addWidget(grp_color)
        
        # -- Symmetry Group --
        grp_sym = QGroupBox("Symmetry")
        lay_sym = QFormLayout()
        self.combo_sym = QComboBox(); self.combo_sym.addItems(["None", "x", "y", "kaleidoscope"])
        self.spin_sym_seg = QSpinBox(); self.spin_sym_seg.setRange(2, 360); self.spin_sym_seg.setValue(self.config.symmetry_segments)
        lay_sym.addRow("Type:", self.combo_sym)
        lay_sym.addRow("Segments:", self.spin_sym_seg)
        grp_sym.setLayout(lay_sym)
        form.addWidget(grp_sym)
        
        # -- Animation Group --
        grp_anim = QGroupBox("Animation (GIF)")
        lay_anim = QFormLayout()
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(0, 120); self.spin_frames.setValue(self.config.frames)
        lay_anim.addRow("Frames (0=Static):", self.spin_frames)
        grp_anim.setLayout(lay_anim)
        form.addWidget(grp_anim)
        
        # -- Buttons --
        self.btn_gen = QPushButton("Generate")
        self.btn_gen.clicked.connect(self.start_generation)
        self.btn_save = QPushButton("Save Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        
        form.addWidget(self.btn_gen)
        form.addWidget(self.btn_save)
        
        # Progress
        self.status_lbl = QLabel("Ready")
        form.addWidget(self.status_lbl)
        
        form.addStretch()
        scroll.setWidget(left_widget)
        layout.addWidget(scroll)
        
        # --- Right Panel: Image ---
        self.img_label = QLabel("Output will appear here")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #222; color: #888;")
        self.img_label.setMinimumSize(600, 600)
        layout.addWidget(self.img_label, 1)
        
    def update_color_btn(self):
        r, g, b = self.bg_color
        self.btn_bg_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {'white' if r+g+b < 400 else 'black'};")
        
    def select_bg_color(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.bg_color = (col.red(), col.green(), col.blue())
            self.update_color_btn()
            
    def load_palettes(self):
        self.palettes_map = {"Random": None}
        p_dir = Path("palettes")
        if p_dir.exists():
            for f in p_dir.iterdir():
                if f.suffix.lower() in [".png", ".txt", ".map"]:
                    self.palettes_map[f.stem] = f
        self.combo_palette.addItems(self.palettes_map.keys())
        
    def get_current_config(self):
        sym_map = {"None": None, "x": "x", "y": "y", "kaleidoscope": "kaleidoscope"}
        
        # Handle Palette
        p_name = self.combo_palette.currentText()
        p_data = None
        if p_name != "Random" and p_name in self.palettes_map:
            try:
                p_path = self.palettes_map[p_name]
                if p_path.suffix == ".png":
                    img = Image.open(p_path).convert("RGB")
                    p_data = np.array(img)[0, :, :]/255.0
                else:
                    p_data = np.loadtxt(p_path, delimiter=",", dtype=np.float32)
                    if np.max(p_data) > 1: p_data /= 255.0
            except: pass
        
        return Config(
            width=self.spin_w.value(),
            height=self.spin_h.value(),
            iterations=self.spin_iter.value(),
            transforms=self.spin_num_trans.value(),
            zoom=self.spin_zoom.value(),
            skip=self.spin_skip.value(),
            gamma=self.spin_gamma.value(),
            layers=self.spin_layers.value(),
            frames=self.spin_frames.value(),
            symmetry=sym_map[self.combo_sym.currentText()],
            symmetry_segments=self.spin_sym_seg.value(),
            use_palette=self.chk_palette.isChecked(),
            palette=p_data,
            art_style=self.combo_style.currentText().lower() if self.combo_style.currentText() != "None" else None,
            background_color=self.bg_color,
            vibrancy=self.slide_vibrancy.value()
        )

    def start_generation(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Generation in progress...")
            return
            
        self.config = self.get_current_config()
        self.worker = GeneratorThread(self.config)
        self.worker.progress.connect(self.status_lbl.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()
        self.btn_gen.setEnabled(False)
        self.btn_save.setEnabled(False)

    def on_finished(self, result):
        self.btn_gen.setEnabled(True)
        if isinstance(result, QImage):
            self.last_image = result
            self.img_label.setPixmap(QPixmap.fromImage(result).scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_save.setEnabled(True)
        elif isinstance(result, str):
            # GIF finished
            self.status_lbl.setText(f"GIF Saved: {result}")
            # Optionally show preview? We just show a placeholder text
            self.img_label.setText(f"Saved to file:\n{result}")

    def save_image(self):
        if not self.last_image: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "flame.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.last_image.save(path)
            self.status_lbl.setText(f"Saved to {path}")

# ==============================================================================
# 4. MAIN
# ==============================================================================

if __name__ == "__main__":
    # Create palette directory if missing for user convenience
    if not Path("palettes").exists():
        Path("palettes").mkdir()
        # Create a dummy palette
        with open("palettes/random_default.txt", "w") as f:
            f.write("1.0,0.0,0.0\n0.0,1.0,0.0\n0.0,0.0,1.0\n1.0,1.0,0.0\n")
            
    app = QApplication(sys.argv)
    window = FlameStudio()
    window.show()
    sys.exit(app.exec())