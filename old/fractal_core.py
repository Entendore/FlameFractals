import logging
import random
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
from numba import njit, prange
import imageio

# Setup module logger
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    def __init__(self,
                 width=800, height=800,
                 iterations=3_000_000, transforms=15,
                 zoom=1.2, skip=100, gamma=2.2,
                 layers=3, frames=0,
                 symmetry=None, symmetry_segments=6,
                 palette=None, use_palette=True,
                 art_style=None, seed=None,
                 background_color=(0, 0, 0),
                 vibrancy=1.0):

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

        # Robust Seeding
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

# --- Numba Optimized Variations ---
# (Variations remain the same)
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
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return theta / math.pi * math.sin(math.pi * r), theta / math.pi * math.cos(math.pi * r)

@njit
def var_blur(x, y):
    angle = np.random.random() * 2.0 * math.pi
    rad = np.random.random() * 0.5
    return x + rad * math.cos(angle), y + rad * math.sin(angle)

@njit
def var_fisheye(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    return 2.0 * y / r, 2.0 * x / r

@njit
def var_julia(x, y):
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    sqrt_r = math.sqrt(r)
    return sqrt_r * math.cos(theta / 2.0 + math.pi / 2.0), sqrt_r * math.sin(theta / 2.0 + math.pi / 2.0)

@njit
def var_bent(x, y):
    if x >= 0 and y >= 0: return x, y
    if x < 0 and y >= 0: return 2.0 * x, y
    if x >= 0 and y < 0: return x, y / 2.0
    return 2.0 * x, y / 2.0

@njit
def var_popcorn(x, y):
    c = 1.0 
    f = 1.0
    return x + c * math.sin(math.tan(3.0 * y)), y + f * math.sin(math.tan(3.0 * x))

@njit
def var_ex(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    p0 = math.sin(theta + r)
    p1 = math.cos(theta - r)
    return r * (p0**3 + p1**3), r * (p0**3 - p1**3)

@njit
def apply_symmetry_points(x, y, sym_type, sym_segs):
    """
    Returns a list of symmetric coordinates for a given point (x, y).
    sym_type: 0=None, 1=X, 2=Y, 3=Kaleidoscope
    """
    # Default: just the point itself
    points = [(x, y)]
    
    if sym_type == 1: # X Symmetry (Mirror Vertical Axis)
        points.append((-x, y))
    elif sym_type == 2: # Y Symmetry (Mirror Horizontal Axis)
        points.append((x, -y))
    elif sym_type == 3: # Kaleidoscope (Rotational)
        # Generate rotated points
        angle_step = 2.0 * math.pi / sym_segs
        for i in range(1, sym_segs):
            theta = i * angle_step
            new_x = x * math.cos(theta) - y * math.sin(theta)
            new_y = x * math.sin(theta) + y * math.cos(theta)
            points.append((new_x, new_y))
            
    return points

@njit
def render_layer(transforms, iterations, width, height, zoom, skip, sym_type, sym_segs):
    density = np.zeros((height, width), dtype=np.float32)
    # Store color index (0.0 to 1.0) instead of RGB
    color_idx_acc = np.zeros((height, width), dtype=np.float32)
    
    x, y = 0.0, 0.0
    c_idx = 0.5 # Start in middle of palette
    
    n_trans = len(transforms)
    
    # Pre-calculate cumulative weights
    weights = transforms[:, 6]
    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]

    for i in range(iterations):
        # --- 1. Weighted Selection ---
        rw = np.random.random() * total_weight
        t_idx = 0
        for k in range(n_trans):
            if rw < cum_weights[k]:
                t_idx = k
                break
        
        t = transforms[t_idx]
        a, b, c, d, e, f = t[0:6]
        var_idx = int(t[7])
        
        # New: Color Index and Speed from transform
        t_c_idx = t[8]    # Transform's palette index (0-1)
        t_c_speed = t[9]  # Transform's color speed (0-1)
        
        # Post-affine
        pa, pb, pc, pd, pe, pf = t[10:16]

        # --- 2. Apply Affine ---
        x1 = a * x + b * y + c
        y1 = d * x + e * y + f

        # --- 3. Variations ---
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
        elif var_idx == 12: vx, vy = var_bent(x1, y1)
        elif var_idx == 13: vx, vy = var_popcorn(x1, y1)
        else: vx, vy = var_ex(x1, y1)

        # --- 4. Post-Affine ---
        x = pa * vx + pb * vy + pc
        y = pd * vx + pe * vy + pf
        
        # --- 5. Color Accumulation ---
        # Interpolate current color index towards transform's color index
        c_idx = (c_idx * (1.0 - t_c_speed) + t_c_idx * t_c_speed)
        # Clamp index
        if c_idx < 0: c_idx = 0
        if c_idx > 1: c_idx = 1

        # --- 6. Plot with Symmetry ---
        if i > skip:
            # Get all symmetric points (including original)
            sym_points = apply_symmetry_points(x, y, sym_type, sym_segs)
            
            for (px, py) in sym_points:
                ix = int((px * zoom + 1.0) * width / 2.0)
                iy = int((py * zoom + 1.0) * height / 2.0)

                if 0 <= ix < width and 0 <= iy < height:
                    density[iy, ix] += 1.0
                    color_idx_acc[iy, ix] += c_idx

    return density, color_idx_acc

@njit(parallel=True)
def render_all_layers_parallel(all_transforms, iterations, width, height, zoom, skip, sym_type, sym_segs):
    num_layers = all_transforms.shape[0]
    all_dens = np.empty((num_layers, height, width), dtype=np.float32)
    all_cidx = np.empty((num_layers, height, width), dtype=np.float32)
    
    for i in prange(num_layers):
        d, c = render_layer(all_transforms[i], iterations, width, height, zoom, skip, sym_type, sym_segs)
        all_dens[i] = d
        all_cidx[i] = c
        
    return all_dens, all_cidx

# --- Main Generator Class ---
class FlameFractal:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_variations = 15

    def build_transforms(self):
        """
        Builds transforms with Color Indexing for Palettes.
        Slot 8: Color Index (0.0 to 1.0) - Position in the palette.
        Slot 9: Color Speed (0.0 to 1.0) - How much it pulls towards that color.
        """
        ts = []
        for _ in range(self.cfg.transforms):
            # Affine
            a = np.random.uniform(-1.5, 1.5)
            b = np.random.uniform(-1.5, 1.5)
            d = np.random.uniform(-1.5, 1.5)
            e = np.random.uniform(-1.5, 1.5)
            
            det = a * e - b * d
            mag = (a*a + b*b + d*d + e*e)
            
            if mag < 0.5 or abs(det) < 0.01:
                a = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                b = np.random.uniform(-0.5, 0.5)
                d = np.random.uniform(-0.5, 0.5)
                e = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)

            c = np.random.uniform(-0.5, 0.5)
            f = np.random.uniform(-0.5, 0.5)

            weight = np.random.uniform(0.1, 1.0)
            var_idx = np.random.randint(0, self.num_variations)
            
            # Palette Logic: Index and Speed
            color_idx = np.random.uniform(0.0, 1.0) # Position on the palette strip
            color_speed = np.random.uniform(0.2, 0.8) # How strongly it defines the color
            
            # Post-affine
            pa, pe = 1.0, 1.0
            pb, pd = 0.0, 0.0
            pc, pf = 0.0, 0.0
            
            if np.random.rand() > 0.5:
                pa = np.random.uniform(0.8, 1.2)
                pe = np.random.uniform(0.8, 1.2)
                pb = np.random.uniform(-0.2, 0.2)
                pd = np.random.uniform(-0.2, 0.2)

            # Slots: 0-5 Affine, 6 Weight, 7 Var, 8 C-Idx, 9 C-Speed, 10-15 Post
            ts.append((a, b, c, d, e, f, weight, var_idx, color_idx, color_speed, pa, pb, pc, pd, pe, pf))
            
        return np.array(ts, dtype=np.float32)

    def apply_final_symmetry(self, img: np.ndarray) -> np.ndarray:
        """Fallback or optional post-process symmetry (mostly deprecated by iterative symmetry)."""
        # Keeping this empty as symmetry is now handled in render_layer
        return img

    def art_filter(self, img: np.ndarray) -> np.ndarray:
        style = self.cfg.art_style
        if style is None: return img
        try:
            pil = Image.fromarray(img)
            if style == "pointillism":
                dots = Image.new("RGB", pil.size)
                draw = ImageDraw.Draw(dots)
                pixels = np.array(pil)
                h, w = pixels.shape[:2]
                for _ in range((w * h) // 400):
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
        except Exception as e:
            logger.error(f"Art filter error: {e}")
            return img

    def render_single(self):
        try:
            # Prepare transforms for all layers
            all_transforms = np.empty((self.cfg.layers, self.cfg.transforms, 16), dtype=np.float32)
            for i in range(self.cfg.layers):
                all_transforms[i] = self.build_transforms()

            # Map string symmetry to int for Numba
            sym_map = {None: 0, "x": 1, "y": 2, "kaleidoscope": 3}
            sym_type = sym_map.get(self.cfg.symmetry, 0)
            
            zoom = np.float32(self.cfg.zoom)
            
            # Render
            all_dens, all_cidx = render_all_layers_parallel(
                all_transforms, self.cfg.iterations, self.cfg.width, self.cfg.height, zoom, self.cfg.skip,
                sym_type, self.cfg.symmetry_segments
            )
            
            # Palette Setup
            palette = self.cfg.palette
            if self.cfg.use_palette and palette is not None and len(palette) > 0:
                palette = np.array(palette, dtype=np.float32)
            else:
                # Fallback: Grayscale palette (or generate a random one on the fly)
                palette = np.linspace(0, 1, 256).repeat(3).reshape(-1, 3).astype(np.float32)

            p_size = len(palette)
            
            combined = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.float32)
            
            # Layer Compositing with Palette Mapping
            for i in range(self.cfg.layers):
                dens = all_dens[i]
                cidx = all_cidx[i]
                
                # Normalize Density (Logarithmic)
                max_d = np.max(dens)
                if max_d > 0:
                    norm_d = (np.log(dens + 1.0) / np.log(max_d + 1.0)).astype(np.float32)
                else:
                    norm_d = dens
                
                # Average Color Index per pixel
                # cidx is sum of indices, dens is count. avg_idx = sum / count
                avg_idx = np.where(dens > 0, cidx / (dens + 1e-6), 0)
                
                # LAYER IMPORTANCE: Offset palette index per layer
                # This ensures different layers use different parts of the palette spectrum
                layer_offset = (i / self.cfg.layers) 
                avg_idx_shifted = (avg_idx + layer_offset) % 1.0
                
                # Map Index to Palette Coordinates
                pal_indices = (avg_idx_shifted * (p_size - 1)).astype(int)
                pal_indices = np.clip(pal_indices, 0, p_size - 1)
                
                # Lookup Colors
                # Using np.take to map 2D index array to 3D color array
                colors = palette[pal_indices]
                
                # Apply Density to Color (Brightness)
                layer_rgb = colors * norm_d[..., None] * 255.0
                
                # Additive Blending (Light)
                combined += layer_rgb
            
            # Post-processing
            # Average the layers (or just cap at 255 for additive light bloom)
            combined = np.clip(combined, 0, 255).astype(np.uint8)
            
            # Background
            bg_color = np.array(self.cfg.background_color, dtype=np.uint8)
            if not np.array_equal(bg_color, [0, 0, 0]):
                mask = np.any(combined > 0, axis=2)
                bg = np.tile(bg_color, (self.cfg.height, self.cfg.width, 1))
                combined = np.where(mask[..., None], combined, bg)
            
            combined = self.art_filter(combined)
            return combined
        except Exception as e:
            logger.error(f"Render Single Error: {e}")
            raise

    def render_animation(self, filename):
        # Animation logic using the same principles
        try:
            frames = []
            base_transforms = self.build_transforms()
            
            sym_map = {None: 0, "x": 1, "y": 2, "kaleidoscope": 3}
            sym_type = sym_map.get(self.cfg.symmetry, 0)
            
            for f in range(self.cfg.frames):
                transforms = base_transforms.copy()
                # Animate weights and color indices slightly
                for t in transforms:
                    t[6] *= random.uniform(0.98, 1.02) # Weight pulse
                    t[8] = (t[8] + random.uniform(-0.05, 0.05)) % 1.0 # Color drift
                
                dens, cidx = render_layer(
                    transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
                    np.float32(self.cfg.zoom), self.cfg.skip, sym_type, self.cfg.symmetry_segments
                )
                
                # Process frame (simplified version of render_single)
                max_d = np.max(dens)
                norm_d = (np.log(dens + 1.0) / np.log(max_d + 1.0)) if max_d > 0 else dens
                avg_idx = np.where(dens > 0, cidx / (dens + 1e-6), 0)
                
                # Palette Lookup
                p_data = self.cfg.palette if self.cfg.use_palette else np.linspace(0, 1, 256).repeat(3).reshape(-1, 3)
                p_data = np.array(p_data, dtype=np.float32)
                p_indices = np.clip((avg_idx * (len(p_data) - 1)).astype(int), 0, len(p_data) - 1)
                colors = p_data[p_indices]
                
                rgb = (colors * norm_d[..., None] * 255.0).astype(np.uint8)
                frames.append(rgb)
                logger.info(f"Rendered frame {f+1}/{self.cfg.frames}")
            
            imageio.mimsave(filename, frames, duration=0.1, loop=0)
            return filename
        except Exception as e:
            logger.error(f"Render Animation Error: {e}")
            raise

# --- Palette Utilities ---

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
                    # Normalize palette to 0.0-1.0 float for rendering math
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