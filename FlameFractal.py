import numpy as np
from PIL import Image
import random
import string
import os
from pathlib import Path

# ---------------- Configuration ----------------
class Config:
    def __init__(self,
                 width=800,
                 height=800,
                 iterations=5_000_000,
                 transforms=35,
                 zoom=1.1,
                 skip=100,
                 gamma=2.2,
                 palette=None,
                 use_palette=True):
        self.width = width
        self.height = height
        self.iterations = iterations
        self.transforms = transforms
        self.zoom = zoom
        self.skip = skip
        self.gamma = gamma
        self.palette = palette        # numpy array Nx3 of RGB values [0,1]
        self.use_palette = use_palette  # True = use palette, False = random colors

# ---------------- Flame Fractal ----------------
class FlameFractal:
    def __init__(self, config: Config):
        self.config = config
        self.VARIATIONS = [
            self.variation_linear,
            self.variation_sinusoidal,
            self.variation_spherical,
            self.variation_swirl,
            self.variation_horseshoe,
            self.variation_polar,
            self.variation_handkerchief,
            self.variation_heart,
            self.variation_disk,
            self.variation_blur,
            self.variation_fisheye,
        ]

    # ---------------- Variations ----------------
    @staticmethod
    def variation_linear(x, y): return x, y
    @staticmethod
    def variation_sinusoidal(x, y): return np.sin(x), np.sin(y)
    @staticmethod
    def variation_spherical(x, y):
        r2 = x*x + y*y + 1e-6
        return x/r2, y/r2
    @staticmethod
    def variation_swirl(x, y):
        r2 = x*x + y*y
        return x*np.sin(r2) - y*np.cos(r2), x*np.cos(r2) + y*np.sin(r2)
    @staticmethod
    def variation_horseshoe(x, y):
        r = np.sqrt(x*x + y*y) + 1e-6
        return (x - y)*(x + y)/r, 2*x*y/r
    @staticmethod
    def variation_polar(x, y):
        r = np.sqrt(x*x + y*y) + 1e-6
        theta = np.arctan2(y, x)
        return theta/np.pi, r-1
    @staticmethod
    def variation_handkerchief(x, y):
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        return r*np.sin(theta+r), r*np.cos(theta-r)
    @staticmethod
    def variation_heart(x, y):
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        return r*np.sin(theta*r), -r*np.cos(theta*r)
    @staticmethod
    def variation_disk(x, y):
        r = np.sqrt(x*x + y*y)/np.pi
        theta = np.arctan2(y, x)
        return theta/np.pi*np.sin(np.pi*r), theta/np.pi*np.cos(np.pi*r)
    @staticmethod
    def variation_blur(x, y):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, 1)
        return x+radius*np.cos(angle), y+radius*np.sin(angle)
    @staticmethod
    def variation_fisheye(x, y):
        r = np.sqrt(x*x + y*y) + 1e-6
        return x/r, y/r

    # ---------------- Color ----------------
    def random_color(self):
        cfg = self.config
        if cfg.use_palette and cfg.palette is not None:
            return random.choice(cfg.palette)
        return np.array([random.random(), random.random(), random.random()])

    # ---------------- Transform ----------------
    def generate_transform(self):
        a, b, c = np.random.uniform(-1, 1, 3)
        d, e, f = np.random.uniform(-1, 1, 3)
        variation = random.choice(self.VARIATIONS)
        weight = np.random.uniform(0.5, 1.5)
        color = self.random_color()
        return (a, b, c, d, e, f, variation, weight, color)

    # ---------------- Single Layer Generation ----------------
    def generate_layer(self):
        cfg = self.config
        transforms = [self.generate_transform() for _ in range(cfg.transforms)]
        
        density = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        color_accum = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
        x, y = 0.0, 0.0

        for i in range(cfg.iterations):
            t = random.choice(transforms)
            a, b, c, d, e, f, var_func, weight, t_color = t

            x1 = a*x + b*y + c
            y1 = d*x + e*y + f
            x2, y2 = var_func(x1, y1)
            x, y = x2*weight, y2*weight

            ix = int((x*cfg.zoom + 1)*cfg.width/2)
            iy = int((y*cfg.zoom + 1)*cfg.height/2)

            if 0 <= ix < cfg.width and 0 <= iy < cfg.height:
                if i > cfg.skip:
                    density[iy, ix] += 1
                    color_accum[iy, ix] += t_color

        max_density = np.max(density)
        safe_density = np.log(density + 1)/np.log(max_density + 1 + 1e-6)

        with np.errstate(invalid='ignore'):
            norm_color = np.where(
                density[..., None] > 0,
                color_accum/(density[..., None] + 1e-6),
                0
            )

        norm_color = np.clip(norm_color, 0, 1)
        norm_color = norm_color ** (1/cfg.gamma)
        rgb = (norm_color * safe_density[..., None]) * 255
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # ---------------- Single Image Generation ----------------
    def generate(self):
        return self.generate_layer()

# ---------------- Palette Loader ----------------
def load_palettes(folder_path):
    palettes = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        try:
            if file.lower().endswith(".png"):
                img = Image.open(path).convert("RGB")
                palette = np.array(img)[0, :, :]/255.0
                palettes.append(palette)
            elif file.lower().endswith(".txt"):
                palette = []
                with open(path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            palette.append([int(p)/255.0 for p in parts])
                palettes.append(np.array(palette))
        except Exception as e:
            print(f"Failed to load palette {file}: {e}")
    return palettes

# ---------------- Main Function ----------------
def main():
    width, height, skip = 1024, 1024, 100
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    palette_dir = parent_dir / "palettes"
    palettes = load_palettes(palette_dir)
    if not palettes:
        palettes = [None]  # fallback to random colors

    # Choose whether to use palette or random colors globally
    use_palette = False 

    for i in range(6):
        params = {
            "iterations": random.randint(2_000_000, 5_000_000),
            "transforms": random.randint(5, 15),
            "zoom": round(random.uniform(0.5, 2.0), 2),
            "gamma": round(random.uniform(1.5, 2.5), 2),
            "palette": random.choice(palettes),
            "use_palette": use_palette
        }

        config = Config(width=width, height=height, skip=skip, **params)
        fractal = FlameFractal(config)
        texture = fractal.generate()
        
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        filename = f"flame_texture_{random_str}.png"
        Image.fromarray(texture, mode='RGB').save(filename)
        print(f"Saved {filename} (use_palette={use_palette}) with params: {params}")

if __name__ == "__main__":
    main()
