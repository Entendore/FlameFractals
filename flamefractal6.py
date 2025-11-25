import numpy as np
from PIL import Image
import random
import string
import os
from numba import njit
import math

# ---------------- Variations ----------------
@njit
def variation_linear(x, y): return x, y
@njit
def variation_sinusoidal(x, y): return math.sin(x), math.sin(y)
@njit
def variation_spherical(x, y):
    r2 = x*x + y*y + 1e-6
    return x/r2, y/r2
@njit
def variation_swirl(x, y):
    r2 = x*x + y*y
    return x*math.sin(r2) - y*math.cos(r2), x*math.cos(r2) + y*math.sin(r2)
@njit
def variation_horseshoe(x, y):
    r = math.sqrt(x*x + y*y) + 1e-6
    return (x - y)*(x + y)/r, 2*x*y/r
@njit
def variation_polar(x, y):
    r = math.sqrt(x*x + y*y) + 1e-6
    theta = math.atan2(y, x)
    return theta / math.pi, r - 1
@njit
def variation_handkerchief(x, y):
    r = math.sqrt(x*x + y*y)
    theta = math.atan2(y, x)
    return r * math.sin(theta + r), r * math.cos(theta - r)
@njit
def variation_heart(x, y):
    r = math.sqrt(x*x + y*y)
    theta = math.atan2(y, x)
    return r * math.sin(theta*r), -r * math.cos(theta*r)
@njit
def variation_disk(x, y):
    r = math.sqrt(x*x + y*y)/math.pi
    theta = math.atan2(y, x)
    return theta/math.pi*math.sin(math.pi*r), theta/math.pi*math.cos(math.pi*r)
@njit
def variation_fisheye(x, y):
    r = math.sqrt(x*x + y*y) + 1e-6
    return x/r, y/r

# ---------------- Fractal Iteration ----------------
@njit(parallel=True)
def fractal_iterate(transforms_array, iterations, width, height, zoom):
    density = np.zeros((height, width), dtype=np.float32)
    color_accum = np.zeros((height, width, 3), dtype=np.float32)
    x, y = 0.0, 0.0
    n_trans = transforms_array.shape[0]

    for i in range(iterations):
        t_idx = np.random.randint(0, n_trans)
        t = transforms_array[t_idx]
        a,b,c,d,e,f = t[0], t[1], t[2], t[3], t[4], t[5]
        weight = t[6]
        variation_id = int(t[7])
        color = (t[8], t[9], t[10])

        x1 = a*x + b*y + c
        y1 = d*x + e*y + f

        # Numba-compatible variation selection
        if variation_id == 0:
            x2, y2 = variation_linear(x1, y1)
        elif variation_id == 1:
            x2, y2 = variation_sinusoidal(x1, y1)
        elif variation_id == 2:
            x2, y2 = variation_spherical(x1, y1)
        elif variation_id == 3:
            x2, y2 = variation_swirl(x1, y1)
        elif variation_id == 4:
            x2, y2 = variation_horseshoe(x1, y1)
        elif variation_id == 5:
            x2, y2 = variation_polar(x1, y1)
        elif variation_id == 6:
            x2, y2 = variation_handkerchief(x1, y1)
        elif variation_id == 7:
            x2, y2 = variation_heart(x1, y1)
        elif variation_id == 8:
            x2, y2 = variation_disk(x1, y1)
        else:
            x2, y2 = variation_fisheye(x1, y1)

        x = x2 * weight
        y = y2 * weight

        ix = int((x*zoom + 1) * width / 2)
        iy = int((y*zoom + 1) * height / 2)

        if 0 <= ix < width and 0 <= iy < height:
            density[iy, ix] += 1.0
            for j in range(3):
                color_accum[iy, ix, j] += color[j]

    return density, color_accum

# ---------------- Flame Fractal ----------------
def generate_fractal(width=1000, height=1000, iterations=100_000_000, transforms=30, zoom=1.4):
    transforms_array = []
    for _ in range(transforms):
        a,b,c,d,e,f = np.random.uniform(-1,1,6)
        weight = np.random.uniform(0.5,1.5)
        variation_id = np.random.randint(0, 10)  # 10 variations
        color = np.random.rand(3)  # random RGB
        transforms_array.append([a,b,c,d,e,f,weight,variation_id,color[0],color[1],color[2]])
    transforms_array = np.array(transforms_array, dtype=np.float32)

    density, color_accum = fractal_iterate(transforms_array, iterations, width, height, zoom)

    max_density = np.max(density)
    safe_density = np.log(density + 1.0) / (math.log(max_density + 1.0) + 1e-12)
    norm_color = np.where(density[..., None] > 0, color_accum / (density[..., None] + 1e-6), 0)
    norm_color = np.clip(norm_color, 0, 1)
    rgb = (norm_color * safe_density[..., None]) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)

# ---------------- Run ----------------
if __name__ == "__main__":

    fractal_img = generate_fractal(width=1000, height=1000)
    filename = "fractal.png"
    Image.fromarray(fractal_img, mode='RGB').save(filename)
    print(f"Fractal saved as {filename}")
