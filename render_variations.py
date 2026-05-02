import math
import numpy as np
from numba import njit

# --- Basic Variations ---

@njit
def var_linear(x, y):
    return x, y

@njit
def var_sinusoidal(x, y):
    return math.sin(x), math.sin(y)

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
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    return theta / math.pi * math.sin(math.pi * r), theta / math.pi * math.cos(math.pi * r)

@njit
def var_fisheye(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    return 2.0 * x / (r + 1.0), 2.0 * y / (r + 1.0)

@njit
def var_bent(x, y):
    if x >= 0 and y >= 0: return x, y
    if x < 0 and y >= 0: return 2.0 * x, y
    if x >= 0 and y < 0: return x, y / 2.0
    return 2.0 * x, y / 2.0

@njit
def var_butterfly(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    # Butterfly variation approximation
    return (math.sin(r) * (math.cos(theta * r))), (math.cos(r) * (math.sin(theta * r)))

@njit
def var_ex(x, y):
    r = math.sqrt(x * x + y * y) + 1e-6
    theta = math.atan2(y, x)
    p0 = math.sin(theta + r)
    p1 = math.cos(theta - r)
    return r * (p0**3 + p1**3), r * (p0**3 - p1**3)

# --- Complex Variations ---

@njit
def var_julia(x, y):
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    # Random rotation component for Julia set effect
    if np.random.random() < 0.5:
        return r**0.5 * math.cos(theta / 2.0), r**0.5 * math.sin(theta / 2.0)
    else:
        return r**0.5 * math.cos(theta / 2.0 + math.pi), r**0.5 * math.sin(theta / 2.0 + math.pi)

@njit
def var_bipolar(x, y):
    # Bipolar variation
    a = 0.5 * math.log((x + 1e-6)**2 + (y + 1.0)**2)
    b = math.atan2(2.0 * y, x**2 + y**2 - 1.0)
    return a, b

@njit
def var_pdj(x, y):
    # Peter De Jong attractor style
    p1, p2, p3, p4 = 0.7, 1.0, 0.5, 0.4 # Fixed params for simplicity
    return math.sin(p1 * y) - math.cos(p2 * x), math.sin(p3 * x) - math.cos(p4 * y)

@njit
def var_popcorn(x, y):
    c = 1.0 
    f = 1.0
    return x + c * math.sin(math.tan(3.0 * y)), y + f * math.sin(math.tan(3.0 * x))

@njit
def var_blur(x, y):
    angle = np.random.random() * 2.0 * math.pi
    rad = np.random.random() * 0.5
    return rad * math.cos(angle), rad * math.sin(angle)

@njit
def var_gaussian(x, y):
    # Gaussian blur with drift
    sig = 0.3
    return x + np.random.normal(0, sig), y + np.random.normal(0, sig)

@njit
def apply_symmetry_points(x, y, sym_type, sym_segs):
    points = [(x, y)]
    
    if sym_type == 1: # X
        points.append((-x, y))
    elif sym_type == 2: # Y
        points.append((x, -y))
    elif sym_type == 3: # Rotational
        angle_step = 2.0 * math.pi / sym_segs
        for i in range(1, sym_segs):
            theta = i * angle_step
            new_x = x * math.cos(theta) - y * math.sin(theta)
            new_y = x * math.sin(theta) + y * math.cos(theta)
            points.append((new_x, new_y))
            
    return points