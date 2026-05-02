import numpy as np
from numba import njit, prange
import render_variations as rv

@njit
def apply_variation_combo(x, y, var_indices, var_weights):
    vx, vy = 0.0, 0.0
    for i in range(4):
        idx = var_indices[i]
        w = var_weights[i]
        if w == 0.0: continue
        
        if idx == 0: tx, ty = rv.var_linear(x, y)
        elif idx == 1: tx, ty = rv.var_sinusoidal(x, y)
        elif idx == 2: tx, ty = rv.var_spherical(x, y)
        elif idx == 3: tx, ty = rv.var_swirl(x, y)
        elif idx == 4: tx, ty = rv.var_horseshoe(x, y)
        elif idx == 5: tx, ty = rv.var_polar(x, y)
        elif idx == 6: tx, ty = rv.var_handkerchief(x, y)
        elif idx == 7: tx, ty = rv.var_heart(x, y)
        elif idx == 8: tx, ty = rv.var_disk(x, y)
        elif idx == 9: tx, ty = rv.var_fisheye(x, y)
        elif idx == 10: tx, ty = rv.var_bent(x, y)
        elif idx == 11: tx, ty = rv.var_butterfly(x, y)
        elif idx == 12: tx, ty = rv.var_ex(x, y)
        elif idx == 13: tx, ty = rv.var_julia(x, y)
        elif idx == 14: tx, ty = rv.var_bipolar(x, y)
        elif idx == 15: tx, ty = rv.var_pdj(x, y)
        elif idx == 16: tx, ty = rv.var_popcorn(x, y)
        elif idx == 17: tx, ty = rv.var_blur(x, y)
        elif idx == 18: tx, ty = rv.var_gaussian(x, y)
        else: tx, ty = x, y
        
        vx += tx * w
        vy += ty * w
        
    return vx, vy

@njit
def render_layer(transforms, iterations, width, height, zoom, skip, sym_type, sym_segs):
    density = np.zeros((height, width), dtype=np.float32)
    color_idx_acc = np.zeros((height, width), dtype=np.float32)
    
    # FIX: Initialize with a random point in a larger range to break center concentration
    x = np.random.uniform(-2.0, 2.0)
    y = np.random.uniform(-2.0, 2.0)
    c_idx = 0.5
    
    n_trans = len(transforms)
    
    weights = transforms[:, 6]
    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]

    for i in range(iterations):
        # --- 1. Select Transform ---
        rw = np.random.random() * total_weight
        t_idx = 0
        for k in range(n_trans):
            if rw < cum_weights[k]:
                t_idx = k
                break
        
        t = transforms[t_idx]
        
        a, b, c_aff, d, e, f = t[0:6]
        t_c_idx = t[7]
        t_c_speed = t[8]
        pa, pb, pc, pd, pe, pf = t[9:15]
        
        var_indices = t[15:19].astype(np.int64)
        var_weights = t[19:23]

        # --- 2. Affine Transform ---
        x1 = a * x + b * y + c_aff
        y1 = d * x + e * y + f

        # --- 3. Variation Combo ---
        vx, vy = apply_variation_combo(x1, y1, var_indices, var_weights)

        # --- 4. Post-Affine ---
        x = pa * vx + pb * vy + pc
        y = pd * vx + pe * vy + pf
        
        # --- Stability & Distribution Check ---
        # If points collapse to center (0,0) or fly away, reset them randomly.
        # This forces the distribution to spread out if the attractor tries to shrink.
        rad_sq = x*x + y*y
        if rad_sq > 100.0 or rad_sq < 1e-6 or not np.isfinite(rad_sq):
             # Reset to a random position on the screen bounds
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            continue

        # --- 5. Color Accumulation ---
        c_idx = (c_idx + t_c_idx) * t_c_speed
        if c_idx < 0: c_idx = 0
        if c_idx > 1: c_idx = 1

        # --- 6. Plot ---
        if i > skip:
            sym_points = rv.apply_symmetry_points(x, y, sym_type, sym_segs)
            
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