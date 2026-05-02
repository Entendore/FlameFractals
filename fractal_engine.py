import logging
import random
import math
import numpy as np
import imageio
from config import Config
from render_kernel import render_all_layers_parallel
from utils_filters import apply_art_filter

logger = logging.getLogger(__name__)

class FlameFractal:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_variations = 19

    def build_transforms(self):
        ts = []
        for _ in range(self.cfg.transforms):
            # --- Improved Affine Randomization ---
            # We want to avoid shrinking the fractal to a point.
            # We construct the matrix from Rotation, Scale, and Translation.
            
            # 1. Rotation: Random angle
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 2. Scale: Enforce a minimum scale to prevent collapsing to a dot.
            # We want the fractal to fill the view.
            # If scale is < 0.5, the fractal tends to shrink.
            if np.random.random() < 0.8:
                # High probability for "normal" or "expanding" scale
                scale = np.random.uniform(0.7, 1.5)
            else:
                # Small chance for detail/shrink, but not too small
                scale = np.random.uniform(0.4, 0.7)

            cos_a = math.cos(angle) * scale
            sin_a = math.sin(angle) * scale
            
            a = cos_a
            b = sin_a
            d = -sin_a
            e = cos_a
            
            # Add slight noise to break perfect symmetry
            a += np.random.uniform(-0.1, 0.1)
            b += np.random.uniform(-0.1, 0.1)
            d += np.random.uniform(-0.1, 0.1)
            e += np.random.uniform(-0.1, 0.1)

            # 3. Translation (Position)
            # Ensure the fractal moves around the screen, not just centered at 0
            # Increase the range of translation to push points outward
            c = np.random.uniform(-1.0, 1.0)
            f = np.random.uniform(-1.0, 1.0)

            weight = np.random.uniform(0.2, 1.0)
            
            # --- Color ---
            color_idx = np.random.uniform(0.0, 1.0)
            color_speed = np.random.uniform(0.2, 0.8)
            
            # --- Post Affine ---
            # Keep post-affine simple or slightly zoomed out
            pa, pe = 1.0, 1.0
            pb, pd, pc, pf = 0.0, 0.0, 0.0, 0.0
            
            # --- Variations ---
            var_indices = np.zeros(4, dtype=np.int64)
            var_weights = np.zeros(4, dtype=np.float32)
            
            num_vars = np.random.randint(1, 4)
            selected_vars = np.random.choice(self.num_variations, num_vars, replace=False)
            
            temp_weights = np.random.random(num_vars)
            temp_weights /= np.sum(temp_weights)
            
            for i in range(num_vars):
                var_indices[i] = selected_vars[i]
                var_weights[i] = temp_weights[i]
            
            row = [
                a, b, c, d, e, f, weight, 
                color_idx, color_speed, 
                pa, pb, pc, pd, pe, pf
            ]
            row.extend(var_indices.tolist())
            row.extend(var_weights.tolist())
            
            ts.append(row)
            
        return np.array(ts, dtype=np.float32)

    def render_single(self):
        try:
            all_transforms = np.empty((self.cfg.layers, self.cfg.transforms, 23), dtype=np.float32)
            for i in range(self.cfg.layers):
                all_transforms[i] = self.build_transforms()

            sym_map = {None: 0, "x": 1, "y": 2, "kaleidoscope": 3}
            sym_type = sym_map.get(self.cfg.symmetry, 0)
            
            zoom = np.float32(self.cfg.zoom)
            
            all_dens, all_cidx = render_all_layers_parallel(
                all_transforms, self.cfg.iterations, self.cfg.width, self.cfg.height, zoom, self.cfg.skip,
                sym_type, self.cfg.symmetry_segments
            )
            
            palette = self.cfg.palette
            if self.cfg.use_palette and palette is not None and len(palette) > 0:
                palette = np.array(palette, dtype=np.float32)
            else:
                palette = np.linspace(0, 1, 256).repeat(3).reshape(-1, 3).astype(np.float32)

            p_size = len(palette)
            
            dens = np.sum(all_dens, axis=0)
            cidx = np.sum(all_cidx, axis=0)
            
            mask = dens > 0
            avg_idx = np.zeros_like(dens)
            avg_idx[mask] = cidx[mask] / dens[mask]
            
            max_d = np.max(dens)
            if max_d > 0:
                norm_d = np.log1p(dens) / np.log1p(max_d)
            else:
                norm_d = dens
            
            gamma = self.cfg.gamma
            norm_d = np.power(norm_d, 1.0 / gamma)
            
            pal_indices = (avg_idx * (p_size - 1)).astype(int)
            pal_indices = np.clip(pal_indices, 0, p_size - 1)
            
            colors = palette[pal_indices]
            layer_rgb = colors * norm_d[..., None] * 255.0
            
            combined = np.clip(layer_rgb, 0, 255).astype(np.uint8)
            
            bg_color = np.array(self.cfg.background_color, dtype=np.uint8)
            if not np.array_equal(bg_color, [0, 0, 0]):
                mask = dens > 0
                bg = np.tile(bg_color, (self.cfg.height, self.cfg.width, 1))
                combined = np.where(mask[..., None], combined, bg)
            
            combined = apply_art_filter(combined, self.cfg.art_style, self.cfg.blur_radius)
            return combined
        except Exception as e:
            logger.error(f"Render Single Error: {e}")
            raise

    def render_animation(self, filename):
        try:
            frames = []
            base_transforms = self.build_transforms()
            
            sym_map = {None: 0, "x": 1, "y": 2, "kaleidoscope": 3}
            sym_type = sym_map.get(self.cfg.symmetry, 0)
            
            from render_kernel import render_layer
            
            for f in range(self.cfg.frames):
                transforms = base_transforms.copy()
                for t in transforms:
                    t[7] = (t[7] + 0.02) % 1.0 
                
                dens, cidx = render_layer(
                    transforms, self.cfg.iterations, self.cfg.width, self.cfg.height,
                    np.float32(self.cfg.zoom), self.cfg.skip, sym_type, self.cfg.symmetry_segments
                )
                
                max_d = np.max(dens)
                norm_d = np.log1p(dens) / np.log1p(max_d) if max_d > 0 else dens
                norm_d = np.power(norm_d, 1.0/self.cfg.gamma)
                
                avg_idx = np.where(dens > 0, cidx / (dens + 1e-6), 0)
                p_data = self.cfg.palette if self.cfg.use_palette else np.linspace(0, 1, 256).repeat(3).reshape(-1, 3)
                p_data = np.array(p_data, dtype=np.float32)
                p_indices = np.clip((avg_idx * (len(p_data) - 1)).astype(int), 0, len(p_data) - 1)
                colors = p_data[p_indices]
                
                rgb = (colors * norm_d[..., None] * 255.0).astype(np.uint8)
                rgb = apply_art_filter(rgb, self.cfg.art_style, self.cfg.blur_radius)
                frames.append(rgb)
                logger.info(f"Rendered frame {f+1}/{self.cfg.frames}")
            
            imageio.mimsave(filename, frames, duration=0.1, loop=0)
            return filename
        except Exception as e:
            logger.error(f"Render Animation Error: {e}")
            raise