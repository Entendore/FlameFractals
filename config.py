import random
import numpy as np

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
                 vibrancy=1.0,
                 blur_radius=0.0): # Added blur_radius

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
        self.blur_radius = blur_radius

        # Robust Seeding
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)