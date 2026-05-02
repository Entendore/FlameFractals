import logging
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

logger = logging.getLogger(__name__)

def apply_art_filter(img: np.ndarray, style: str, blur_radius: float = 0.0) -> np.ndarray:
    pil = Image.fromarray(img)
    
    # 1. Apply Gaussian Blur if radius > 0
    if blur_radius > 0.0:
        try:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except Exception as e:
            logger.error(f"Blur filter error: {e}")

    # 2. Apply Art Style
    if style is None: 
        return np.array(pil)
        
    try:
        if style == "pointillism":
            dots = Image.new("RGB", pil.size)
            draw = ImageDraw.Draw(dots)
            pixels = np.array(pil)
            h, w = pixels.shape[:2]
            for _ in range((w * h) // 200):
                x, y = random.randint(0, w - 1), random.randint(0, h - 1)
                patch = pixels[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)]
                color = tuple(np.mean(patch.reshape(-1, 3), axis=0).astype(int))
                draw.ellipse([x-3, y-3, x+3, y+3], fill=color)
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
        return np.array(pil)