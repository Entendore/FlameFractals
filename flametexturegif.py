import numpy as np
from PIL import Image
import random
import string
import imageio

class Config:
    def __init__(self,
                 width=800,
                 height=800,
                 iterations=2_000_000,
                 transforms=15,
                 zoom=1.2,
                 skip=100,
                 gamma=2.2,
                 layers=3,
                 frames=30):
        self.width = width
        self.height = height
        self.iterations = iterations
        self.transforms = transforms
        self.zoom = zoom
        self.skip = skip
        self.gamma = gamma
        self.layers = layers
        self.frames = frames

class FlameFractal:
    def __init__(self, config: Config):
        self.config = config
        self.VARIATIONS = [
            self.variation_linear, self.variation_sinusoidal,
            self.variation_spherical, self.variation_swirl,
            self.variation_horseshoe, self.variation_polar,
            self.variation_handkerchief, self.variation_heart,
            self.variation_disk, self.variation_blur, self.variation_fisheye,
        ]

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
        r = np.sqrt(x*x + y*y)+1e-6
        return (x-y)*(x+y)/r, 2*x*y/r
    @staticmethod
    def variation_polar(x, y):
        r = np.sqrt(x*x + y*y)+1e-6
        theta = np.arctan2(y,x)
        return theta/np.pi, r-1
    @staticmethod
    def variation_handkerchief(x, y):
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y,x)
        return r*np.sin(theta+r), r*np.cos(theta-r)
    @staticmethod
    def variation_heart(x, y):
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y,x)
        return r*np.sin(theta*r), -r*np.cos(theta*r)
    @staticmethod
    def variation_disk(x, y):
        r = np.sqrt(x*x + y*y)/np.pi
        theta = np.arctan2(y,x)
        return theta/np.pi*np.sin(np.pi*r), theta/np.pi*np.cos(np.pi*r)
    @staticmethod
    def variation_blur(x, y):
        angle = np.random.uniform(0,2*np.pi)
        radius = np.random.uniform(0,1)
        return x+radius*np.cos(angle), y+radius*np.sin(angle)
    @staticmethod
    def variation_fisheye(x, y):
        r = np.sqrt(x*x + y*y)+1e-6
        return x/r, y/r

    @staticmethod
    def random_color():
        return np.array([random.random(), random.random(), random.random()])

    def generate_transform(self):
        a,b,c = np.random.uniform(-1,1,3)
        d,e,f = np.random.uniform(-1,1,3)
        variation = random.choice(self.VARIATIONS)
        weight = np.random.uniform(0.5,1.5)
        color = self.random_color()
        return [a,b,c,d,e,f,variation,weight,color]

    def generate_layer(self, transforms):
        cfg = self.config
        density = np.zeros((cfg.height,cfg.width),dtype=np.float32)
        color_accum = np.zeros((cfg.height,cfg.width,3),dtype=np.float32)
        x,y = 0.0,0.0

        for i in range(cfg.iterations):
            t = random.choice(transforms)
            a,b,c,d,e,f,var_func,weight,t_color = t
            x1 = a*x+b*y+c
            y1 = d*x+e*y+f
            x2,y2 = var_func(x1,y1)
            x,y = x2*weight, y2*weight

            ix = int((x*cfg.zoom+1)*cfg.width/2)
            iy = int((y*cfg.zoom+1)*cfg.height/2)
            if 0<=ix<cfg.width and 0<=iy<cfg.height:
                if i>cfg.skip:
                    density[iy,ix]+=1
                    color_accum[iy,ix]+=t_color

        max_density = np.max(density)
        safe_density = np.log(density+1)/np.log(max_density+1+1e-6)
        with np.errstate(invalid='ignore'):
            norm_color = np.where(density[...,None]>0, color_accum/(density[...,None]+1e-6),0)
        norm_color = np.clip(norm_color,0,1)**(1/cfg.gamma)
        rgb = (norm_color*safe_density[...,None])*255
        return np.clip(rgb,0,255).astype(np.uint8)

    def generate_frame(self, transforms):
        # Combine layers
        combined = np.zeros((self.config.height,self.config.width,3),dtype=np.float32)
        for _ in range(self.config.layers):
            layer_rgb = self.generate_layer(transforms).astype(np.float32)
            combined += layer_rgb
        combined = np.clip(combined/self.config.layers,0,255).astype(np.uint8)
        return combined

    def generate_animation(self, filename="flame_animation.gif"):
        transforms = [self.generate_transform() for _ in range(self.config.transforms)]
        frames_list = []
        for frame_idx in range(self.config.frames):
            # Slightly vary weights/colors per frame
            for t in transforms:
                t[7] *= random.uniform(0.95,1.05)  # weight jitter
                t[8] = t[8]*0.9 + self.random_color()*0.1  # color drift
            frame = self.generate_frame(transforms)
            frames_list.append(frame)
            print(f"Generated frame {frame_idx+1}/{self.config.frames}")
        # Save GIF
        imageio.mimsave(filename, frames_list, duration=0.1)
        print(f"Saved animation: {filename}")

def main():
    config = Config(width=600,height=600,iterations=1_500_000,transforms=12,layers=3,frames=40)
    fractal = FlameFractal(config)
    fractal.generate_animation("animated_flame.gif")

if __name__=="__main__":
    main()
