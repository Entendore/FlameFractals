import logging
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import fractal_engine as engine
from config import Config

logger = logging.getLogger(__name__)

class GeneratorThread(QThread):
    progress = Signal(str)
    finished = Signal(object) # QImage or str (filepath)
    error = Signal(str)

    def __init__(self, config: Config, save_path=None):
        super().__init__()
        self.config = config
        self.save_path = save_path
        self.is_anim = config.frames > 0

    def run(self):
        try:
            self.progress.emit("Initializing Engine...")
            fractal = engine.FlameFractal(self.config)
            
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
                
                h, w, c = img_array.shape
                bytes_per_line = 3 * w
                q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                
                if self.save_path:
                    q_img.save(self.save_path)
                    self.progress.emit("Image Saved.")
                
                self.finished.emit(q_img)

        except Exception as e:
            logger.exception("Thread Error")
            self.error.emit(str(e))