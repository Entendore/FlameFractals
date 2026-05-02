import sys
import logging
import json
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, 
                             QPushButton, QComboBox, QCheckBox, QFileDialog,
                             QGroupBox, QFormLayout, QColorDialog, QMessageBox,
                             QScrollArea, QProgressBar)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor

import utils_palettes
from config import Config
from fractal_engine import FlameFractal

logger = logging.getLogger(__name__)

class GeneratorThread(QThread):
    progress = Signal(str)
    finished = Signal(object) 
    error = Signal(str)

    def __init__(self, config: Config, save_path=None):
        super().__init__()
        self.config = config
        self.save_path = save_path
        self.is_anim = config.frames > 0

    def run(self):
        try:
            self.progress.emit("Initializing Engine...")
            fractal = FlameFractal(self.config)
            
            if self.is_anim:
                self.progress.emit(f"Generating {self.config.frames} Frames...")
                fname = self.save_path or "animation.gif"
                fractal.render_animation(fname)
                self.finished.emit(fname)
            else:
                self.progress.emit("Rendering Layers...")
                img_array = fractal.render_single()
                
                h, w, c = img_array.shape
                bytes_per_line = 3 * w
                q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                
                if self.save_path:
                    q_img.save(self.save_path)
                
                self.finished.emit(q_img)
        except Exception as e:
            logger.exception("Thread Error")
            self.error.emit(str(e))

class FlameStudio(QMainWindow):
    SETTINGS_FILE = "settings.json"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Flame Fractal Studio Pro")
        self.resize(1250, 850)
        
        self.config = Config()
        self.worker = None
        self.last_image = None
        self.palettes = {}
        
        self.batch_running = False
        self.batch_current = 0
        self.batch_total = 0
        self.batch_is_anim = False
        
        self.init_ui()
        self.refresh_palettes()
        self.load_settings()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 12px; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 20px; font-weight: bold; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 5px; min-height: 20px; }
            QPushButton:hover { background-color: #4e4e4e; border: 1px solid #777; }
            QPushButton:pressed { background-color: #606060; }
            QPushButton:disabled { background-color: #2b2b2b; color: #555; }
            QSpinBox, QDoubleSpinBox, QComboBox { background-color: #3c3c3c; border: 1px solid #555; border-radius: 3px; padding: 3px; }
            QScrollBar:vertical { background-color: #2b2b2b; width: 12px; }
            QScrollBar::handle:vertical { background-color: #555; border-radius: 5px; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; background-color: #2b2b2b; }
            QProgressBar::chunk { background-color: #4a86e8; width: 10px; }
        """)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(420)
        left_widget = QWidget()
        form = QVBoxLayout(left_widget)
        form.setSpacing(10)
        
        # Dimensions
        grp_dim = QGroupBox("Dimensions & Quality")
        lay_dim = QFormLayout()
        self.spin_w = QSpinBox(); self.spin_w.setRange(100, 8000); self.spin_w.setValue(self.config.width)
        self.spin_h = QSpinBox(); self.spin_h.setRange(100, 8000); self.spin_h.setValue(self.config.height)
        self.spin_iter = QSpinBox(); self.spin_iter.setRange(10000, 100000000); self.spin_iter.setValue(self.config.iterations); self.spin_iter.setSingleStep(100000)
        self.spin_skip = QSpinBox(); self.spin_skip.setRange(0, 2000); self.spin_skip.setValue(self.config.skip)
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(0, 999999); self.spin_seed.setValue(0); self.spin_seed.setSpecialValueText("Random")
        lay_dim.addRow("Width:", self.spin_w); lay_dim.addRow("Height:", self.spin_h)
        lay_dim.addRow("Iterations:", self.spin_iter); lay_dim.addRow("Skip:", self.spin_skip)
        lay_dim.addRow("Seed:", self.spin_seed)
        grp_dim.setLayout(lay_dim)
        form.addWidget(grp_dim)

        # Transforms
        grp_trans = QGroupBox("Transforms")
        lay_trans = QFormLayout()
        self.spin_num_trans = QSpinBox(); self.spin_num_trans.setRange(1, 100); self.spin_num_trans.setValue(self.config.transforms)
        self.spin_zoom = QDoubleSpinBox(); self.spin_zoom.setRange(0.1, 100.0); self.spin_zoom.setValue(self.config.zoom); self.spin_zoom.setSingleStep(0.1)
        self.spin_layers = QSpinBox(); self.spin_layers.setRange(1, 50); self.spin_layers.setValue(self.config.layers)
        lay_trans.addRow("Num Transforms:", self.spin_num_trans); lay_trans.addRow("Zoom:", self.spin_zoom); lay_trans.addRow("Layers:", self.spin_layers)
        grp_trans.setLayout(lay_trans)
        form.addWidget(grp_trans)

        # Color & Style
        grp_color = QGroupBox("Color & Style")
        lay_color = QFormLayout()
        self.chk_palette = QCheckBox(); self.chk_palette.setChecked(self.config.use_palette)
        self.combo_palette = QComboBox()
        self.btn_gen_palette = QPushButton("Generate New")
        self.btn_gen_palette.clicked.connect(self.generate_new_palette)
        pal_layout = QHBoxLayout(); pal_layout.addWidget(self.combo_palette, 1); pal_layout.addWidget(self.btn_gen_palette)
        self.lbl_palette_preview = QLabel(); self.lbl_palette_preview.setFixedHeight(30); self.lbl_palette_preview.setStyleSheet("background-color: #333; border: 1px solid #555;")
        
        # REMOVED "blur" and "soft_glow" from here
        self.combo_style = QComboBox(); self.combo_style.addItems(["None", "pointillism", "expressionist", "oil", "watercolor"])
        
        self.spin_gamma = QDoubleSpinBox(); self.spin_gamma.setRange(0.1, 5.0); self.spin_gamma.setValue(self.config.gamma); self.spin_gamma.setSingleStep(0.1)
        self.slide_vibrancy = QDoubleSpinBox(); self.slide_vibrancy.setRange(0.0, 1.0); self.slide_vibrancy.setValue(self.config.vibrancy); self.slide_vibrancy.setSingleStep(0.1)
        
        # ADDED Blur Radius Input
        self.spin_blur = QDoubleSpinBox(); self.spin_blur.setRange(0.0, 50.0); self.spin_blur.setValue(0.0); self.spin_blur.setSingleStep(0.5)
        self.spin_blur.setToolTip("Applies Gaussian Blur. 0 = Disabled.")

        self.btn_bg_color = QPushButton("Select Color")
        self.bg_color = self.config.background_color
        self.update_color_btn()
        self.btn_bg_color.clicked.connect(self.select_bg_color)
        
        lay_color.addRow("Use Palette:", self.chk_palette); lay_color.addRow("Palette:", pal_layout)
        lay_color.addRow("Preview:", self.lbl_palette_preview); lay_color.addRow("Art Style:", self.combo_style)
        lay_color.addRow("Blur Radius:", self.spin_blur) # Added
        lay_color.addRow("Gamma:", self.spin_gamma); lay_color.addRow("Vibrancy:", self.slide_vibrancy)
        lay_color.addRow("Background:", self.btn_bg_color)
        grp_color.setLayout(lay_color)
        form.addWidget(grp_color)
        self.combo_palette.currentIndexChanged.connect(self.update_palette_preview)
        
        # Symmetry
        grp_sym = QGroupBox("Symmetry")
        lay_sym = QFormLayout()
        self.combo_sym = QComboBox(); self.combo_sym.addItems(["None", "x", "y", "kaleidoscope"])
        self.spin_sym_seg = QSpinBox(); self.spin_sym_seg.setRange(2, 360); self.spin_sym_seg.setValue(self.config.symmetry_segments)
        lay_sym.addRow("Type:", self.combo_sym); lay_sym.addRow("Segments:", self.spin_sym_seg)
        grp_sym.setLayout(lay_sym)
        form.addWidget(grp_sym)
        
        # Animation
        grp_anim = QGroupBox("Animation (GIF)")
        lay_anim = QFormLayout()
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(0, 120); self.spin_frames.setValue(self.config.frames)
        lay_anim.addRow("Frames (0=Static):", self.spin_frames)
        grp_anim.setLayout(lay_anim)
        form.addWidget(grp_anim)
        
        # Batch
        grp_batch = QGroupBox("Batch Processing")
        lay_batch = QFormLayout()
        self.spin_batch_count = QSpinBox(); self.spin_batch_count.setRange(1, 1000); self.spin_batch_count.setValue(1)
        self.btn_batch = QPushButton("Start Batch")
        self.btn_batch.setStyleSheet("font-weight: bold; background-color: #e6e6fa; color: black;")
        self.btn_batch.clicked.connect(self.start_batch_generation)
        lay_batch.addRow("Count:", self.spin_batch_count); lay_batch.addRow("", self.btn_batch)
        grp_batch.setLayout(lay_batch)
        form.addWidget(grp_batch)

        # Buttons
        self.btn_gen = QPushButton("Generate Single")
        self.btn_gen.setStyleSheet("font-weight: bold; height: 35px; background-color: #4a86e8; color: white;")
        self.btn_gen.clicked.connect(self.start_generation)
        self.btn_save = QPushButton("Save Image"); self.btn_save.clicked.connect(self.save_image); self.btn_save.setEnabled(False)
        self.btn_reset = QPushButton("Reset Defaults"); self.btn_reset.clicked.connect(self.reset_defaults)
        form.addWidget(self.btn_gen); form.addWidget(self.btn_save); form.addWidget(self.btn_reset)
        
        self.status_lbl = QLabel("Ready"); self.status_lbl.setStyleSheet("font-style: italic; color: #aaa;")
        form.addWidget(self.status_lbl)
        
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        form.addWidget(self.progress_bar)
        
        form.addStretch()
        scroll.setWidget(left_widget)
        layout.addWidget(scroll)
        
        self.img_label = QLabel("Output will appear here")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555; font-size: 16px;")
        self.img_label.setMinimumSize(600, 600)
        layout.addWidget(self.img_label, 1)

    def reset_defaults(self):
        self.spin_w.setValue(800); self.spin_h.setValue(800); self.spin_iter.setValue(3000000)
        self.spin_skip.setValue(20); self.spin_seed.setValue(0); self.spin_num_trans.setValue(10)
        self.spin_zoom.setValue(1.2); self.spin_layers.setValue(3); self.chk_palette.setChecked(True)
        self.combo_style.setCurrentIndex(0); self.spin_blur.setValue(0.0) # Reset blur
        self.spin_gamma.setValue(2.2); self.slide_vibrancy.setValue(1.0)
        self.combo_sym.setCurrentIndex(0); self.spin_sym_seg.setValue(6)
        self.spin_frames.setValue(0); self.spin_batch_count.setValue(1)
        self.bg_color = (0, 0, 0); self.update_color_btn()

    def update_palette_preview(self):
        p_name = self.combo_palette.currentText()
        p_data = self.palettes.get(p_name, None)
        if p_data is not None:
            p_display = (p_data * 255).astype(np.uint8)
            w, h = 250, 30; pm = QPixmap(w, h); painter = QPainter(pm)
            step = w / len(p_display)
            for i, col in enumerate(p_display):
                r, g, b = int(col[0]), int(col[1]), int(col[2])
                painter.setPen(QColor(r, g, b)); painter.setBrush(QColor(r, g, b))
                painter.drawRect(int(i * step), 0, int(step) + 1, h)
            painter.end()
            self.lbl_palette_preview.setPixmap(pm)
        else:
            self.lbl_palette_preview.setText("No Palette Selected")

    def closeEvent(self, event):
        if self.batch_running:
            reply = QMessageBox.question(self, 'Batch Running', "A batch is currently generating. Exit anyway?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: event.ignore(); return
            if self.worker: self.worker.terminate()
        self.save_settings(); event.accept()

    def save_settings(self):
        settings = {
            "dimensions": {"width": self.spin_w.value(), "height": self.spin_h.value(), "iterations": self.spin_iter.value(), "skip": self.spin_skip.value(), "seed": self.spin_seed.value()},
            "transforms": {"num_transforms": self.spin_num_trans.value(), "zoom": self.spin_zoom.value(), "layers": self.spin_layers.value()},
            "style": {"use_palette": self.chk_palette.isChecked(), "palette_name": self.combo_palette.currentText(), "art_style": self.combo_style.currentIndex(), "blur_radius": self.spin_blur.value(), "gamma": self.spin_gamma.value(), "vibrancy": self.slide_vibrancy.value(), "bg_color": list(self.bg_color)},
            "symmetry": {"type_index": self.combo_sym.currentIndex(), "segments": self.spin_sym_seg.value()},
            "animation": {"frames": self.spin_frames.value()},
            "batch": {"count": self.spin_batch_count.value()}
        }
        try:
            with open(self.SETTINGS_FILE, "w") as f: json.dump(settings, f, indent=4)
        except Exception as e: logger.error(f"Failed to save settings: {e}")

    def load_settings(self):
        if not os.path.exists(self.SETTINGS_FILE): self.save_settings(); return
        try:
            with open(self.SETTINGS_FILE, "r") as f: settings = json.load(f)
            dims = settings.get("dimensions", {})
            self.spin_w.setValue(dims.get("width", 800)); self.spin_h.setValue(dims.get("height", 800))
            self.spin_iter.setValue(dims.get("iterations", 3000000)); self.spin_skip.setValue(dims.get("skip", 20)); self.spin_seed.setValue(dims.get("seed", 0))
            trans = settings.get("transforms", {})
            self.spin_num_trans.setValue(trans.get("num_transforms", 10)); self.spin_zoom.setValue(trans.get("zoom", 1.2)); self.spin_layers.setValue(trans.get("layers", 3))
            style = settings.get("style", {})
            self.chk_palette.setChecked(style.get("use_palette", True))
            pal_name = style.get("palette_name", "")
            idx = self.combo_palette.findText(pal_name)
            if idx != -1: self.combo_palette.setCurrentIndex(idx)
            self.combo_style.setCurrentIndex(style.get("art_style", 0))
            self.spin_blur.setValue(style.get("blur_radius", 0.0)) # Load blur
            self.spin_gamma.setValue(style.get("gamma", 2.2)); self.slide_vibrancy.setValue(style.get("vibrancy", 1.0))
            bg = style.get("bg_color", [0, 0, 0]); self.bg_color = tuple(bg); self.update_color_btn()
            sym = settings.get("symmetry", {})
            self.combo_sym.setCurrentIndex(sym.get("type_index", 0)); self.spin_sym_seg.setValue(sym.get("segments", 6))
            anim = settings.get("animation", {}); self.spin_frames.setValue(anim.get("frames", 0))
            batch = settings.get("batch", {}); self.spin_batch_count.setValue(batch.get("count", 1))
        except Exception as e: logger.error(f"Failed to load settings: {e}")
        self.update_palette_preview()

    def update_color_btn(self):
        r, g, b = self.bg_color
        self.btn_bg_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {'white' if r+g+b < 400 else 'black'};")
        
    def select_bg_color(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.bg_color = (col.red(), col.green(), col.blue())
            self.update_color_btn()

    def refresh_palettes(self):
        self.palettes = utils_palettes.load_palettes("palettes")
        self.combo_palette.clear()
        self.combo_palette.addItems(self.palettes.keys())
        if not self.palettes: self.combo_palette.addItem("No Palettes Found")

    def generate_new_palette(self):
        utils_palettes.generate_palette_file("palettes")
        self.refresh_palettes()
        self.combo_palette.setCurrentIndex(self.combo_palette.count() - 1)
        
    def get_current_config(self, override_seed=None):
        sym_map = {"None": None, "x": "x", "y": "y", "kaleidoscope": "kaleidoscope"}
        p_name = self.combo_palette.currentText()
        p_data = self.palettes.get(p_name, None)
        if override_seed is not None: final_seed = override_seed
        elif self.spin_seed.value() > 0: final_seed = self.spin_seed.value()
        else: final_seed = random.randint(0, 2**32 - 1)
        
        return Config(
            width=self.spin_w.value(), height=self.spin_h.value(),
            iterations=self.spin_iter.value(), transforms=self.spin_num_trans.value(),
            zoom=self.spin_zoom.value(), skip=self.spin_skip.value(),
            gamma=self.spin_gamma.value(), layers=self.spin_layers.value(),
            frames=self.spin_frames.value(), symmetry=sym_map[self.combo_sym.currentText()],
            symmetry_segments=self.spin_sym_seg.value(), use_palette=self.chk_palette.isChecked(),
            palette=p_data, art_style=self.combo_style.currentText().lower() if self.combo_style.currentText() != "None" else None,
            background_color=self.bg_color, vibrancy=self.slide_vibrancy.value(),
            blur_radius=self.spin_blur.value(), # Pass blur radius
            seed=final_seed
        )

    def start_generation(self):
        if self.worker and self.worker.isRunning(): return
        self.config = self.get_current_config()
        save_path = None
        if self.config.frames > 0:
            output_dir = Path("output"); output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = output_dir / f"single_anim_{timestamp}.gif"
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Animation", str(default_path), "GIF (*.gif)")
            if not save_path: return

        self.worker = GeneratorThread(self.config, save_path)
        self.worker.progress.connect(self.status_lbl.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", f"Generation failed:\n{e}"))
        self.worker.start()
        self.btn_gen.setEnabled(False); self.btn_batch.setEnabled(False); self.btn_save.setEnabled(False)

    def on_finished(self, result):
        self.btn_gen.setEnabled(True); self.btn_batch.setEnabled(True)
        if isinstance(result, QImage):
            self.last_image = result
            self.img_label.setPixmap(QPixmap.fromImage(result).scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_save.setEnabled(True)
            self.status_lbl.setText("Render Complete")
        elif isinstance(result, str):
            self.status_lbl.setText(f"GIF Saved: {result}")

    def start_batch_generation(self):
        if self.worker and self.worker.isRunning(): return
        self.batch_total = self.spin_batch_count.value()
        self.batch_current = 0
        self.batch_running = True
        self.batch_is_anim = self.spin_frames.value() > 0
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, self.batch_total); self.progress_bar.setValue(0)
        self.btn_gen.setEnabled(False); self.btn_batch.setEnabled(False); self.btn_save.setEnabled(False)
        self.run_next_batch_step()

    def run_next_batch_step(self):
        if not self.batch_running: return
        if self.batch_current < self.batch_total:
            self.batch_current += 1
            self.progress_bar.setValue(self.batch_current)
            self.status_lbl.setText(f"Batch Progress: {self.batch_current}/{self.batch_total}")
            seed = random.randint(0, 2**32 - 1)
            self.config = self.get_current_config(override_seed=seed)
            output_dir = Path("output"); output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "gif" if self.batch_is_anim else "png"
            fname = f"batch_{timestamp}_{self.batch_current:03d}.{ext}"
            save_path = str(output_dir / fname)
            self.worker = GeneratorThread(self.config, save_path)
            self.worker.progress.connect(self.status_lbl.setText)
            self.worker.finished.connect(self.on_batch_item_finished)
            self.worker.error.connect(self.on_batch_error)
            self.worker.start()
        else:
            self.finish_batch()

    def on_batch_item_finished(self, result):
        if isinstance(result, QImage):
            self.last_image = result
            self.img_label.setPixmap(QPixmap.fromImage(result).scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.run_next_batch_step()

    def on_batch_error(self, error_msg):
        logger.error(f"Batch item failed: {error_msg}")
        self.run_next_batch_step()

    def finish_batch(self):
        self.batch_running = False
        self.progress_bar.setVisible(False)
        self.btn_gen.setEnabled(True); self.btn_batch.setEnabled(True)
        self.status_lbl.setText("Batch Complete!")
        QMessageBox.information(self, "Batch Complete", f"Finished generating {self.batch_total} images/animations.")

    def save_image(self):
        if not self.last_image: return
        output_dir = Path("output"); output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = output_dir / f"flame_{timestamp}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", str(default_path), "PNG (*.png);;JPEG (*.jpg)")
        if path:
            if self.last_image.save(path): logger.info(f"Image saved to {path}")
            else: logger.error(f"Failed to save image to {path}")