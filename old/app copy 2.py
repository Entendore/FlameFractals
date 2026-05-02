import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Import the GUI window and core utilities
import FlameFractals.old.gui as gui
import FlameFractals.old.fractal_core as core

def setup_logging():
    """Configures logging to output to the terminal with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # 1. Setup Logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application starting...")

    # 2. Ensure necessary directories exist
    palette_dir = Path("palettes")
    if not palette_dir.exists():
        palette_dir.mkdir()
        logger.info("Created 'palettes' directory.")
        
        logger.info("Generating initial sample palettes...")
        for _ in range(5):
            core.generate_palette_file("palettes")

    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()
        logger.info("Created 'output' directory for saved images.")

    # 3. Start Application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = gui.FlameStudio()
    window.show()
    
    ret = app.exec()
    
    logger.info("Application closing.")
    sys.exit(ret)

if __name__ == "__main__":
    main()