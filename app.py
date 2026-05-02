import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Import GUI and Utilities
import gui_main
import utils_palettes

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application starting...")

    palette_dir = Path("palettes")
    if not palette_dir.exists():
        palette_dir.mkdir()
        logger.info("Created 'palettes' directory.")
        for _ in range(5):
            utils_palettes.generate_palette_file("palettes")

    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = gui_main.FlameStudio()
    window.show()
    
    ret = app.exec()
    
    logger.info("Application closing.")
    sys.exit(ret)

if __name__ == "__main__":
    main()