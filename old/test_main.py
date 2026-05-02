#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

# Add parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))
import main

class TestFlameMaster(unittest.TestCase):
    def setUp(self):
        """Create temporary directories and test resources"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "outputs"
        self.palette_dir = Path(self.temp_dir.name) / "palettes"
        self.output_dir.mkdir()
        self.palette_dir.mkdir()

        # Create ONLY text palette file to avoid ambiguity in loading tests
        with open(self.palette_dir / "test_palette.txt", 'w') as f:
            f.write("255,0,0\n0,255,0\n0,0,255")

    def tearDown(self):
        """Clean up temporary resources"""
        self.temp_dir.cleanup()

    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Valid config
        main.Config(
            width=100, height=100, iterations=1000, transforms=5,
            zoom=1.0, gamma=2.2, layers=1, frames=0
        )
        
        # Invalid parameters
        invalid_cases = [
            ("width", 0),
            ("height", -10),
            ("iterations", 0),
            ("zoom", -0.5),
            ("gamma", 0),
            ("symmetry_segments", 0, {"symmetry": "kaleidoscope"}),
            ("vibrancy", 1.5),
            ("vibrancy", -0.1)
        ]
        
        for param, value, *extras in invalid_cases:
            kwargs = extras[0] if extras else {}
            with self.subTest(param=param, value=value):
                with self.assertRaises(ValueError):
                    main.Config(**{param: value, **kwargs})

    def test_palette_loading(self):
        """Test palette loading functionality"""
        # Test directory loading
        palettes = main.load_palettes(self.palette_dir)
        self.assertEqual(len(palettes), 1, "Should load exactly one palette")
        
        # Verify palette content
        palette = palettes[0]
        self.assertEqual(palette.shape, (3, 3), "Palette should have 3 colors with RGB values")
        np.testing.assert_array_almost_equal(
            palette[0], [1.0, 0.0, 0.0], decimal=5,
            err_msg="First color should be red (normalized)"
        )
        
        # Test missing directory handling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            empty_pals = main.load_palettes(Path("nonexistent_dir"))
            self.assertEqual(len(empty_pals), 0, "Should return empty list for missing directory")
            self.assertGreater(len(w), 0, "Should issue warning for missing directory")
            self.assertIn("not found", str(w[0].message).lower())

    def test_preset_operations(self):
        """Test preset saving/loading functionality"""
        preset_path = self.output_dir / "test_preset.json"
        
        # Create config with complex parameters
        original_palette = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        cfg = main.Config(
            palette=original_palette,
            final_transform=(1.0, 0.1, 0.2, 0.3, 1.0, 0.4),
            background_color=(10, 20, 30),
            vibrancy=0.75
        )
        
        # Save and reload preset
        cfg.save_to_file(preset_path)
        
        with open(preset_path, 'r') as f:
            preset_dict = json.load(f)
        loaded_cfg = main.Config.from_dict(preset_dict)
        
        # Verify parameters
        np.testing.assert_array_almost_equal(
            loaded_cfg.palette, original_palette, decimal=5,
            err_msg="Palette should match after save/load"
        )
        
        # Compare final_transform (JSON converts tuples to lists)
        self.assertEqual(
            tuple(loaded_cfg.final_transform), cfg.final_transform,
            "Final transform should match after save/load"
        )
        
        self.assertEqual(
            loaded_cfg.background_color, cfg.background_color,
            "Background color should match after save/load"
        )
        self.assertAlmostEqual(
            loaded_cfg.vibrancy, cfg.vibrancy, places=5,
            msg="Vibrancy should match after save/load"
        )

    def test_single_image_rendering(self):
        """Test basic image rendering with minimal parameters"""
        output_path = self.output_dir / "test_image.png"
        
        cfg = main.Config(
            width=100, height=100, iterations=5000, transforms=5,
            layers=1, skip=100, seed=42, palette=None,
            zoom=0.8  # Reduced zoom to ensure background visibility
        )
        
        fractal = main.FlameFractal(cfg)
        img = fractal.render_single()
        
        # Verify image properties
        self.assertEqual(img.shape, (100, 100, 3), "Image dimensions should match config")
        self.assertGreater(np.max(img), 10, "Image should contain visible pixels")
        
        # Save and verify file
        Image.fromarray(img).save(output_path)
        self.assertTrue(output_path.exists(), "Output file should be created")
        self.assertGreater(output_path.stat().st_size, 1000, "Output file should have reasonable size")

    def test_symmetry_features(self):
        """Test symmetry transformations"""
        base_cfg = main.Config(
            width=50, height=50, iterations=3000, transforms=3,
            layers=1, skip=100, seed=42, zoom=1.0
        )
        
        # Test X symmetry (doubles width)
        cfg = base_cfg.model_copy(update={"symmetry": "x"})
        img = main.FlameFractal(cfg).render_single()
        self.assertEqual(img.shape, (50, 100, 3), "X symmetry should double width")
        self.assertGreater(np.max(img), 10, "X symmetry image should have visible content")
        
        # Test Y symmetry (doubles height)
        cfg = base_cfg.model_copy(update={"symmetry": "y"})
        img = main.FlameFractal(cfg).render_single()
        self.assertEqual(img.shape, (100, 50, 3), "Y symmetry should double height")
        self.assertGreater(np.max(img), 10, "Y symmetry image should have visible content")
        
        # Test kaleidoscope
        cfg = base_cfg.model_copy(update={
            "symmetry": "kaleidoscope",
            "symmetry_segments": 4
        })
        img = main.FlameFractal(cfg).render_single()
        self.assertEqual(img.shape, (50, 50, 3), "Kaleidoscope should maintain dimensions")
        self.assertGreater(np.max(img), 10, "Kaleidoscope image should have visible content")

    def test_artistic_styles(self):
        """Test artistic filter rendering"""
        styles = ["expressionist", "oil", "watercolor"]
        
        for style in styles:
            with self.subTest(style=style):
                cfg = main.Config(
                    width=60, height=60, iterations=2500, transforms=4,
                    layers=1, art_style=style, seed=42, zoom=1.0
                )
                img = main.FlameFractal(cfg).render_single()
                self.assertEqual(img.shape, (60, 60, 3), f"{style} style should maintain dimensions")
                self.assertGreater(np.max(img), 10, f"{style} style should have visible content")

    def test_background_color(self):
        """Test custom background color"""
        cfg = main.Config(
            width=60, height=60, iterations=2000, transforms=3,
            layers=1, background_color=(255, 0, 0), seed=42,
            zoom=0.3  # Low zoom to ensure background visibility
        )
        img = main.FlameFractal(cfg).render_single()
        
        # Check corners are red (background)
        for corner in [(0,0), (0,-1), (-1,0), (-1,-1)]:
            np.testing.assert_array_equal(
                img[corner], [255, 0, 0],
                err_msg=f"Corner {corner} should be background color"
            )
        
        # Verify some foreground pixels exist
        foreground_mask = np.all(img != [255, 0, 0], axis=-1)
        self.assertTrue(np.any(foreground_mask), "Should contain non-background pixels")

    def test_vibrancy_control(self):
        """Test vibrancy parameter effects"""
        base_cfg = main.Config(
            width=50, height=50, iterations=3000, transforms=3,
            layers=1, seed=42, zoom=1.0
        )
        
        # Full vibrancy (color)
        color_cfg = base_cfg.model_copy(update={"vibrancy": 1.0})
        color_img = main.FlameFractal(color_cfg).render_single()
        
        # Zero vibrancy (grayscale)
        gray_cfg = base_cfg.model_copy(update={"vibrancy": 0.0})
        gray_img = main.FlameFractal(gray_cfg).render_single()
        
        # Verify grayscale image has equal RGB channels in non-black areas
        non_black = np.max(gray_img, axis=-1) > 10
        self.assertTrue(np.any(non_black), "Grayscale image should have visible content")
        
        r, g, b = gray_img[non_black, 0], gray_img[non_black, 1], gray_img[non_black, 2]
        np.testing.assert_array_almost_equal(r, g, decimal=0, err_msg="R should equal G in grayscale")
        np.testing.assert_array_almost_equal(g, b, decimal=0, err_msg="G should equal B in grayscale")
        
        # Verify color image has channel variation
        color_non_black = np.max(color_img, axis=-1) > 10
        self.assertTrue(np.any(color_non_black), "Color image should have visible content")
        
        r, g, b = color_img[color_non_black, 0], color_img[color_non_black, 1], color_img[color_non_black, 2]
        self.assertFalse(
            np.allclose(r, g) and np.allclose(g, b),
            "Color channels should vary in full vibrancy mode"
        )

    def test_final_transform(self):
        """Test final transform application"""
        cfg = main.Config(
            width=60, height=60, iterations=2000, transforms=3,
            layers=1, final_transform=(1.0, 0.5, 0.0, 0.0, 1.0, 0.0),
            seed=42, zoom=1.0
        )
        img = main.FlameFractal(cfg).render_single()
        self.assertEqual(img.shape, (60, 60, 3), "Final transform should maintain dimensions")
        self.assertGreater(np.max(img), 10, "Transformed image should have visible content")

    def test_animation_rendering(self):
        """Test GIF animation generation"""
        output_path = self.output_dir / "test_anim.gif"
        
        cfg = main.Config(
            width=40, height=40, iterations=1500, transforms=3,
            frames=3, layers=1, skip=100, seed=42, zoom=1.0
        )
        
        fractal = main.FlameFractal(cfg)
        fractal.render_animation(str(output_path))
        
        # Verify file exists and has content
        self.assertTrue(output_path.exists(), "Animation file should be created")
        self.assertGreater(output_path.stat().st_size, 1000, "Animation file should have reasonable size")
        
        # Verify frame count
        with Image.open(output_path) as gif:
            self.assertEqual(gif.n_frames, 3, "Animation should contain exactly 3 frames")

    @patch('main.load_palettes')
    def test_cli_operations(self, mock_load_palettes):
        """Test command-line interface operations"""
        output_path = self.output_dir / "cli_test.png"
        preset_path = self.output_dir / "cli_preset.json"
        
        # Mock palette to avoid external dependencies
        mock_load_palettes.return_value = [np.array([[0.5, 0.5, 0.5]])]
        
        # Test preset saving
        test_args = [
            'main.py',
            '--width', '50',
            '--height', '50',
            '--iterations', '500',
            '--save-preset', str(preset_path),
            '--palette', 'mock_palette',
            '--seed', '42'
        ]
        with patch.object(sys, 'argv', test_args):
            main.main()
        self.assertTrue(preset_path.exists(), "Preset file should be created")
        
        # Test preset loading and image generation
        test_args = [
            'main.py',
            '--preset', str(preset_path),
            '--output', str(output_path),
            '--bg-color', '100', '100', '100',
            '--vibrancy', '0.8'
        ]
        with patch.object(sys, 'argv', test_args):
            main.main()
        
        # Verify output
        self.assertTrue(output_path.exists(), "CLI output file should be created")
        with Image.open(output_path) as img:
            self.assertEqual(img.size, (50, 50), "CLI output should match configured dimensions")

if __name__ == "__main__":
    unittest.main(verbosity=2)