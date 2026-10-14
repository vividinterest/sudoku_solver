import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from example import clean_black_white, save_capture


class PaperCleanupTests(unittest.TestCase):
    def test_shadows_become_white_and_strokes_remain_black(self):
        # Darker paper on the left, with camera noise and a thin ink stroke.
        rng = np.random.default_rng(7)
        gray = np.tile(np.linspace(100, 235, 600), (420, 1))
        gray = np.clip(gray + rng.normal(0, 1, gray.shape), 0, 255).astype(np.uint8)
        gray[160:164, 60:540] //= 3
        result = clean_black_white(gray, dpi=100)
        self.assertEqual(set(np.unique(result)), {0, 255})
        self.assertTrue(np.all(result[161:163, 70:530] == 0))
        self.assertGreater(np.mean(result[:120] == 255), 0.999)

    def test_faint_pencil_sensitivity(self):
        gray = np.full((150, 200), 200, np.uint8)
        gray[70:73, 30:170] = 190
        self.assertTrue(np.all(clean_black_white(gray, ink_contrast=18)[71, 40:160] == 255))
        self.assertTrue(np.all(clean_black_white(gray, ink_contrast=8)[71, 40:160] == 0))

    def test_optional_speck_and_border_removal(self):
        gray = np.full((200, 200), 230, np.uint8)
        gray[100, 100] = 0
        cv2.line(gray, (40, 60), (160, 60), 0, 2)
        gray[:2] = 0
        self.assertEqual(clean_black_white(gray)[100, 100], 0)
        result = clean_black_white(gray, dpi=100, min_ink_area=3, border_mm=1)
        self.assertEqual(result[100, 100], 255)
        self.assertEqual(result[60, 100], 0)
        self.assertTrue(np.all(result[:4] == 255))

    def test_saved_binary_and_color_share_a4_geometry(self):
        frame = np.full((300, 220, 3), 230, np.uint8)
        cv2.line(frame, (50, 60), (140, 190), (10, 10, 10), 3)
        corners = np.float32([[10, 10], [209, 10], [209, 289], [10, 289]])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), dpi=100, orientation='portrait',
                                   ink_contrast=18, background_mm=12, min_ink_area=0, border_mm=0)
            save_capture(frame, corners, args)
            with Image.open(next(Path(directory).glob('*_bw.png'))) as binary, \
                    Image.open(next(Path(directory).glob('*_color.png'))) as color:
                self.assertEqual(binary.mode, '1')
                self.assertEqual(binary.size, (827, 1169))
                self.assertEqual(binary.size, color.size)
                self.assertLess(abs(binary.info['dpi'][0] - 100), 0.1)
                self.assertEqual(set(np.unique(np.array(binary.convert('L')))), {0, 255})


if __name__ == '__main__':
    unittest.main()
