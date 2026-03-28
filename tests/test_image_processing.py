import unittest
import cv2
import numpy as np
import os
import shutil
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_processing import ImageProcessor

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_workspace"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a mock template with distinct features
        self.template_path = os.path.join(self.test_dir, "template.png")
        img = np.ones((500, 500), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (150, 150), 0, -1)
        cv2.circle(img, (400, 100), 40, 0, -1)
        cv2.putText(img, "MOCK TEXT", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        cv2.imwrite(self.template_path, img)

        # Create config
        self.config_path = os.path.join(self.test_dir, "config.json")
        with open(self.config_path, "w") as f:
            json.dump([{"id": "roi1", "x": 100, "y": 250, "w": 200, "h": 50}], f)
            
        self.processor = ImageProcessor(self.template_path, self.config_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_alignment_and_rotation(self):
        # Create a test image, rotated 180
        test_img_path = os.path.join(self.test_dir, "test_input.png")
        template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        
        # Rotate 180 degrees
        rotated = cv2.rotate(template, cv2.ROTATE_180)
        # Shift slightly
        shifted = np.pad(rotated, ((10, 0), (20, 0)), mode='constant', constant_values=255)[:500, :500]
        cv2.imwrite(test_img_path, shifted)

        aligned, status = self.processor.align_and_correct_rotation(test_img_path)
        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(aligned)
        
        # Test ROI extraction
        out_dir = os.path.join(self.test_dir, "out")
        results = self.processor.extract_rois(aligned, out_dir, "test1")
        self.assertEqual(len(results), 1)
        self.assertTrue(os.path.exists(results[0]["crop_path"]))

if __name__ == '__main__':
    unittest.main()
