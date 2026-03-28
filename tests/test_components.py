import unittest
import cv2
import numpy as np
import os
import shutil
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_processing import ImageProcessor
from src.ocr_engine import OCREngine

class TestComponents(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_workspace_components"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a mock template
        self.template_path = os.path.join(self.test_dir, "template.png")
        img = np.ones((500, 500), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 100), (300, 150), 0, 2)
        cv2.putText(img, "1234567", (110, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        
        cv2.rectangle(img, (100, 200), (400, 250), 0, 2)
        cv2.putText(img, "HELLO", (110, 235), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        cv2.imwrite(self.template_path, img)

        # Create Config
        self.config_path = os.path.join(self.test_dir, "config.json")
        with open(self.config_path, "w") as f:
            json.dump([
                {"id": "test_digits", "type": "digits", "x": 100, "y": 100, "w": 200, "h": 50, "expected_length": 7},
                {"id": "test_text", "type": "text", "x": 100, "y": 200, "w": 300, "h": 50, "expected_length": 0}
            ], f)
            
        self.processor = ImageProcessor(self.template_path, self.config_path)
        self.engine = OCREngine(use_gpu=False)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_roi_extraction(self):
        test_img = cv2.imread(self.template_path)
        out_dir = os.path.join(self.test_dir, "out")
        
        results = self.processor.extract_rois(test_img, out_dir, "testfile")
        self.assertEqual(len(results), 2)
        
        digits_path = next(r["crop_path"] for r in results if r["roi_id"] == "test_digits")
        self.assertTrue(os.path.exists(digits_path))

    def test_ocr_digits_and_text(self):
        test_img = cv2.imread(self.template_path)
        out_dir = os.path.join(self.test_dir, "out")
        roi_infos = self.processor.extract_rois(test_img, out_dir, "testfile")
        
        digits_roi = next(r for r in roi_infos if r["roi_id"] == "test_digits")
        text_roi = next(r for r in roi_infos if r["roi_id"] == "test_text")
        
        # Test Digits
        res_digits = self.engine.process_roi(digits_roi)
        self.assertIn("1234567", res_digits["raw_text"])
        self.assertEqual(len(res_digits["error_flags"]), 0)
        
        # Test error flag logic
        digits_roi["expected_length"] = 8
        res_digits_error = self.engine.process_roi(digits_roi)
        self.assertIn("DIGIT_MISMATCH", res_digits_error["error_flags"])
            
if __name__ == '__main__':
    unittest.main()
