import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.image_processing import ImageMatcher, ImageAligner, RoiExtractor
from src.ocr_engine import IOcrStrategy, NdlOcrLiteTextStrategy, OcrEngineFactory, OcrResult, RoiConfig


class DummyOcrStrategy(IOcrStrategy):
    def __init__(self, use_gpu: bool = False):
        self.warmup_called = False

    def warmup(self):
        self.warmup_called = True

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        error_flags = []
        result_text = "DUMMY_TEXT"
        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("LENGTH_MISMATCH")
        return OcrResult(raw_text=result_text, error_flags=error_flags)


class TestRefactoredArchitecture(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_workspace_refactored"
        os.makedirs(self.test_dir, exist_ok=True)

        self.img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.img, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.putText(self.img, "MOCK", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_roi_config_dataclass(self):
        config = RoiConfig(roi_id="test_id", crop_path="dummy.png", roi_type="digits", expected_length=5)
        self.assertEqual(config.roi_id, "test_id")
        self.assertEqual(config.roi_type, "digits")
        self.assertEqual(config.expected_length, 5)

    def test_dummy_ocr_strategy(self):
        strategy = DummyOcrStrategy()
        strategy.warmup()
        self.assertTrue(strategy.warmup_called)

        res_error = strategy.read(self.img, expected_length=5)
        self.assertIn("LENGTH_MISMATCH", res_error.error_flags)

        res_ok = strategy.read(self.img, expected_length=-1)
        self.assertEqual(res_ok.raw_text, "DUMMY_TEXT")
        self.assertEqual(len(res_ok.error_flags), 0)

    def test_ocr_factory(self):
        text_engine = OcrEngineFactory.create_text_engine("EasyOCR", use_gpu=False)
        self.assertEqual(text_engine.__class__.__name__, "EasyOcrTextStrategy")

        ndlocr_engine = OcrEngineFactory.create_text_engine("NDLOCR-Lite", use_gpu=False)
        self.assertEqual(ndlocr_engine.__class__.__name__, "NdlOcrLiteTextStrategy")

        digit_engine = OcrEngineFactory.create_digit_engine(use_gpu=False)
        self.assertEqual(digit_engine.__class__.__name__, "EasyOcrDigitStrategy")

        fallback_engine = OcrEngineFactory.create_text_engine("UNKNOWN_ENGINE", use_gpu=False)
        self.assertEqual(fallback_engine.__class__.__name__, "EasyOcrTextStrategy")

    @patch("src.ocr_engine.subprocess.run")
    def test_ndlocr_lite_strategy_reads_txt_output(self, mock_run):
        def fake_run(command, capture_output, text, check, cwd, timeout):
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "result.txt").write_text("テスト結果", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        strategy = NdlOcrLiteTextStrategy(
            use_gpu=False,
            command_template='mock-ndl --input "{input}" --output "{output}"',
        )
        result = strategy.read(self.img)

        self.assertEqual(result.raw_text, "テスト結果")
        self.assertEqual(result.error_flags, [])

    def test_ndlocr_lite_strategy_requires_placeholders(self):
        strategy = NdlOcrLiteTextStrategy(use_gpu=False, command_template="ndlocr-lite")
        result = strategy.read(self.img)

        self.assertEqual(result.raw_text, "")
        self.assertIn("IMAGE_READ_ERROR", result.error_flags)

    @patch("src.ocr_engine.subprocess.run")
    def test_ndlocr_lite_strategy_can_use_existing_crop_file(self, mock_run):
        called = {}

        def fake_run(command, capture_output, text, check, cwd, timeout):
            called["input_path"] = command[command.index("--input") + 1]
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "result.txt").write_text("既存切り抜き", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        with tempfile.TemporaryDirectory() as temp_dir:
            crop_path = os.path.join(temp_dir, "crop.png")
            cv2.imwrite(crop_path, self.img)

            strategy = NdlOcrLiteTextStrategy(
                use_gpu=False,
                command_template='mock-ndl --input "{input}" --output "{output}"',
            )
            result = strategy.read_from_file(crop_path)

        self.assertEqual(called["input_path"], crop_path)
        self.assertEqual(result.raw_text, "既存切り抜き")
        self.assertEqual(result.error_flags, [])

    def test_roi_extractor(self):
        config = [
            {"id": "roi_1", "type": "text", "x": 100, "y": 100, "w": 50, "h": 50},
            {"id": "roi_2", "type": "digits", "x": -10, "y": -10, "w": 20, "h": 20},
        ]

        out_dir = os.path.join(self.test_dir, "extracted")
        results = RoiExtractor.extract_and_save(self.img, config, out_dir, "testfile")

        self.assertEqual(len(results), 2)
        roi1 = next(r for r in results if r["roi_id"] == "roi_1")
        self.assertEqual(roi1["type"], "text")
        self.assertTrue(os.path.exists(roi1["crop_path"]))


if __name__ == "__main__":
    unittest.main()
