import unittest
import cv2
import numpy as np
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processing import ImageMatcher, ImageAligner, RoiExtractor
from src.ocr_engine import OcrResult, RoiConfig, IOcrStrategy, OcrEngineFactory

class DummyOcrStrategy(IOcrStrategy):
    """
    テスト用のモックOCRエンジン。
    実際の重いモデルをロードせず、テストを高速かつ安定して行うためのStrategyです。
    このようなMockクラスが簡単に作れるのも、Strategyパターンの強みです。
    """
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
        
        # ダミー画像を作成
        self.img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.img, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.putText(self.img, "MOCK", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------
    # 1. OCRエンジン層のテスト (単体テスト)
    # -------------------------------------------------------------
    def test_roi_config_dataclass(self):
        """DTO(データクラス)が意図通りに機能するか確認"""
        config = RoiConfig(roi_id="test_id", crop_path="dummy.png", roi_type="digits", expected_length=5)
        self.assertEqual(config.roi_id, "test_id")
        self.assertEqual(config.roi_type, "digits")
        self.assertEqual(config.expected_length, 5)

    def test_dummy_ocr_strategy(self):
        """
        モックを用いた独立したOCRのテスト。
        これにより「ファイルI/O」や「画像処理」と切り離して、
        OCRの結果ハンドリングだけをテストできます。
        """
        strategy = DummyOcrStrategy()
        strategy.warmup()
        self.assertTrue(strategy.warmup_called)
        
        # 期待する長さと一致しない場合（エラーフラグのテスト）
        res_error = strategy.read(self.img, expected_length=5)
        self.assertIn("LENGTH_MISMATCH", res_error.error_flags)
        
        # 期待する長さと一致した場合、または指定しない場合
        res_ok = strategy.read(self.img, expected_length=-1)
        self.assertEqual(res_ok.raw_text, "DUMMY_TEXT")
        self.assertEqual(len(res_ok.error_flags), 0)

    def test_ocr_factory(self):
        """Factoryが適切なクラスのインスタンスを生成するか確認"""
        text_engine = OcrEngineFactory.create_text_engine("EasyOCR", use_gpu=False)
        self.assertEqual(text_engine.__class__.__name__, "EasyOcrTextStrategy")
        
        digit_engine = OcrEngineFactory.create_digit_engine(use_gpu=False)
        self.assertEqual(digit_engine.__class__.__name__, "EasyOcrDigitStrategy")

        fallback_engine = OcrEngineFactory.create_text_engine("UNKNOWN_ENGINE", use_gpu=False)
        # 未知エンジンの場合はフォールバックとしてMangaOCRが返る設計
        self.assertEqual(fallback_engine.__class__.__name__, "MangaOcrStrategy")

    # -------------------------------------------------------------
    # 2. 画像処理層のテスト (単体テスト)
    # -------------------------------------------------------------
    def test_roi_extractor(self):
        """
        RoiExtractor クラスの単体テスト
        画像全体からの切り出し（Crop）と保存だけを担うため、
        テンプレートマッチングなどを実行せずに単独でテスト可能です。
        """
        config = [
            {"id": "roi_1", "type": "text", "x": 100, "y": 100, "w": 50, "h": 50},
            {"id": "roi_2", "type": "digits", "x": -10, "y": -10, "w": 20, "h": 20} # はみ出す意地悪い設定
        ]
        
        out_dir = os.path.join(self.test_dir, "extracted")
        results = RoiExtractor.extract_and_save(self.img, config, out_dir, "testfile")
        
        # はみ出す設定のうち、部分的にでも切り出せるものは保存される設計になっていますが、
        # 完全に枠外のものは弾かれます。今回は(0,0)からサイズ(10,10)分だけクロップされるはずです。
        self.assertEqual(len(results), 2)
        
        roi1 = next(r for r in results if r["roi_id"] == "roi_1")
        self.assertEqual(roi1["type"], "text")
        self.assertTrue(os.path.exists(roi1["crop_path"]))

if __name__ == '__main__':
    unittest.main()
