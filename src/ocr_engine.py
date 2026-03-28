import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import cv2
import numpy as np
from dataclasses import dataclass, field

# =====================================================================
# 1. DTO (Data Transfer Object)層
# メソッド間で受け渡す値を明確な「型」として定義します。
# 辞書型（Dict）の代わりにdataclassを使うことで、可読性と保守性が向上します。
# =====================================================================
@dataclass
class OcrResult:
    """OCRエンジンの認識結果を保持するデータクラス"""
    raw_text: str
    error_flags: List[str] = field(default_factory=list)

@dataclass
class RoiConfig:
    """ROI（関心領域）の設定を保持するデータクラス"""
    roi_id: str
    crop_path: str
    roi_type: str = "text"
    expected_length: int = -1

# =====================================================================
# 2. OCRエンジン層 (Strategy パターン)
# エンジンごとの実装を一つのインターフェース(IOcrStrategy)に統一します。
# =====================================================================
class IOcrStrategy(ABC):
    """
    OCRエンジンの共通インターフェースです。
    将来新しい高精度エンジン（例：PaddleOCR, OpenAI API等）が出た際は、
    このクラスを継承して新しいStrategyクラスを作るだけで済みます。
    （Open/Closedの原則：拡張に対して開かれ、修正に対して閉じているべき）
    """
    
    @abstractmethod
    def __init__(self, use_gpu: bool = True):
        pass

    @abstractmethod
    def warmup(self):
        """モデルの事前読み込みや初期化処理（ウォームアップ）を行います"""
        pass

    @abstractmethod
    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        """
        画像配列を受け取り、OCR結果を返します。
        以前のようにファイルパスではなく画像データそのものを受け取ることで
        I/Oと認識処理を分離しています（単一責任の原則）。
        """
        pass

class EasyOcrDigitStrategy(IOcrStrategy):
    """EasyOCRを用いた数字認識ロジック（Strategyパターン）"""
    def __init__(self, use_gpu: bool = True):
        import easyocr
        logging.info("Initializing EasyOCR for digits...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def warmup(self):
        logging.info("Warming up DigitStrategy...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.reader.readtext(dummy_img, allowlist="0123456789")
        except Exception:
            pass

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        error_flags = []
        result_text = ""
        
        # NOTE: より厳密に責務を分ける場合、この前処理(cv2.threshold)は
        # 別の「ImagePreprocessor」クラスなどに切り出すのが理想的です。
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        _, processed = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
        res = self.reader.readtext(processed, allowlist="0123456789")
        result_text = "".join([r[1] for r in res])
            
        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("DIGIT_MISMATCH")
            
        return OcrResult(raw_text=result_text, error_flags=error_flags)

class EasyOcrTextStrategy(IOcrStrategy):
    """EasyOCRを用いた文字認識ロジック（Strategyパターン）"""
    def __init__(self, use_gpu: bool = True):
        import easyocr
        logging.info("Initializing EasyOCR for Japanese Text...")
        self.reader = easyocr.Reader(['ja', 'en'], gpu=use_gpu)

    def warmup(self):
        logging.info("Warming up EasyTextStrategy...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.reader.readtext(dummy_img)
        except Exception as e:
            logging.error(f"TextStrategy warmup error: {e}")

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        error_flags = []
        res = self.reader.readtext(image, detail=0)
        result_text = " ".join(res)
            
        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("LENGTH_MISMATCH")
        
        return OcrResult(raw_text=result_text, error_flags=error_flags)

class MangaOcrStrategy(IOcrStrategy):
    """MangaOCRを用いた文字認識ロジック（Strategyパターン）"""
    def __init__(self, use_gpu: bool = True):
        from manga_ocr import MangaOcr
        logging.info("Initializing MangaOCR for Japanese Text...")
        self.reader = MangaOcr()

    def warmup(self):
        logging.info("Warming up MangaOCR...")
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        try:
            self.reader(dummy_img)
        except Exception as e:
            logging.error(f"MangaOCR warmup error: {e}")

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        from PIL import Image
        error_flags = []
        result_text = ""
        
        try:
            # np.ndarray -> PIL Image への変換（MangaOCR用）
            # データ形式の変換もこのクラスの中で完結させることで他クラスに影響を与えません
            if len(image.shape) == 2:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            else:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result_text = self.reader(img_pil)
        except Exception:
            error_flags.append("IMAGE_READ_ERROR")
            
        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("LENGTH_MISMATCH")
        
        return OcrResult(raw_text=result_text, error_flags=error_flags)


# =====================================================================
# 3. Factory パターン
# OCRインスタンスの生成を専門に行うクラスです。
# =====================================================================
class OcrEngineFactory:
    """
    指定された名前から対応するOCR Strategyオブジェクトを生成するFactoryです。
    将来エンジンが増えた際は、このクラスの if-elif を追加するだけで済みます。
    """
    @staticmethod
    def create_text_engine(engine_name: str, use_gpu: bool = True) -> IOcrStrategy:
        if engine_name == "EasyOCR":
            return EasyOcrTextStrategy(use_gpu)
        elif engine_name == "MangaOCR":
            return MangaOcrStrategy(use_gpu)
        else:
            # 未知のエンジンが指定された場合のデフォルト(または例外発生)
            logging.warning(f"Unknown engine '{engine_name}', falling back to MangaOCR.")
            return MangaOcrStrategy(use_gpu)
            
    @staticmethod
    def create_digit_engine(use_gpu: bool = True) -> IOcrStrategy:
        # 数字専用エンジン。将来新しい数字エンジンが登場した場合も差し替え容易です。
        return EasyOcrDigitStrategy(use_gpu)


# =====================================================================
# 4. アプリケーション層 (Facade パターン)
# =====================================================================
class OCREngine:
    """
    Facadeクラスです。GUI等の外部コードからはこのクラスのみを操作します。
    （以前のインターフェースを維持し、既存コードを壊さずに中身だけをリファクタリングしています）
    """
    def __init__(self, use_gpu: bool = True, text_engine: str = "MangaOCR"):
        # 以前はここにif-elseがありましたが、Factoryに処理を委譲することでスッキリしました
        self.digit_reader: IOcrStrategy = OcrEngineFactory.create_digit_engine(use_gpu)
        self.text_reader: IOcrStrategy = OcrEngineFactory.create_text_engine(text_engine, use_gpu)

    def warmup(self):
        self.digit_reader.warmup()
        self.text_reader.warmup()

    def process_roi(self, roi_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        GUIからの辞書データを受け取り、内部のデータクラスに変換して処理を行います。
        外部の仕様変更(Dict -> OcrResult変換)の防波堤（アダプター）としての役割も持ちます。
        """
        config = RoiConfig(
            roi_id=roi_info.get("roi_id", "unknown"),
            crop_path=roi_info.get("crop_path", ""),
            roi_type=roi_info.get("type", "text"),
            expected_length=roi_info.get("expected_length", -1)
        )
        
        if not config.crop_path:
            return {
                "roi_id": config.roi_id,
                "raw_text": "",
                "error_flags": ["INVALID_CROP_PATH"]
            }
            
        img = cv2.imread(config.crop_path)
        if img is None:
            return {
                "roi_id": config.roi_id,
                "raw_text": "",
                "error_flags": ["IMAGE_LOAD_ERROR"]
            }
        
        # Strategyに処理を委譲
        if config.roi_type == "digits":
            res_obj = self.digit_reader.read(img, config.expected_length)
        else:
            res_obj = self.text_reader.read(img, config.expected_length)
            
        # 外部プログラムが期待する辞書型に戻して返す
        return {
            "roi_id": config.roi_id,
            "raw_text": res_obj.raw_text,
            "error_flags": res_obj.error_flags
        }
