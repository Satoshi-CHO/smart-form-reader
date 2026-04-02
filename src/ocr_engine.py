import logging
import os
import shlex
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EASYOCR_CACHE_DIR = PROJECT_ROOT / ".cache" / "easyocr"


def configure_easyocr_cache() -> str:
    EASYOCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(EASYOCR_CACHE_DIR))
    return os.environ["EASYOCR_MODULE_PATH"]


def get_easyocr_directories() -> tuple[str, str]:
    base_dir = configure_easyocr_cache()
    model_dir = str(Path(base_dir) / "model")
    user_network_dir = str(Path(base_dir) / "user_network")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(user_network_dir).mkdir(parents=True, exist_ok=True)
    return model_dir, user_network_dir


@dataclass
class OcrResult:
    raw_text: str
    error_flags: List[str] = field(default_factory=list)


@dataclass
class RoiConfig:
    roi_id: str
    crop_path: str
    roi_type: str = "text"
    expected_length: int = -1


class IOcrStrategy(ABC):
    @abstractmethod
    def __init__(self, use_gpu: bool = True):
        pass

    @abstractmethod
    def warmup(self):
        pass

    @abstractmethod
    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        pass


class EasyOcrDigitStrategy(IOcrStrategy):
    def __init__(self, use_gpu: bool = True):
        import easyocr

        logging.info("Initializing EasyOCR for digits...")
        model_dir, user_network_dir = get_easyocr_directories()
        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory=model_dir,
            user_network_directory=user_network_dir,
        )

    def warmup(self):
        logging.info("Warming up DigitStrategy...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.reader.readtext(dummy_img, allowlist="0123456789")
        except Exception:
            pass

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        error_flags: List[str] = []
        result_text = ""

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, processed = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
        res = self.reader.readtext(processed, allowlist="0123456789")
        result_text = "".join([r[1] for r in res])

        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("DIGIT_MISMATCH")

        return OcrResult(raw_text=result_text, error_flags=error_flags)


class EasyOcrTextStrategy(IOcrStrategy):
    def __init__(self, use_gpu: bool = True):
        import easyocr

        logging.info("Initializing EasyOCR for Japanese Text...")
        model_dir, user_network_dir = get_easyocr_directories()
        self.reader = easyocr.Reader(
            ["ja", "en"],
            gpu=use_gpu,
            model_storage_directory=model_dir,
            user_network_directory=user_network_dir,
        )

    def warmup(self):
        logging.info("Warming up EasyTextStrategy...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.reader.readtext(dummy_img)
        except Exception as exc:
            logging.error("TextStrategy warmup error: %s", exc)

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        error_flags: List[str] = []
        res = self.reader.readtext(image, detail=0)
        result_text = " ".join(res)

        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("LENGTH_MISMATCH")

        return OcrResult(raw_text=result_text, error_flags=error_flags)


class NdlOcrLiteTextStrategy(IOcrStrategy):
    DEFAULT_COMMAND_TEMPLATE = 'ndlocr-lite --input "{input}" --output "{output}"'

    def __init__(self, use_gpu: bool = True, command_template: str = ""):
        self.use_gpu = use_gpu
        self.command_template = (
            command_template
            or os.environ.get("NDLOCR_LITE_COMMAND")
            or self.DEFAULT_COMMAND_TEMPLATE
        )

    def warmup(self):
        logging.info("NDLOCR-Lite is invoked via subprocess; skipping warmup.")

    def _build_command(self, input_path: str, output_dir: str) -> List[str]:
        if "{input}" not in self.command_template or "{output}" not in self.command_template:
            raise ValueError("NDLOCR-Lite command template must contain both {input} and {output}.")
        command = self.command_template.format(input=input_path, output=output_dir)
        return shlex.split(command, posix=True)

    def _read_output_text(self, output_dir: Path, stdout: str) -> str:
        txt_files = sorted(output_dir.rglob("*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
        for txt_file in txt_files:
            try:
                return txt_file.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                return txt_file.read_text(encoding="cp932", errors="ignore").strip()
        return stdout.strip()

    def _infer_workdir(self, command: List[str]) -> str | None:
        for part in command:
            if part.lower().endswith("ocr.py") and os.path.isfile(part):
                return str(Path(part).resolve().parent)
        return None

    def _run_command(self, input_path: str, expected_length: int = -1) -> OcrResult:
        error_flags: List[str] = []
        result_text = ""

        try:
            with tempfile.TemporaryDirectory(prefix="ndlocr_lite_") as temp_dir:
                output_dir = Path(temp_dir) / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                command = self._build_command(str(input_path), str(output_dir))
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=self._infer_workdir(command),
                    timeout=120,
                )
                result_text = self._read_output_text(output_dir, completed.stdout)
                if not result_text:
                    error_flags.append("NDLOCR_OUTPUT_NOT_FOUND")
        except FileNotFoundError:
            error_flags.append("NDLOCR_EXECUTABLE_NOT_FOUND")
        except subprocess.TimeoutExpired:
            error_flags.append("NDLOCR_TIMEOUT")
        except subprocess.CalledProcessError as exc:
            logging.error("NDLOCR-Lite failed: %s", exc.stderr or exc.stdout or exc)
            error_flags.append("NDLOCR_COMMAND_ERROR")
        except Exception as exc:
            logging.error("NDLOCR-Lite image read error: %s", exc)
            error_flags.append("IMAGE_READ_ERROR")

        if expected_length > 0 and len(result_text) != expected_length:
            error_flags.append("LENGTH_MISMATCH")

        return OcrResult(raw_text=result_text, error_flags=error_flags)

    def read_from_file(self, input_path: str, expected_length: int = -1) -> OcrResult:
        return self._run_command(input_path, expected_length)

    def read(self, image: np.ndarray, expected_length: int = -1) -> OcrResult:
        try:
            with tempfile.TemporaryDirectory(prefix="ndlocr_lite_input_") as temp_dir:
                input_path = Path(temp_dir) / "roi.png"
                if not cv2.imwrite(str(input_path), image):
                    raise RuntimeError("Failed to write temporary ROI image.")
                return self._run_command(str(input_path), expected_length)
        except Exception as exc:
            logging.error("NDLOCR-Lite image read error: %s", exc)
            return OcrResult(raw_text="", error_flags=["IMAGE_READ_ERROR"])


class OcrEngineFactory:
    @staticmethod
    def create_text_engine(engine_name: str, use_gpu: bool = True, ndlocr_command: str = "") -> IOcrStrategy:
        if engine_name == "EasyOCR":
            return EasyOcrTextStrategy(use_gpu)
        if engine_name == "NDLOCR-Lite":
            return NdlOcrLiteTextStrategy(use_gpu, command_template=ndlocr_command)

        logging.warning("Unknown engine '%s', falling back to EasyOCR.", engine_name)
        return EasyOcrTextStrategy(use_gpu)

    @staticmethod
    def create_digit_engine(use_gpu: bool = True) -> IOcrStrategy:
        return EasyOcrDigitStrategy(use_gpu)


class OCREngine:
    def __init__(self, use_gpu: bool = True, text_engine: str = "EasyOCR", ndlocr_command: str = ""):
        self.digit_reader: IOcrStrategy = OcrEngineFactory.create_digit_engine(use_gpu)
        self.text_reader: IOcrStrategy = OcrEngineFactory.create_text_engine(text_engine, use_gpu, ndlocr_command)

    def warmup(self):
        self.digit_reader.warmup()
        self.text_reader.warmup()

    def process_roi(self, roi_info: Dict[str, Any]) -> Dict[str, Any]:
        config = RoiConfig(
            roi_id=roi_info.get("roi_id", "unknown"),
            crop_path=roi_info.get("crop_path", ""),
            roi_type=roi_info.get("type", "text"),
            expected_length=roi_info.get("expected_length", -1),
        )

        if not config.crop_path:
            return {"roi_id": config.roi_id, "raw_text": "", "error_flags": ["INVALID_CROP_PATH"]}

        img = cv2.imread(config.crop_path)
        if img is None:
            return {"roi_id": config.roi_id, "raw_text": "", "error_flags": ["IMAGE_LOAD_ERROR"]}

        if config.roi_type == "digits":
            res_obj = self.digit_reader.read(img, config.expected_length)
        elif isinstance(self.text_reader, NdlOcrLiteTextStrategy):
            res_obj = self.text_reader.read_from_file(config.crop_path, config.expected_length)
        else:
            res_obj = self.text_reader.read(img, config.expected_length)

        return {
            "roi_id": config.roi_id,
            "raw_text": res_obj.raw_text,
            "error_flags": res_obj.error_flags,
        }
