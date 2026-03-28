# Smart Form Reader 📄🔍

*Read this in [日本語 (Japanese)](README_ja.md).*

**Smart Form Reader** is a highly efficient OCR desktop application that analyzes scanned forms and documents, extracting text and digits from specific, predefined regions.

While the image processing and cropping functionalities are highly accurate, this project is fundamentally designed with the assumption that **"when a more advanced or highly accurate OCR engine (AI models or APIs) becomes available in the future, it can be seamlessly swapped in."** The architecture is built entirely upon Object-Oriented best practices (such as SOLID principles and Design Patterns).

As such, it serves as a highly extensible template for OCR systems, as well as an excellent **educational resource for learning Python architecture, refactoring, and design patterns.**

---

## 🎯 Key Features and Design Intentions

1. **Extensible Architecture (Strategy & Factory Patterns)**  
   If you want to add a new OCR engine (e.g., OpenAI Vision, PaddleOCR, etc.), you can simply create a new class implementing the `IOcrStrategy` interface and register it with the Factory. There is virtually no need to edit the existing code. (Open-Closed Principle)
2. **Separation of Concerns (Single Responsibility Principle)**  
   The feature point extraction (AKAZE), alignment/rotation correction (Homography), image cropping, and the actual OCR processing are entirely separated into independent classes, making each component independently testable.
3. **Comprehensive GUI Tool**  
   It comes with an intuitive UI built with Tkinter, allowing you to visually auto-scan or manually adjust the reading areas (masks).

---

## 🛠 Getting Started

### Prerequisites
* Python 3.8 or higher.
* (Optional) NVIDIA GPU with CUDA toolkit is recommended for faster inference.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-form-reader.git
cd smart-form-reader
```

### 2. Environment Setup
Running the provided batch or shell scripts will automatically create a virtual environment (`venv`) and install the necessary dependencies (`torch`, `easyocr`, `manga-ocr`, `opencv-python`, etc.).

* **Windows:**
  ```cmd
  run_app.bat
  ```
* **macOS / Linux:**
  ```bash
  chmod +x run_app.sh
  ./run_app.sh
  ```

*Note: The first launch may take some time as it downloads the initial OCR models.*

---

## 🚀 Usage

After setting up the environment, running `run_app.bat` (or `run_app.sh`) again will automatically launch the GUI application.

### Basic Workflow
1. **Configure Paths**
   * **Input Dir**: Folder containing the scanned forms to process (Default: `./input`).
   * **Output Dir**: Folder where processed results and cropped images will be saved (Default: `./output`).
   * **Template Image**: The standard format image used as a reference point (Default: `./template/template.png`).
   * **Masks Directory**: Folder containing the coordinate data and images of the regions to extract (Default: `./masks`).

2. **Prepare Masks (Sample files are included)**
   * The repository includes sample masks and a template image.
   * Clicking "Scan Masks Directory" will automatically calculate rectangular regions from the images inside the `masks/` folder and add them to the table.
   * You can select a row and click "Edit Region" to visually adjust the reading area.
   * You can configure each region to read "digits" or "text", and enforce fixed-length string validation.

3. **Start Processing**
   * Once configured, click "Start Processing".
   * The application will process all images in the `input` directory: correcting rotation, aligning with the template, cropping regions, and executing OCR recognition.
   * The final results are saved in `output/results.json`.

---

## 🏗️ For Developers: How to Add a New Engine

If you wish to add a new OCR engine, you can implement it in the following steps:

1. Open `src/ocr_engine.py`.
2. Create a new class (e.g., `NextGenOcrStrategy`) inheriting from `IOcrStrategy`, and implement the `read(self, image: np.ndarray, expected_length: int)` method.
3. Add a new branch in the `OcrEngineFactory` class to return your newly created strategy class.
4. That's it! The backend will seamlessly switch to your new engine.

---

## 📝 Project Structure

```text
smart-form-reader/
├── src/                        # Core Logic
│   ├── image_processing.py     # Alignment & Cropping Layer
│   └── ocr_engine.py           # OCR Strategy & Factory Layer
├── tests/                      # Unit Tests
├── docs/                       # Documentation
├── main_gui.py                 # Tkinter GUI Main Application
├── setup_env.py                # Environment Setup Script
├── run_app.bat / .sh           # App Launch Scripts
├── template/                   # Reference Template Image (Sample Included)
├── masks/                      # Extraction Region Masks (Sample Included)
├── input/                      # Target Images for Processing
└── output/                     # Output Destination
```
