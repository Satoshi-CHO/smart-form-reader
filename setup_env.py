import os
import sys
import subprocess
import platform
import argparse


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
EASYOCR_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "easyocr")
NDLOCR_ROOT = os.path.join(PROJECT_ROOT, ".tools", "ndlocr-lite")
NDLOCR_REPO_DIR = os.path.join(NDLOCR_ROOT, "repo")
NDLOCR_VENV_DIR = os.path.join(NDLOCR_ROOT, "venv")
NDLOCR_REPO_URL = "https://github.com/ndl-lab/ndlkotenocr-lite.git"


def configure_easyocr_cache():
    os.makedirs(EASYOCR_CACHE_DIR, exist_ok=True)
    os.environ["EASYOCR_MODULE_PATH"] = EASYOCR_CACHE_DIR
    print(f"EasyOCR cache directory: {EASYOCR_CACHE_DIR}")


def warmup_easyocr_models():
    print("Pre-downloading EasyOCR models...")
    configure_easyocr_cache()
    import easyocr

    # Digits and Japanese text are both used by the application.
    easyocr.Reader(['en'], gpu=False)
    easyocr.Reader(['ja', 'en'], gpu=False)


def get_venv_python(venv_dir):
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def install_ndlocr_lite():
    print("Installing NDLOCR-Lite into project-managed tools directory...")
    os.makedirs(NDLOCR_ROOT, exist_ok=True)

    if not os.path.isdir(NDLOCR_REPO_DIR):
        subprocess.check_call(["git", "clone", NDLOCR_REPO_URL, NDLOCR_REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", NDLOCR_REPO_DIR, "pull", "--ff-only"])

    if not os.path.isdir(NDLOCR_VENV_DIR):
        subprocess.check_call([sys.executable, "-m", "venv", NDLOCR_VENV_DIR])

    ndlocr_python = get_venv_python(NDLOCR_VENV_DIR)
    requirements_path = os.path.join(NDLOCR_REPO_DIR, "requirements.txt")
    subprocess.check_call([ndlocr_python, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([ndlocr_python, "-m", "pip", "install", "-r", requirements_path])

    print(f"NDLOCR-Lite ready at: {os.path.join(NDLOCR_REPO_DIR, 'src', 'ocr.py')}")

def install_requirements():
    print("Installing requirements...")
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Auto-detect GPU (NVIDIA) presence
    try:
        subprocess.check_output(['nvidia-smi'])
        print("NVIDIA GPU detected. Installing PyTorch and PaddlePaddle with CUDA support...")
        is_gpu = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No NVIDIA GPU detected. Installing CPU versions...")
        is_gpu = False

    # Install PyTorch based on OS and GPU
    if platform.system() == 'Windows' or platform.system() == 'Linux':
        if is_gpu:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        # macOS
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])

    # Install other packages
    other_requirements = [
        "easyocr",
        "opencv-python",
        "numpy"
    ]
    
    print("Installing other required packages...")
    for req in other_requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

    warmup_easyocr_models()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-ndlocr-lite", action="store_true")
    args = parser.parse_args()

    if args.install_ndlocr_lite:
        install_ndlocr_lite()
        return

    print("Environment setup started...")
    configure_easyocr_cache()
    install_requirements()

    flag_path = os.path.join("venv", ".setup_complete")
    with open(flag_path, "w") as f:
        f.write("Setup completed successfully.\n")

    print("Environment setup completed successfully.")


if __name__ == "__main__":
    main()
