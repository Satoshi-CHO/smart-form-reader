import os
import sys
import subprocess
import platform

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
        "manga-ocr",
        "opencv-python",
        "numpy"
    ]
    
    print("Installing other required packages...")
    for req in other_requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

if __name__ == "__main__":
    print("Environment setup started...")
    install_requirements()
    
    # Create the setup_complete flag file
    flag_path = os.path.join("venv", ".setup_complete")
    with open(flag_path, "w") as f:
        f.write("Setup completed successfully.\n")
        
    print("Environment setup completed successfully.")
