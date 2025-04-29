import os
import sys
import cv2
import torch
import subprocess
import platform

def detect_available_cameras():
    """Detect available camera devices on Windows"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def select_best_device():
    """Automatically select the best available device on Windows"""
    if torch.cuda.is_available():
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"Found CUDA-capable GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
        return "cuda"
    else:
        print("No CUDA-capable GPU found, using CPU")
        return "cpu"

def check_windows_camera_drivers():
    """Check if camera drivers are properly installed on Windows"""
    # This is a basic check - it just verifies if any camera can be opened
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("WARNING: Could not open camera. Please check camera drivers are installed correctly.")
        print("You can verify camera functionality in Windows Camera app.")
        return False
    cap.release()
    return True

def set_process_priority(high_priority=True):
    """Set process priority for better performance on Windows"""
    if platform.system() != 'Windows':
        return False
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        if high_priority:
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            print("Process priority set to HIGH")
        else:
            process.nice(psutil.NORMAL_PRIORITY_CLASS)
            print("Process priority set to NORMAL")
        return True
    except (ImportError, PermissionError) as e:
        print(f"Could not set process priority: {e}")
        return False

def configure_torch_num_threads(num_threads=None):
    """Configure number of threads for PyTorch on Windows"""
    if num_threads is None:
        # Use half of available CPU cores by default
        import multiprocessing
        num_threads = max(1, multiprocessing.cpu_count() // 2)
    
    torch.set_num_threads(num_threads)
    print(f"Set PyTorch to use {num_threads} threads")
    return num_threads

def download_models(download_dir='.'):
    """Download necessary model files for face detection on Windows"""
    import urllib.request
    
    models = {
        'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    }
    
    for filename, url in models.items():
        filepath = os.path.join(download_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists")

def check_webcam_resolution(camera_idx=0):
    """Check and report webcam resolution on Windows"""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Could not open camera {camera_idx}")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera {camera_idx} resolution: {width}x{height} @ {fps} FPS")
    cap.release()
    return (width, height), fps

# Example usage in main script:
if __name__ == "__main__":
    print("Windows Helper Utilities for Emotion Recognition")
    print("-" * 50)
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    print("\nDetecting available cameras...")
    cameras = detect_available_cameras()
    if cameras:
        print(f"Found cameras at indices: {cameras}")
        for cam_idx in cameras:
            check_webcam_resolution(cam_idx)
    else:
        print("No cameras detected")
    
    print("\nRecommended device:", select_best_device())
    
    print("\nDownloading required model files...")
    download_models()
    
    print("\nSetup complete! You can now run the emotion recognition scripts.")