import os
import re
import sys
import shutil

def patch_file_paths(file_path, replacements):
    """
    Replace path patterns in a file with new paths.
    
    Args:
        file_path: Path to the file to modify
        replacements: List of tuples (pattern, replacement)
    """
    # First check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False
    
    # Create backup
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply replacements
    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    else:
        print(f"No changes needed for: {file_path}")
        return False

def main():
    # Directory structure to adjust
    base_dir = "."  # Update this if running from a different directory
    
    # File paths
    error_analyzer_path = os.path.join(base_dir, "src", "facial", "error_analyzer.py")
    main_path = os.path.join(base_dir, "src", "facial", "main.py")
    realtime_emotion_path = os.path.join(base_dir, "src", "facial", "realtime_emotion.py")
    temp_calibration_path = os.path.join(base_dir, "src", "facial", "temperature_calibration.py")
    test_emotion_path = os.path.join(base_dir, "src", "facial", "test_emotion.py")
    
    # Path replacements for error_analyzer.py
    error_analyzer_replacements = [
        # Update output directory default
        (r"default='./error_analysis'", r"default='./src/facial/error_analysis'"),
        # Update checkpoint file paths in any command line arguments
        (r"--model (.+?)\.pth\.tar", r"--model \1.pth.tar"),
        # Update face detection model paths
        (r"face_cascade = cv2\.CascadeClassifier\(cv2\.data\.haarcascades \+ 'haarcascade_frontalface_default\.xml'\)", 
         r"face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')"),
    ]
    
    # Path replacements for main.py
    main_replacements = [
        # Update checkpoint paths
        (r"default='./checkpoint/model.pth.tar'", r"default='./src/facial/checkpoint/model.pth.tar'"),
        (r"default='./checkpoint/model_best.pth.tar'", r"default='./src/facial/checkpoint/model_best.pth.tar'"),
        # Update error samples directory
        (r"default='./error_samples'", r"default='./src/facial/error_samples'"),
        # Update log directory operations
        (r"os\.makedirs\('./checkpoint', exist_ok=True\)", r"os.makedirs('./src/facial/checkpoint', exist_ok=True)"),
        (r"os\.makedirs\('./log', exist_ok=True\)", r"os.makedirs('./src/facial/log', exist_ok=True)"),
        # Update fine model checkpoint path
        (r"'./checkpoint/fine_model.pth.tar'", r"'./src/facial/checkpoint/fine_model.pth.tar'"),
        # Update ensemble model checkpoint paths
        (r"'./checkpoint/' \+ f'ensemble_model_{i\+1}\.pth\.tar'", r"'./src/facial/checkpoint/' + f'ensemble_model_{i+1}.pth.tar'"),
        # Update pretrained model loading paths
        (r"torch\.load\('./chekpoint/Pretrained_EfficientFace\.tar'", r"torch.load('./src/facial/checkpoint/Pretrained_EfficientFace.tar'"),
        (r"torch\.load\('./chekpoint/Pretrained_LDG\.tar'", r"torch.load('./src/facial/checkpoint/Pretrained_LDG.tar'"),
        # Update log file paths
        (r"txt_name = './log/' \+ time_str \+ 'log\.txt'", r"txt_name = './src/facial/log/' + time_str + 'log.txt'"),
        (r"curve_name = time_str \+ 'log\.png'", r"curve_name = time_str + 'log.png'"),
        (r"recorder\.plot_curve\(os\.path\.join\('./log/', curve_name\)\)", r"recorder.plot_curve(os.path.join('./src/facial/log/', curve_name))"),
    ]
    
    # Path replacements for realtime_emotion.py
    realtime_replacements = [
        # Update face detector model paths
        (r"face_net = cv2\.dnn\.readNetFromCaffe\(\s+os\.path\.join\(os\.path\.dirname\(__file__\), 'deploy\.prototxt'\),\s+os\.path\.join\(os\.path\.dirname\(__file__\), 'res10_300x300_ssd_iter_140000\.caffemodel'\)\s+\)",
         r"face_net = cv2.dnn.readNetFromCaffe(\n            os.path.join(os.path.dirname(__file__), 'deploy.prototxt'),\n            os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel')\n        )"),
    ]
    
    # Path replacements for temperature_calibration.py
    temp_calibration_replacements = [
        # Update output path
        (r"default='./calibrated_model.pth.tar'", r"default='./src/facial/calibrated_model.pth.tar'"),
    ]
    
    # Path replacements for test_emotion.py
    test_emotion_replacements = [
        # Update test and output directories
        (r"default='./test_image'", r"default='./src/facial/test_image'"),
        (r"default='./test_results'", r"default='./src/facial/test_results'"),
    ]
    
    # Apply patches
    patch_file_paths(error_analyzer_path, error_analyzer_replacements)
    patch_file_paths(main_path, main_replacements)
    patch_file_paths(realtime_emotion_path, realtime_replacements)
    patch_file_paths(temp_calibration_path, temp_calibration_replacements)
    patch_file_paths(test_emotion_path, test_emotion_replacements)
    
    print("\nAll files have been processed. Backups with .bak extension were created.")
    print("Please check the files to make sure the changes are correct.")

if __name__ == "__main__":
    main()