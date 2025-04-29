@echo off
REM setup.bat - Install and configure emotion recognition system on Windows
echo Emotion Recognition System - Setup Script
echo =============================================

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found!
    echo Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH" during installation
    goto :error
)

echo Python detected. Checking and installing required packages...

REM Create required directories
echo Creating directories...
if not exist ".\src\facial\checkpoint" mkdir ".\src\facial\checkpoint"
if not exist ".\src\facial\error_samples" mkdir ".\src\facial\error_samples"
if not exist ".\src\facial\error_analysis_results" mkdir ".\src\facial\error_analysis_results"
if not exist ".\src\facial\test_image" mkdir ".\src\facial\test_image"
if not exist ".\src\facial\test_results" mkdir ".\src\facial\test_results"
if not exist ".\src\facial\log" mkdir ".\src\facial\log"

REM Install required packages
echo Installing required Python packages...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib pillow numpy scikit-learn pandas seaborn tqdm

REM Download face detection models for DNN detector
echo Downloading face detection models...
if not exist ".\src\facial\deploy.prototxt" (
    echo Downloading deploy.prototxt...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', '.\src\facial\deploy.prototxt')"
)

if not exist ".\src\facial\res10_300x300_ssd_iter_140000.caffemodel" (
    echo Downloading res10_300x300_ssd_iter_140000.caffemodel...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', '.\src\facial\res10_300x300_ssd_iter_140000.caffemodel')"
)

REM Check CUDA availability
echo Checking for CUDA support...
python -c "import torch; print('CUDA Available: ' + str(torch.cuda.is_available()))"

REM Check for existing model file
if exist ".\src\facial\checkpoint\model_best.pth.tar" (
    echo Found existing model file in checkpoint directory.
) else (
    echo No model file found in checkpoint directory.
    echo You will need to train a model or copy your pre-trained model to:
    echo   .\src\facial\checkpoint\model_best.pth.tar
)

REM Verify RAF-DB dataset path
if exist ".\src\facial\RAF-DB\DATASET" (
    echo Found RAF-DB dataset directory.
) else (
    echo Warning: RAF-DB dataset not found at .\src\facial\RAF-DB\DATASET
    echo Make sure to place the dataset in this location before training.
)

echo.
echo Setup completed successfully!
echo You can now run the batch scripts to use the emotion recognition system.
echo See README.txt for usage instructions.
echo =============================================
pause
goto :eof

:error
echo.
echo Setup failed. Please fix the errors and try again.
pause