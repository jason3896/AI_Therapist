@echo off
REM update_batch_paths.bat - Update paths in batch files for AI_Therapist directory structure

echo Updating paths in batch files...
echo =============================================

set BATCH_DIR=.\src\facial
set MODEL_PATH=%BATCH_DIR%\checkpoint\model_best.pth.tar
set TEST_DIR=%BATCH_DIR%\test_image
set OUTPUT_DIR=%BATCH_DIR%\test_results
set ERROR_DIR=%BATCH_DIR%\error_analysis_results
set DATA_DIR=%BATCH_DIR%\RAF-DB\DATASET

REM 1. Update run_main.bat
echo @echo off > run_main.bat
echo REM Windows batch equivalent of run_main.sh >> run_main.bat
echo. >> run_main.bat
echo python .\src\facial\main.py ^^ >> run_main.bat
echo     --data ".\src\facial\RAF-DB\DATASET" ^^ >> run_main.bat
echo     --batch-size 64 ^^ >> run_main.bat
echo     --lr 0.0001 ^^ >> run_main.bat
echo     --epochs 50 ^^ >> run_main.bat
echo     --use-class-balanced ^^ >> run_main.bat
echo     --use-focal-loss ^^ >> run_main.bat
echo     --beta 0.99 ^^ >> run_main.bat
echo     --gamma 3.5 ^^ >> run_main.bat
echo     --two-stage ^^ >> run_main.bat
echo     --use-label-smoothing ^^ >> run_main.bat
echo     --label-smoothing 0.1 ^^ >> run_main.bat
echo     --use-temperature-scaling ^^ >> run_main.bat
echo     --save-error-samples ^^ >> run_main.bat
echo     --error-samples-dir ".\src\facial\error_samples" ^^ >> run_main.bat
echo     --grad-clip 1.0 ^^ >> run_main.bat
echo     --save-tsne ^^ >> run_main.bat
echo     --save-confidence-hist ^^ >> run_main.bat
echo     --use-ensemble ^^ >> run_main.bat
echo     --num-ensemble-models 3 >> run_main.bat
echo. >> run_main.bat
echo echo Training completed! >> run_main.bat
echo pause >> run_main.bat

echo Created run_main.bat

REM 2. Update temperature.bat
echo @echo off > temperature.bat
echo REM Windows batch equivalent of temperature.sh >> temperature.bat
echo. >> temperature.bat
echo REM Check if CUDA is available >> temperature.bat
echo python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> temperature.bat
echo if %%errorlevel%% equ 0 ( >> temperature.bat
echo     set device=cuda >> temperature.bat
echo     echo Using CUDA for temperature calibration >> temperature.bat
echo ) else ( >> temperature.bat
echo     set device=cpu >> temperature.bat
echo     echo CUDA not available, using CPU for temperature calibration >> temperature.bat
echo ) >> temperature.bat
echo. >> temperature.bat
echo python .\src\facial\temperature_calibration.py --data ".\src\facial\RAF-DB\DATASET\test" --model ".\src\facial\checkpoint\model_best.pth.tar" --device %%device%% >> temperature.bat
echo. >> temperature.bat
echo echo Temperature calibration completed! >> temperature.bat
echo pause >> temperature.bat

echo Created temperature.bat

REM 3. Update error.bat
echo @echo off > error.bat
echo REM Windows batch equivalent of error.sh >> error.bat
echo. >> error.bat
echo REM Check if CUDA is available >> error.bat
echo python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> error.bat
echo if %%errorlevel%% equ 0 ( >> error.bat
echo     set device=cuda >> error.bat
echo     echo Using CUDA for error analysis >> error.bat
echo ) else ( >> error.bat
echo     set device=cpu >> error.bat
echo     echo CUDA not available, using CPU for error analysis >> error.bat
echo ) >> error.bat
echo. >> error.bat
echo python .\src\facial\error_analyzer.py ^^ >> error.bat
echo   --data ".\src\facial\RAF-DB\DATASET\test" ^^ >> error.bat
echo   --model ".\src\facial\checkpoint\model_best.pth.tar" ^^ >> error.bat
echo   --output-dir ".\src\facial\error_analysis_results" ^^ >> error.bat
echo   --device %%device%% ^^ >> error.bat
echo   --visualize ^^ >> error.bat
echo   --temperature 1.5 >> error.bat
echo. >> error.bat
echo echo Error analysis completed! >> error.bat
echo pause >> error.bat

echo Created error.bat

REM 4. Update realtime.bat
echo @echo off > realtime.bat
echo REM Windows batch script for running realtime emotion detection >> realtime.bat
echo. >> realtime.bat
echo REM Check if model exists >> realtime.bat
echo if not exist ".\src\facial\checkpoint\model_best.pth.tar" ( >> realtime.bat
echo     echo Error: Model file not found! >> realtime.bat
echo     echo Please ensure model_best.pth.tar exists in the checkpoint directory. >> realtime.bat
echo     pause >> realtime.bat
echo     exit /b 1 >> realtime.bat
echo ) >> realtime.bat
echo. >> realtime.bat
echo REM Check if CUDA is available >> realtime.bat
echo python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> realtime.bat
echo if %%errorlevel%% equ 0 ( >> realtime.bat
echo     set device=cuda >> realtime.bat
echo     echo CUDA is available. Using GPU acceleration. >> realtime.bat
echo ) else ( >> realtime.bat
echo     set device=cpu >> realtime.bat
echo     echo CUDA not available. Using CPU. >> realtime.bat
echo ) >> realtime.bat
echo. >> realtime.bat
echo REM Select face detection model >> realtime.bat
echo set /p face_model="Use DNN face detector? (y/n, default=n): " >> realtime.bat
echo if /i "%%face_model%%"=="y" ( >> realtime.bat
echo     set detector=dnn >> realtime.bat
echo     >> realtime.bat
echo     REM Check if DNN model files exist >> realtime.bat
echo     if not exist ".\src\facial\deploy.prototxt" ( >> realtime.bat
echo         echo DNN model files not found. Attempting to download... >> realtime.bat
echo         python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', '.\src\facial\deploy.prototxt'); urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', '.\src\facial\res10_300x300_ssd_iter_140000.caffemodel')" >> realtime.bat
echo     ) >> realtime.bat
echo ) else ( >> realtime.bat
echo     set detector=haarcascade >> realtime.bat
echo ) >> realtime.bat
echo. >> realtime.bat
echo REM Ask for camera index >> realtime.bat
echo set /p camera="Enter camera index (default=0): " >> realtime.bat
echo if "%%camera%%"=="" set camera=0 >> realtime.bat
echo. >> realtime.bat
echo REM Run the application >> realtime.bat
echo python .\src\facial\realtime_emotion.py ^^ >> realtime.bat
echo     --model ".\src\facial\checkpoint\model_best.pth.tar" ^^ >> realtime.bat
echo     --device %%device%% ^^ >> realtime.bat
echo     --camera %%camera%% ^^ >> realtime.bat
echo     --face-detection-model %%detector%% >> realtime.bat
echo. >> realtime.bat
echo echo Real-time emotion detection finished. >> realtime.bat
echo pause >> realtime.bat

echo Created realtime.bat

REM 5. Update test_emotion.bat
echo @echo off > test_emotion.bat
echo REM Windows batch script for testing emotion recognition on images >> test_emotion.bat
echo. >> test_emotion.bat
echo REM Check if model exists >> test_emotion.bat
echo if not exist ".\src\facial\checkpoint\model_best.pth.tar" ( >> test_emotion.bat
echo     echo Error: Model file not found! >> test_emotion.bat
echo     echo Please ensure model_best.pth.tar exists in the checkpoint directory. >> test_emotion.bat
echo     pause >> test_emotion.bat
echo     exit /b 1 >> test_emotion.bat
echo ) >> test_emotion.bat
echo. >> test_emotion.bat
echo REM Check if test image directory exists >> test_emotion.bat
echo if not exist ".\src\facial\test_image" ( >> test_emotion.bat
echo     echo Creating test_image directory... >> test_emotion.bat
echo     mkdir ".\src\facial\test_image" >> test_emotion.bat
echo     echo Please place test images in the test_image directory. >> test_emotion.bat
echo     pause >> test_emotion.bat
echo     exit /b 1 >> test_emotion.bat
echo ) >> test_emotion.bat
echo. >> test_emotion.bat
echo REM Check if test images exist >> test_emotion.bat
echo dir /b ".\src\facial\test_image\*.jpg" ".\src\facial\test_image\*.png" ".\src\facial\test_image\*.jpeg" 2^>nul | find /v "" ^>nul >> test_emotion.bat
echo if %%errorlevel%% neq 0 ( >> test_emotion.bat
echo     echo No image files found in test_image directory. >> test_emotion.bat
echo     echo Please add some images (JPG, PNG, JPEG) to the test_image folder. >> test_emotion.bat
echo     pause >> test_emotion.bat
echo     exit /b 1 >> test_emotion.bat
echo ) >> test_emotion.bat
echo. >> test_emotion.bat
echo REM Check if CUDA is available >> test_emotion.bat
echo python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> test_emotion.bat
echo if %%errorlevel%% equ 0 ( >> test_emotion.bat
echo     set device=cuda >> test_emotion.bat
echo     echo CUDA is available. Using GPU acceleration. >> test_emotion.bat
echo ) else ( >> test_emotion.bat
echo     set device=cpu >> test_emotion.bat
echo     echo CUDA not available. Using CPU. >> test_emotion.bat
echo ) >> test_emotion.bat
echo. >> test_emotion.bat
echo REM Run the test script >> test_emotion.bat
echo python .\src\facial\test_emotion.py ^^ >> test_emotion.bat
echo     --model ".\src\facial\checkpoint\model_best.pth.tar" ^^ >> test_emotion.bat
echo     --test-dir ".\src\facial\test_image" ^^ >> test_emotion.bat
echo     --output-dir ".\src\facial\test_results" ^^ >> test_emotion.bat
echo     --device %%device%% ^^ >> test_emotion.bat
echo     --visualize ^^ >> test_emotion.bat
echo     --temperature 1.5 >> test_emotion.bat
echo. >> test_emotion.bat
echo echo Test emotion recognition completed. Results saved to test_results directory. >> test_emotion.bat
echo pause >> test_emotion.bat

echo Created test_emotion.bat

REM 6. Update setup.bat
echo @echo off > setup.bat
echo REM setup.bat - Install and configure emotion recognition system on Windows >> setup.bat
echo echo Emotion Recognition System - Setup Script >> setup.bat
echo echo ============================================= >> setup.bat
echo. >> setup.bat
echo REM Check if Python is installed >> setup.bat
echo python --version ^> nul 2^>^&1 >> setup.bat
echo if %%errorlevel%% neq 0 ( >> setup.bat
echo     echo Error: Python not found! >> setup.bat
echo     echo Please install Python 3.8+ from python.org >> setup.bat
echo     echo Make sure to check "Add Python to PATH" during installation >> setup.bat
echo     goto :error >> setup.bat
echo ) >> setup.bat
echo. >> setup.bat
echo echo Python detected. Checking and installing required packages... >> setup.bat
echo. >> setup.bat
echo REM Create required directories >> setup.bat
echo echo Creating directories... >> setup.bat
echo if not exist ".\src\facial\checkpoint" mkdir ".\src\facial\checkpoint" >> setup.bat
echo if not exist ".\src\facial\error_samples" mkdir ".\src\facial\error_samples" >> setup.bat
echo if not exist ".\src\facial\error_analysis_results" mkdir ".\src\facial\error_analysis_results" >> setup.bat
echo if not exist ".\src\facial\test_image" mkdir ".\src\facial\test_image" >> setup.bat
echo if not exist ".\src\facial\test_results" mkdir ".\src\facial\test_results" >> setup.bat
echo if not exist ".\src\facial\log" mkdir ".\src\facial\log" >> setup.bat
echo. >> setup.bat
echo REM Install required packages >> setup.bat
echo echo Installing required Python packages... >> setup.bat
echo pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 >> setup.bat
echo pip install opencv-python matplotlib pillow numpy scikit-learn pandas seaborn tqdm >> setup.bat
echo. >> setup.bat
echo REM Download face detection models for DNN detector >> setup.bat
echo echo Downloading face detection models... >> setup.bat
echo if not exist ".\src\facial\deploy.prototxt" ( >> setup.bat
echo     echo Downloading deploy.prototxt... >> setup.bat
echo     powershell -Command "(New-Object Net.WebClient).DownloadFile('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', '.\src\facial\deploy.prototxt')" >> setup.bat
echo ) >> setup.bat
echo. >> setup.bat
echo if not exist ".\src\facial\res10_300x300_ssd_iter_140000.caffemodel" ( >> setup.bat
echo     echo Downloading res10_300x300_ssd_iter_140000.caffemodel... >> setup.bat
echo     powershell -Command "(New-Object Net.WebClient).DownloadFile('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', '.\src\facial\res10_300x300_ssd_iter_140000.caffemodel')" >> setup.bat
echo ) >> setup.bat
echo. >> setup.bat
echo REM Check CUDA availability >> setup.bat
echo echo Checking for CUDA support... >> setup.bat
echo python -c "import torch; print('CUDA Available: ' + str(torch.cuda.is_available()))" >> setup.bat
echo. >> setup.bat
echo REM Check for existing model file >> setup.bat
echo if exist ".\src\facial\checkpoint\model_best.pth.tar" ( >> setup.bat
echo     echo Found existing model file in checkpoint directory. >> setup.bat
echo ) else ( >> setup.bat
echo     echo No model file found in checkpoint directory. >> setup.bat
echo     echo You will need to train a model or copy your pre-trained model to: >> setup.bat
echo     echo   .\src\facial\checkpoint\model_best.pth.tar >> setup.bat
echo ) >> setup.bat
echo. >> setup.bat
echo REM Verify RAF-DB dataset path >> setup.bat
echo if exist ".\src\facial\RAF-DB\DATASET" ( >> setup.bat
echo     echo Found RAF-DB dataset directory. >> setup.bat
echo ) else ( >> setup.bat
echo     echo Warning: RAF-DB dataset not found at .\src\facial\RAF-DB\DATASET >> setup.bat
echo     echo Make sure to place the dataset in this location before training. >> setup.bat
echo ) >> setup.bat
echo. >> setup.bat
echo echo. >> setup.bat
echo echo Setup completed successfully! >> setup.bat
echo echo You can now run the batch scripts to use the emotion recognition system. >> setup.bat
echo echo See README.txt for usage instructions. >> setup.bat
echo echo ============================================= >> setup.bat
echo pause >> setup.bat
echo goto :eof >> setup.bat
echo. >> setup.bat
echo :error >> setup.bat
echo echo. >> setup.bat
echo echo Setup failed. Please fix the errors and try again. >> setup.bat
echo pause >> setup.bat

echo Created setup.bat

REM 7. Create a master run_all.bat script
echo @echo off > run_all.bat
echo REM Master batch script to run all facial emotion recognition tasks >> run_all.bat
echo. >> run_all.bat
echo echo Emotion Recognition System - Main Menu >> run_all.bat
echo echo ======================================= >> run_all.bat
echo echo. >> run_all.bat
echo echo Select a task to run: >> run_all.bat
echo echo 1) Setup environment and install dependencies >> run_all.bat
echo echo 2) Train emotion recognition model >> run_all.bat
echo echo 3) Calibrate temperature scaling >> run_all.bat
echo echo 4) Run error analysis >> run_all.bat
echo echo 5) Test on static images >> run_all.bat
echo echo 6) Run real-time emotion detection >> run_all.bat
echo echo 7) Exit >> run_all.bat
echo echo. >> run_all.bat
echo set /p choice="Enter your choice (1-7): " >> run_all.bat
echo. >> run_all.bat
echo if "%%choice%%"=="1" goto setup >> run_all.bat
echo if "%%choice%%"=="2" goto train >> run_all.bat
echo if "%%choice%%"=="3" goto temperature >> run_all.bat
echo if "%%choice%%"=="4" goto error >> run_all.bat
echo if "%%choice%%"=="5" goto test >> run_all.bat
echo if "%%choice%%"=="6" goto realtime >> run_all.bat
echo if "%%choice%%"=="7" goto end >> run_all.bat
echo. >> run_all.bat
echo echo Invalid choice. Please try again. >> run_all.bat
echo pause >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :setup >> run_all.bat
echo call setup.bat >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :train >> run_all.bat
echo call run_main.bat >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :temperature >> run_all.bat
echo call temperature.bat >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :error >> run_all.bat
echo call error.bat >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :test >> run_all.bat
echo call test_emotion.bat >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :realtime >> run_all.bat
echo call realtime.bat >> run_all.bat
echo goto :eof >> run_all.bat
echo. >> run_all.bat
echo :end >> run_all.bat
echo echo Goodbye! >> run_all.bat

echo Created run_all.bat

echo.
echo All batch files have been updated with the correct paths.
echo You can now run them from the root directory of your project.
echo =============================================
pause