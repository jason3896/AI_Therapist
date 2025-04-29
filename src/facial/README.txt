Instructions for Setting Up Emotion Recognition System on Windows
Follow these step-by-step instructions to update your emotion recognition scripts and prepare them for Windows:
Step 1: Save the Scripts

Save the path_patcher.py file to your project root directory (AI_Therapist folder)
Save the update_batch_paths.bat file to your project root directory

Step 2: Run the Path Patcher

Open Command Prompt
Navigate to your project root directory:
cd path\to\AI_Therapist

Run the Python script:
python path_patcher.py

The script will:

Create backups of your original .py files with .bak extension
Update paths in all Python files
Print messages about each file it updates



Step 3: Run the Batch File Updater

In the same Command Prompt window, run:
update_batch_paths.bat

This will create all necessary batch files:

setup.bat
run_main.bat
temperature.bat
error.bat
realtime.bat
test_emotion.bat
run_all.bat



Step 4: Set Up Your Environment

Run the setup script:
setup.bat

The script will:

Check for Python installation
Create required directories
Install necessary packages
Download face detection models
Verify CUDA availability



Step 5: Place Required Files

Ensure your dataset is in the proper location:
src\facial\RAF-DB\DATASET\

Make sure your model files are in the checkpoint directory:
src\facial\checkpoint\model_best.pth.tar


Step 6: Run the System
You can now run any of the batch files:

run_all.bat - Shows a menu with all options
run_main.bat - Train the emotion recognition model
temperature.bat - Calibrate temperature scaling
error.bat - Run error analysis
test_emotion.bat - Test on static images
realtime.bat - Run real-time emotion detection

All scripts are configured to run from the root directory of your project.
Note on File Changes

Python Files: All paths in the Python files have been updated to include the src/facial prefix
Batch Files: Created with proper Windows-style paths using backslashes
Backup Files: Original Python files are backed up with .bak extension in case you need to restore them