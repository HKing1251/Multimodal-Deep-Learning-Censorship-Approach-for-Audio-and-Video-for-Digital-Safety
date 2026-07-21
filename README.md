# Multimodal Video Censorship System

Final Year Project by **Chong Chun Kit (TP077436)**

This Windows desktop application accepts an MP4 video, detects inappropriate visual and spoken content, and produces a censored MP4 output.

The system uses:

- **YOLOv8s-cls** to classify video frames as Safe or NSFW
- **AudioCNN** to screen one-second audio windows for profanity-like acoustic patterns
- **Whisper Base** to transcribe speech, identify prohibited words, and obtain word timestamps
- **OpenCV** to blur NSFW frames
- **Pydub** to mute confirmed profanity intervals
- **MoviePy and FFmpeg** to extract, process, and merge media
- **Tkinter** to provide the desktop graphical interface

The final implementation uses **modality-specific decision-level integration**. Visual and audio censorship decisions are performed separately, and the censored video and audio streams are merged into `final_output.mp4`.

---

## 1. System Requirements

### Operating system

The application was developed and tested on:

- Windows 10, 64-bit

### Hardware used during development

- NVIDIA GeForce GTX 1650
- 4 GB GPU memory
- CUDA-enabled PyTorch

A CUDA-compatible NVIDIA GPU is recommended for faster YOLOv8, AudioCNN, and Whisper processing. CPU execution may be possible but will be considerably slower.

### Python version

The system was developed using:

```text
Python 3.8
```

Use the same Python environment that was used to build and test the application.

---

## 2. Required Python Libraries

Open Command Prompt or PowerShell in the project folder and install the required packages.

### Core runtime packages

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy
pip install librosa
pip install openai-whisper
pip install moviepy==1.0.3
pip install pydub
```

### Optional drag-and-drop support

```bash
pip install tkinterdnd2
```

The application still works without `tkinterdnd2`. Users can select a video through the **Insert Video** button.

### Executable packaging

```bash
pip install pyinstaller
```

### Development and evaluation packages

These packages are mainly required for model evaluation, graphs, confusion matrices, and notebooks:

```bash
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install jupyter
```

### Tested PyTorch environment

The project was tested with:

```text
PyTorch 2.4.1+cu121
CUDA available: True
```

For GPU execution, install a PyTorch build compatible with the CUDA version supported by the computer. A CPU-only PyTorch installation can also run the code, but processing will be slower.

---

## 3. FFmpeg Requirement

The project includes a local FFmpeg folder:

```text
ffmpeg/
└── bin/
    ├── ffmpeg.exe
    ├── ffplay.exe
    └── ffprobe.exe
```

Do not delete or move this folder. The application automatically adds `ffmpeg/bin` to the runtime PATH.

The standalone `ffmpeg.exe` visible in the main source folder is not required by the final application when the complete `ffmpeg/bin` folder is present.

---

## 4. Source Project Folder Structure

Before running the source code or building the executable, the project folder should follow this structure:

```text
FYP CODE/
├── Build_exe.py
├── milestone1_video_censorship.py
├── milestone2_audio_censorship.py
├── milestone3_multimedia.py
├── milestone4_system.py
├── bad_words.txt
│
├── Models/
│   ├── Visual/
│   │   └── best.pt
│   └── Audio/
│       └── best_model.pt
│
├── ffmpeg/
│   └── bin/
│       ├── ffmpeg.exe
│       ├── ffplay.exe
│       └── ffprobe.exe
│
├── build/                 # PyInstaller temporary build files
└── dist/                  # Final packaged application
```

### File purposes

| File or folder | Purpose |
|---|---|
| `milestone1_video_censorship.py` | Classifies video frames using YOLOv8s-cls and blurs NSFW frames |
| `milestone2_audio_censorship.py` | Runs AudioCNN and Whisper, combines their decisions, and mutes confirmed profanity |
| `milestone3_multimedia.py` | Merges the censored video and censored audio |
| `milestone4_system.py` | Main Tkinter GUI and pipeline controller |
| `Build_exe.py` | Builds the Windows executable and copies runtime files |
| `bad_words.txt` | Prohibited-word dictionary used by Whisper matching |
| `Models/Visual/best.pt` | Trained YOLOv8s-cls model weights |
| `Models/Audio/best_model.pt` | Trained AudioCNN model weights |
| `ffmpeg/bin` | Local FFmpeg executables |
| `build` | Temporary PyInstaller working files; not distributed |
| `dist` | Final deployable application folder |

---

## 5. Running the System from Python Source

### Step 1: Open the project folder

```bash
cd "PATH\TO\FYP CODE"
```

Example:

```bash
cd "C:\Users\user\Documents\APU FINAL YEAR\Y3S2\FYP\Code\FYP CODE"
```

### Step 2: Activate the correct Python environment

Example for a virtual environment:

```bash
.venv\Scripts\activate
```

### Step 3: Verify the required packages

```bash
python -c "import cv2, torch, ultralytics, whisper, librosa, moviepy, pydub; print('Required libraries loaded successfully')"
```

### Step 4: Start the graphical application

```bash
python milestone4_system.py
```

### Step 5: Use the application

1. Click **Insert Video**.
2. Select an MP4 video.
3. Click **Start Censorship**.
4. Wait for the visual, audio, and merge stages to complete.
5. Click **Download Video**.
6. Select where to save the final censored MP4 file.

---

## 6. Running Individual Processing Modules

These commands are mainly useful for testing and debugging.

### Visual censorship

```bash
python milestone1_video_censorship.py "input_video.mp4"
```

Output:

```text
censored_video.mp4
```

### Audio censorship

```bash
python milestone2_audio_censorship.py "input_video.mp4"
```

Outputs:

```text
temp_audio.wav
censored_audio.wav
```

### Final media merge

Run this only after the visual and audio output files have been created:

```bash
python milestone3_multimedia.py
```

Output:

```text
final_output.mp4
```

---

## 7. Building the Windows Executable

Run the build script from the working Python environment:

```bash
python Build_exe.py
```

The script performs the following actions:

1. Checks whether PyInstaller is installed.
2. Verifies the main runtime packages.
3. Verifies the scripts, models, dictionary, and FFmpeg files.
4. Removes the previous packaged output.
5. Builds `VideoCensorshipSystem.exe` using PyInstaller `--onedir`.
6. Copies the processing scripts, models, dictionary, and FFmpeg executables.
7. Creates `python_runtime.txt` containing the Python interpreter path used during the build.
8. Displays a checklist of the generated files.

---

## 8. Folder Structure After Building the Executable

After `Build_exe.py` completes, the main output is:

```text
FYP CODE/
├── build/
│   └── VideoCensorshipSystem/
│       ├── Analysis-00.toc
│       ├── base_library.zip
│       ├── COLLECT-00.toc
│       ├── EXE-00.toc
│       ├── PKG-00.toc
│       ├── PYZ-00.pyz
│       ├── warn-VideoCensorshipSystem.txt
│       └── other temporary PyInstaller files
│
└── dist/
    └── VideoCensorshipSystem/
        ├── VideoCensorshipSystem.exe
        ├── python_runtime.txt
        ├── bad_words.txt
        ├── milestone1_video_censorship.py
        ├── milestone2_audio_censorship.py
        ├── milestone3_multimedia.py
        │
        ├── Models/
        │   ├── Visual/
        │   │   └── best.pt
        │   └── Audio/
        │       └── best_model.pt
        │
        ├── ffmpeg/
        │   └── bin/
        │       ├── ffmpeg.exe
        │       ├── ffplay.exe
        │       └── ffprobe.exe
        │
        └── _internal/
            └── PyInstaller runtime files
```

### Important difference between `build` and `dist`

- `build/` contains temporary PyInstaller analysis and compilation files.
- `dist/VideoCensorshipSystem/` is the actual application folder that should be run, copied, zipped, demonstrated, or submitted.

Do not distribute the `build` folder.

---

## 9. Running the Built Application

Open:

```text
dist\VideoCensorshipSystem\
```

Then run:

```text
VideoCensorshipSystem.exe
```

Keep the complete `VideoCensorshipSystem` folder together. Do not move only the `.exe`, because the program requires the scripts, models, FFmpeg files, dictionary, and `_internal` folder.

---

## 10. Important Deployment Limitation

The current executable packages the graphical interface using PyInstaller, but the visual, audio, and merge scripts are launched through the Python interpreter recorded in:

```text
python_runtime.txt
```

Therefore, the current version is **not fully standalone**.

The target computer must still have:

- the recorded Python interpreter;
- the required Python libraries;
- a compatible operating environment.

Do not delete or move the Python environment used to build the application unless the executable is rebuilt afterwards.

For a demonstration on the same development laptop, this setup is suitable. For deployment to another computer, the same Python environment and required libraries must be installed, or the application must be repackaged into a fully standalone build.

---

## 11. Main Generated Files

During processing, the following files are generated in the application folder:

```text
temp_audio.wav
censored_video.mp4
censored_audio.wav
final_output.mp4
```

| Output file | Description |
|---|---|
| `temp_audio.wav` | Audio extracted from the selected MP4 video |
| `censored_video.mp4` | Video containing blurred NSFW frames |
| `censored_audio.wav` | Audio containing muted profanity intervals |
| `final_output.mp4` | Final merged censored video |

The user downloads a copy of `final_output.mp4` through the GUI.

---

## 12. Current Processing Rules

### Visual censorship

- Each frame is classified using YOLOv8s-cls.
- NSFW probability threshold: `0.80`.
- Frames above the threshold are blurred.
- The complete frame is blurred because the model performs image classification rather than bounding-box detection.

### Audio censorship

- Window size: `1.0 second`.
- Step size: `0.1 second`.
- AudioCNN profanity threshold: `0.50`.
- Whisper model: `base`.
- Whisper language: English.
- Whisper uses word timestamps and matches recognised words against `bad_words.txt`.

### AudioCNN and Whisper decisions

| AudioCNN | Whisper | Action |
|---|---|---|
| Positive | Positive | Mute |
| Negative | Positive | Mute |
| Positive | Negative | Reject AudioCNN candidate |
| Negative | Negative | Preserve audio |

---

## 13. Troubleshooting

### Python interpreter cannot be found

Check that `python_runtime.txt` contains a valid Python executable path. Rebuild the application using the working Python environment.

### Missing required package

Activate the environment used for the project and install the missing package:

```bash
pip install <package-name>
```

Then rebuild the application if necessary.

### FFmpeg not found

Confirm that this file exists:

```text
ffmpeg\bin\ffmpeg.exe
```

Also keep `ffprobe.exe` and `ffplay.exe` in the same folder.

### Visual model not found

Confirm that this file exists:

```text
Models\Visual\best.pt
```

### AudioCNN model not found

Confirm that this file exists:

```text
Models\Audio\best_model.pt
```

### Bad-word dictionary not found

Confirm that this file exists:

```text
bad_words.txt
```

### Unsupported file format

The current prototype accepts only `.mp4` videos.

### Processing is slow

Processing speed depends on video duration, resolution, frame rate, GPU availability, and Whisper transcription time. A CUDA-compatible GPU is recommended.

### The application reports a stage failure

Run the individual script in Command Prompt to view its complete error:

```bash
python milestone1_video_censorship.py "input_video.mp4"
python milestone2_audio_censorship.py "input_video.mp4"
python milestone3_multimedia.py
```

---

## 14. Notes

- The system is designed for offline MP4 processing.
- It does not perform real-time livestream censorship.
- It does not analyse subtitles, captions, or on-screen text.
- Whisper and the prohibited-word dictionary are mainly configured for English speech.
- The application processes one selected video at a time.
- The final visual and audio branches are integrated through separate censorship decisions and media-output merging, not product-rule late fusion.