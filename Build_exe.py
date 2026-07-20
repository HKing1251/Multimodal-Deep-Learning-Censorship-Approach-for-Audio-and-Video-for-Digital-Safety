"""
build_exe.py
============
Run this script ONCE to build VideoCensorshipSystem.exe

Usage:
    python build_exe.py

What it does:
    1. Installs PyInstaller if not already installed
    2. Builds the .exe from milestone4_system.py
    3. Copies all required runtime files into dist/VideoCensorshipSystem/
    4. Prints a checklist of what to verify before running the .exe

Required folder structure (must already exist before running):
    CODE/
    ├── milestone1_video_censorship.py
    ├── milestone2_audio_censorship.py
    ├── milestone3_multimedia.py
    ├── milestone4_system.py        <- the GUI (entry point)
    ├── bad_words.txt
    ├── evaluation/
    │   └── audio/
    │       └── best_model.pt
    └── runs/
        └── classify/
            └── yolov8_nsfw_classifier_v2/
                └── weights/
                    └── best.pt

Output:
    CODE/
    └── dist/
        └── VideoCensorshipSystem/
            ├── VideoCensorshipSystem.exe   <- run this
            ├── milestone1_video_censorship.py
            ├── milestone2_audio_censorship_v2.py
            ├── milestone3_multimedia.py
            ├── bad_words.txt
            ├── evaluation/audio/best_model.pt
            └── runs/classify/yolov8_nsfw_classifier_v2/weights/best.pt
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


GUI_SCRIPT = "milestone4_system.py"
APP_NAME = "VideoCensorshipSystem"

RUNTIME_FILES = [
    (
        "milestone1_video_censorship.py",
        "milestone1_video_censorship.py",
    ),
    (
        "milestone2_audio_censorship.py",
        "milestone2_audio_censorship.py",
    ),
    (
        "milestone3_multimedia.py",
        "milestone3_multimedia.py",
    ),
    (
        "bad_words.txt",
        "bad_words.txt",
    ),
    (
        "Models/Audio/best_model.pt",
        "Models/Audio/best_model.pt",
    ),
    (
        "Models/Visual/best.pt",
        "Models/Visual/best.pt",
    ),
    (
        "ffmpeg/bin/ffmpeg.exe",
        "ffmpeg/bin/ffmpeg.exe",
    ),
    (
        "ffmpeg/bin/ffprobe.exe",
        "ffmpeg/bin/ffprobe.exe",
    ),
    (
        "ffmpeg/bin/ffplay.exe",
        "ffmpeg/bin/ffplay.exe",
    ),
]


def run(
    command: list[str],
    label: str,
) -> None:
    print("\n" + "=" * 68)
    print(f"  {label}")
    print("=" * 68)

    result = subprocess.run(
        command,
        text=True,
    )

    if result.returncode != 0:
        print(
            f"\n[ERROR] {label} failed with "
            f"exit code {result.returncode}"
        )
        sys.exit(result.returncode)


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller

        print(
            "[OK] PyInstaller already installed: "
            f"{PyInstaller.__version__}"
        )
    except ImportError:
        print(
            "[INFO] PyInstaller not found. Installing..."
        )

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pyinstaller",
            ],
            "Installing PyInstaller",
        )


def verify_python_environment() -> None:
    """
    Confirm that the interpreter used for the build can import the most
    important runtime packages.
    """
    print("\n[CHECK] Python runtime environment")
    print(f"        Interpreter: {sys.executable}")

    required_imports = [
        ("cv2", "opencv-python"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("whisper", "openai-whisper"),
        ("librosa", "librosa"),
    ]

    missing = []

    for module_name, package_name in required_imports:
        try:
            __import__(module_name)
            print(f"        [OK] {module_name}")
        except ImportError:
            missing.append(
                (module_name, package_name)
            )
            print(f"        [MISS] {module_name}")

    if missing:
        print(
            "\n[ERROR] The current Python environment is missing "
            "required packages."
        )

        for module_name, package_name in missing:
            print(
                f"        {module_name} "
                f"(install package: {package_name})"
            )

        print(
            "\nActivate the environment where the full pipeline works, "
            "then run build_exe.py again."
        )
        sys.exit(1)


def check_source_files(
    base: Path,
) -> None:
    print(
        "\n[CHECK] Verifying source and runtime files..."
    )

    missing = []

    gui_path = base / GUI_SCRIPT

    if not gui_path.exists():
        missing.append(GUI_SCRIPT)

    for source_relative, _ in RUNTIME_FILES:
        source_path = base / source_relative

        if not source_path.exists():
            missing.append(source_relative)

    if missing:
        print(
            "\n[ERROR] The following files are missing:"
        )

        for item in missing:
            print(f"        {item}")

        sys.exit(1)

    print("        All required files found.")


def remove_previous_output(
    base: Path,
) -> None:
    dist_app = (
        base
        / "dist"
        / APP_NAME
    )

    if dist_app.exists():
        print(
            f"\n[CLEAN] Removing previous output:\n"
            f"        {dist_app}"
        )
        shutil.rmtree(
            dist_app,
            ignore_errors=True,
        )


def build_exe(
    base: Path,
) -> Path:
    dist_dir = (
        base
        / "dist"
        / APP_NAME
    )

    pyinstaller_args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        f"--name={APP_NAME}",
        f"--distpath={base / 'dist'}",
        f"--workpath={base / 'build'}",
        f"--specpath={base}",
        "--hidden-import=tkinter",
        "--hidden-import=tkinter.ttk",
        "--hidden-import=tkinter.filedialog",
        "--hidden-import=tkinter.messagebox",
        "--hidden-import=queue",
        "--hidden-import=threading",
        "--hidden-import=pathlib",
        "--hidden-import=shutil",
        "--hidden-import=re",
        str(base / GUI_SCRIPT),
    ]

    run(
        pyinstaller_args,
        f"Building {APP_NAME}.exe",
    )

    return dist_dir


def copy_runtime_files(
    base: Path,
    dist_dir: Path,
) -> None:
    print("\n" + "=" * 68)
    print("  Copying runtime files")
    print("=" * 68)

    for source_relative, destination_relative in RUNTIME_FILES:
        source = base / source_relative
        destination = (
            dist_dir
            / destination_relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        print(
            f"  [COPIED] {source_relative}\n"
            f"        -> {destination_relative}"
        )

    runtime_file = (
        dist_dir
        / "python_runtime.txt"
    )

    runtime_file.write_text(
        str(Path(sys.executable).resolve()),
        encoding="utf-8",
    )

    print(
        "\n  [CREATED] python_runtime.txt"
    )
    print(
        f"        {Path(sys.executable).resolve()}"
    )


def print_summary(
    dist_dir: Path,
) -> None:
    exe_path = (
        dist_dir
        / f"{APP_NAME}.exe"
    )

    checks = [
        (
            exe_path,
            "Main executable",
        ),
        (
            dist_dir
            / "python_runtime.txt",
            "Recorded Python interpreter",
        ),
        (
            dist_dir
            / "milestone1_video_censorship.py",
            "Visual censorship script",
        ),
        (
            dist_dir
            / "milestone2_audio_censorship.py",
            "Audio censorship script",
        ),
        (
            dist_dir
            / "milestone3_multimedia.py",
            "Media merge script",
        ),
        (
            dist_dir
            / "bad_words.txt",
            "Bad words dictionary",
        ),
        (
            dist_dir
            / "Models"
            / "Audio"
            / "best_model.pt",
            "AudioCNN weights",
        ),
        (
            dist_dir
            / "Models"
            / "Visual"
            / "best.pt",
            "YOLO weights",
        ),
        (
            dist_dir
            / "ffmpeg"
            / "bin"
            / "ffmpeg.exe",
            "FFmpeg",
        ),
        (
            dist_dir
            / "ffmpeg"
            / "bin"
            / "ffprobe.exe",
            "FFprobe",
        ),
    ]

    print("\n" + "=" * 68)
    print("  BUILD COMPLETE")
    print("=" * 68)
    print(f"\nExecutable:\n    {exe_path}")
    print(f"\nOutput folder:\n    {dist_dir}")

    all_ok = True

    print("\nChecklist:")

    for path, label in checks:
        exists = path.exists()

        if not exists:
            all_ok = False

        print(
            f"  {'[OK]' if exists else '[MISS]'} "
            f"{label}"
        )

        try:
            relative = path.relative_to(
                dist_dir
            )
        except ValueError:
            relative = path

        print(f"       {relative}")

    print("\n" + "=" * 68)

    if all_ok:
        print(
            "All files are present.\n\n"
            "The EXE will use this Python interpreter:\n"
            f"    {Path(sys.executable).resolve()}\n\n"
            "Do not move or delete that Python environment unless "
            "you rebuild the application."
        )
    else:
        print(
            "Some files are missing. Do not run the EXE "
            "until the checklist is complete."
        )

    print("=" * 68 + "\n")


def main() -> None:
    base = Path(__file__).resolve().parent

    print(f"Build directory: {base}")

    ensure_pyinstaller()
    verify_python_environment()
    check_source_files(base)
    remove_previous_output(base)

    dist_dir = build_exe(base)

    copy_runtime_files(
        base,
        dist_dir,
    )

    print_summary(dist_dir)


if __name__ == "__main__":
    main()
