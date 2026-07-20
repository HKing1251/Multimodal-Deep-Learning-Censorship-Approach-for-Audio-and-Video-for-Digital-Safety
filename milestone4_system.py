"""
Milestone 4 GUI — Multimodal Video Censorship System
FYP: Chong Chun Kit (TP077436)

Features:
- Insert Video button + optional drag-and-drop
- Start Censorship button
- Download Video button
- Elapsed-time timer + ETA display
- True visual-stage progress from "Processed X/Y"
- Indeterminate animation during Whisper transcription
- Full system log

Expected files:
    milestone1_video_censorship.py
    milestone2_audio_censorship.py
    milestone3_multimedia.py
    bad_words.txt
    evaluation/audio/best_model.pt
    runs/classify/yolov8_nsfw_classifier_v2/weights/best.pt

Optional drag-and-drop:
    pip install tkinterdnd2
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# =====================================================
# OPTIONAL DRAG AND DROP
# =====================================================

DND_AVAILABLE = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except ImportError:
    DND_FILES = None
    TkinterDnD = None


# =====================================================
# CONFIGURATION
# =====================================================

APP_TITLE = "Multimodal Video Censorship System"

VISUAL_SCRIPT = "milestone1_video_censorship.py"
AUDIO_SCRIPT = "milestone2_audio_censorship.py"
MERGE_SCRIPT = "milestone3_multimedia.py"

FINAL_OUTPUT_NAME = "final_output.mp4"

VIDEO_FILE_TYPES = [
    ("MP4 Video", "*.mp4"),
    ("All Files", "*.*"),
]

VISUAL_START = 0.0
VISUAL_END = 45.0

AUDIO_START = 45.0
AUDIO_END = 90.0

MERGE_START = 90.0
MERGE_END = 100.0

BaseWindow = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk


# =====================================================
# PATH AND PYTHON HELPERS
# =====================================================

def get_app_directory() -> Path:
    """
    Return the folder containing the EXE when frozen,
    or the folder containing this source file when run normally.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def get_python_command(base_dir: Path) -> list[str]:
    """
    Return the Python interpreter that should run milestone 1, 2 and 3.

    Priority:
    1. python_runtime.txt written by build_exe.py.
    2. sys.executable when the GUI is run normally.
    3. python.exe found in PATH.
    4. Windows py launcher.
    """
    runtime_file = base_dir / "python_runtime.txt"

    if runtime_file.exists():
        recorded_path = runtime_file.read_text(
            encoding="utf-8"
        ).strip()

        if recorded_path:
            python_path = Path(recorded_path)

            if python_path.exists():
                return [str(python_path)]

    if not getattr(sys, "frozen", False):
        return [sys.executable]

    python_exe = shutil.which("python")

    if python_exe:
        return [python_exe]

    py_launcher = shutil.which("py")

    if py_launcher:
        return [py_launcher, "-3"]

    raise RuntimeError(
        "A suitable Python interpreter could not be found.\n\n"
        "Rebuild the application from the Python environment that "
        "contains OpenCV, PyTorch, Ultralytics, Whisper and the other "
        "required packages."
    )


def configure_ffmpeg(base_dir: Path) -> None:
    """
    Add the local FFmpeg folder to PATH when it exists.
    """
    ffmpeg_dir = base_dir / "ffmpeg" / "bin"

    if ffmpeg_dir.exists():
        os.environ["PATH"] = (
            str(ffmpeg_dir)
            + os.pathsep
            + os.environ.get("PATH", "")
        )


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"

    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


# =====================================================
# APPLICATION
# =====================================================

class CensorshipApp(BaseWindow):
    def __init__(self) -> None:
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("980x790")
        self.minsize(900, 700)

        self.project_dir = get_app_directory()
        configure_ffmpeg(self.project_dir)

        try:
            self.python_command = get_python_command(
                self.project_dir
            )
        except Exception as exc:
            messagebox.showerror(
                "Python Environment Error",
                str(exc),
            )
            self.destroy()
            return

        self.selected_video: Path | None = None
        self.final_output = (
            self.project_dir / FINAL_OUTPUT_NAME
        )

        self.processing = False
        self.current_stage = "idle"

        self.start_time: float | None = None
        self.eta_seconds: float | None = None

        self.ui_queue: queue.Queue[
            tuple[str, str | None]
        ] = queue.Queue()

        self.recent_log_lines: deque[str] = deque(
            maxlen=40
        )

        self._build_ui()
        self._setup_drag_and_drop()

        self.after(100, self._process_ui_queue)
        self.after(500, self._update_timer)

    # =================================================
    # UI
    # =================================================

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=22)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Multimodal Video Censorship System",
            font=("Segoe UI", 22, "bold"),
        ).pack(pady=(0, 5))

        ttk.Label(
            main,
            text=(
                "Insert an MP4 video. The system will blur detected "
                "NSFW visual content and mute detected spoken profanity."
            ),
            font=("Segoe UI", 10),
            wraplength=850,
            justify="center",
        ).pack(pady=(0, 18))

        input_frame = ttk.LabelFrame(
            main,
            text="1. Insert Video",
            padding=16,
        )
        input_frame.pack(fill="x", pady=(0, 12))

        drop_text = (
            "Drag and drop an MP4 video here\n"
            "or click the Insert Video button"
            if DND_AVAILABLE
            else
            "Click Insert Video to choose an MP4 file\n"
            "Optional drag-and-drop requires tkinterdnd2"
        )

        self.drop_label = ttk.Label(
            input_frame,
            text=drop_text,
            anchor="center",
            justify="center",
            font=("Segoe UI", 11),
        )
        self.drop_label.pack(
            fill="x",
            pady=(2, 8),
        )

        self.video_path_var = tk.StringVar(
            value="No video selected"
        )

        ttk.Label(
            input_frame,
            textvariable=self.video_path_var,
            wraplength=850,
            anchor="center",
            justify="center",
        ).pack(fill="x", pady=(0, 10))

        self.insert_button = ttk.Button(
            input_frame,
            text="Insert Video",
            command=self.select_video,
        )
        self.insert_button.pack(
            ipadx=24,
            ipady=6,
        )

        process_frame = ttk.LabelFrame(
            main,
            text="2. Process Video",
            padding=16,
        )
        process_frame.pack(fill="x", pady=(0, 12))

        self.status_var = tk.StringVar(
            value="Status: Ready"
        )

        ttk.Label(
            process_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.progress = ttk.Progressbar(
            process_frame,
            mode="determinate",
            maximum=100,
            value=0,
        )
        self.progress.pack(fill="x", pady=(0, 8))

        timing_row = ttk.Frame(process_frame)
        timing_row.pack(fill="x", pady=(0, 10))

        self.progress_text_var = tk.StringVar(
            value="Progress: 0.0%"
        )
        self.elapsed_var = tk.StringVar(
            value="Elapsed: 00:00"
        )
        self.eta_var = tk.StringVar(
            value="ETA: --:--"
        )

        ttk.Label(
            timing_row,
            textvariable=self.progress_text_var,
        ).pack(side="left")

        ttk.Label(
            timing_row,
            textvariable=self.elapsed_var,
        ).pack(side="left", padx=(25, 0))

        ttk.Label(
            timing_row,
            textvariable=self.eta_var,
        ).pack(side="right")

        self.process_button = ttk.Button(
            process_frame,
            text="Start Censorship",
            command=self.start_processing,
            state="disabled",
        )
        self.process_button.pack(
            ipadx=24,
            ipady=6,
        )

        output_frame = ttk.LabelFrame(
            main,
            text="3. Download Video",
            padding=16,
        )
        output_frame.pack(fill="x", pady=(0, 12))

        self.output_var = tk.StringVar(
            value="No censored video available yet"
        )

        ttk.Label(
            output_frame,
            textvariable=self.output_var,
            wraplength=850,
            anchor="center",
            justify="center",
        ).pack(fill="x", pady=(0, 10))

        self.download_button = ttk.Button(
            output_frame,
            text="Download Video",
            command=self.download_video,
            state="disabled",
        )
        self.download_button.pack(
            ipadx=24,
            ipady=6,
        )

        log_frame = ttk.LabelFrame(
            main,
            text="System Log",
            padding=8,
        )
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=14,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )

        scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
        )

        self.log_text.configure(
            yscrollcommand=scrollbar.set
        )

        self.log_text.pack(
            side="left",
            fill="both",
            expand=True,
        )
        scrollbar.pack(
            side="right",
            fill="y",
        )

        self._append_log(
            f"Project folder: {self.project_dir}\n"
        )
        self._append_log(
            "Python command: "
            + " ".join(self.python_command)
            + "\n"
        )
        self._append_log(
            "Drag-and-drop support: "
            + (
                "Available\n"
                if DND_AVAILABLE
                else "Not installed\n"
            )
        )

    # =================================================
    # DRAG AND DROP
    # =================================================

    def _setup_drag_and_drop(self) -> None:
        if not DND_AVAILABLE:
            return

        self.drop_label.drop_target_register(
            DND_FILES
        )
        self.drop_label.dnd_bind(
            "<<Drop>>",
            self._on_drop,
        )

    def _on_drop(self, event) -> None:
        if self.processing:
            return

        paths = self.tk.splitlist(event.data)

        if paths:
            self._set_selected_video(
                Path(paths[0])
            )

    # =================================================
    # FILE SELECTION
    # =================================================

    def select_video(self) -> None:
        if self.processing:
            return

        selected = filedialog.askopenfilename(
            title="Select MP4 Video",
            filetypes=VIDEO_FILE_TYPES,
        )

        if selected:
            self._set_selected_video(
                Path(selected)
            )

    def _set_selected_video(
        self,
        video_path: Path,
    ) -> None:
        try:
            video_path = (
                video_path.expanduser().resolve()
            )
        except Exception:
            messagebox.showerror(
                "Invalid File",
                "The selected file path is invalid.",
            )
            return

        if (
            not video_path.exists()
            or not video_path.is_file()
        ):
            messagebox.showerror(
                "Invalid File",
                "The selected video does not exist.",
            )
            return

        if video_path.suffix.lower() != ".mp4":
            messagebox.showwarning(
                "Unsupported Video Format",
                "The current prototype supports MP4 videos only.",
            )
            return

        self.selected_video = video_path
        self.video_path_var.set(str(video_path))
        self.status_var.set(
            "Status: Video selected and ready"
        )
        self._set_progress(0)
        self.output_var.set(
            "No censored video available yet"
        )
        self.download_button.configure(
            state="disabled"
        )
        self.process_button.configure(
            state="normal"
        )

        self._append_log(
            f"\nSelected video:\n{video_path}\n"
        )

    # =================================================
    # VALIDATION
    # =================================================

    def _validate_project_files(
        self,
    ) -> dict[str, Path]:
        required = {
            "visual": (
                self.project_dir
                / VISUAL_SCRIPT
            ),
            "audio": (
                self.project_dir
                / AUDIO_SCRIPT
            ),
            "merge": (
                self.project_dir
                / MERGE_SCRIPT
            ),
            "bad_words": (
                self.project_dir
                / "bad_words.txt"
            ),
            "visual_model": (
                self.project_dir
                / "Models"
                / "Visual"
                / "best.pt"
            ),
            "audio_model": (
                self.project_dir
                / "Models"
                / "Audio"
                / "best_model.pt"
            ),
        }

        missing = [
            path
            for path in required.values()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required file(s):\n\n"
                + "\n".join(
                    str(path)
                    for path in missing
                )
            )

        ffmpeg_exe = (
            self.project_dir
            / "ffmpeg"
            / "bin"
            / "ffmpeg.exe"
        )

        if not ffmpeg_exe.exists():
            raise FileNotFoundError(
                "FFmpeg was not found:\n\n"
                f"{ffmpeg_exe}"
            )

        return required

    # =================================================
    # PROCESSING
    # =================================================

    def start_processing(self) -> None:
        if self.processing:
            return

        if self.selected_video is None:
            messagebox.showwarning(
                "No Video Selected",
                "Please insert a video first.",
            )
            return

        try:
            scripts = self._validate_project_files()
        except Exception as exc:
            messagebox.showerror(
                "Project File Error",
                str(exc),
            )
            return

        for old_output in self._possible_output_paths():
            try:
                if old_output.exists():
                    old_output.unlink()
            except OSError:
                pass

        self.processing = True
        self.current_stage = "starting"
        self.start_time = time.monotonic()
        self.eta_seconds = None
        self.recent_log_lines.clear()

        self.insert_button.configure(
            state="disabled"
        )
        self.process_button.configure(
            state="disabled"
        )
        self.download_button.configure(
            state="disabled"
        )

        self._set_progress(0)
        self.status_var.set(
            "Status: Starting censorship pipeline..."
        )
        self.output_var.set(
            "Processing video..."
        )

        self._append_log(
            "\n"
            + "=" * 68
            + "\nSTARTING MULTIMODAL CENSORSHIP PIPELINE\n"
            + "=" * 68
            + "\n"
        )

        threading.Thread(
            target=self._processing_worker,
            args=(scripts,),
            daemon=True,
        ).start()

    def _processing_worker(
        self,
        scripts: dict[str, Path],
    ) -> None:
        assert self.selected_video is not None

        try:
            self.current_stage = "visual"
            self.ui_queue.put(
                (
                    "status",
                    "Status: Processing visual content with YOLOv8...",
                )
            )
            self.ui_queue.put(
                ("progress", str(VISUAL_START))
            )
            self.ui_queue.put(
                ("log", "\n[1/3] VISUAL CENSORSHIP\n")
            )

            self._run_subprocess(
                [
                    *self.python_command,
                    str(scripts["visual"]),
                    str(self.selected_video),
                ],
                stage="visual",
            )

            self.ui_queue.put(
                ("progress", str(VISUAL_END))
            )

            self.current_stage = "audio"
            self.ui_queue.put(
                (
                    "status",
                    "Status: Analysing audio with AudioCNN + Whisper...",
                )
            )
            self.ui_queue.put(
                ("progress", str(AUDIO_START))
            )
            self.ui_queue.put(
                ("log", "\n[2/3] AUDIO CENSORSHIP\n")
            )

            self._run_subprocess(
                [
                    *self.python_command,
                    str(scripts["audio"]),
                    str(self.selected_video),
                ],
                stage="audio",
            )

            self.ui_queue.put(
                ("progress", str(AUDIO_END))
            )

            self.current_stage = "merge"
            self.ui_queue.put(
                (
                    "status",
                    "Status: Merging censored video and audio...",
                )
            )
            self.ui_queue.put(
                ("progress", str(MERGE_START))
            )
            self.ui_queue.put(
                ("log", "\n[3/3] MERGING FINAL OUTPUT\n")
            )

            self._run_subprocess(
                [
                    *self.python_command,
                    str(scripts["merge"]),
                ],
                stage="merge",
            )

            self.ui_queue.put(
                ("progress", str(MERGE_END))
            )

            found_output = self._find_final_output()

            if found_output is None:
                raise FileNotFoundError(
                    "Pipeline finished but final_output.mp4 "
                    "was not created."
                )

            self.final_output = found_output
            self.ui_queue.put(
                ("success", str(found_output))
            )

        except Exception as exc:
            self.ui_queue.put(
                ("error", str(exc))
            )

    def _possible_output_paths(self) -> list[Path]:
        paths = [
            self.project_dir / FINAL_OUTPUT_NAME,
        ]

        if self.selected_video is not None:
            paths.append(
                self.selected_video.parent
                / FINAL_OUTPUT_NAME
            )

        return paths

    def _find_final_output(self) -> Path | None:
        for path in self._possible_output_paths():
            if path.exists() and path.is_file():
                return path.resolve()

        return None

    # =================================================
    # SUBPROCESS HANDLING
    # =================================================

    def _run_subprocess(
        self,
        command: list[str],
        stage: str,
    ) -> None:
        creation_flags = 0

        if os.name == "nt":
            creation_flags = (
                subprocess.CREATE_NO_WINDOW
            )

        self.ui_queue.put(
            (
                "log",
                "\nCommand: "
                + " ".join(command)
                + "\n",
            )
        )

        process = subprocess.Popen(
            command,
            cwd=str(self.project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
            env=os.environ.copy(),
        )

        if process.stdout is None:
            raise RuntimeError(
                "Unable to capture subprocess output."
            )

        for line in process.stdout:
            self.ui_queue.put(
                ("log", line)
            )
            self.recent_log_lines.append(
                line.rstrip()
            )

            if stage == "visual":
                self._parse_visual_progress(line)
            elif stage == "audio":
                self._parse_audio_status(line)
            elif stage == "merge":
                self._parse_merge_status(line)

        return_code = process.wait()

        if return_code != 0:
            diagnostic = "\n".join(
                self.recent_log_lines
            )

            raise RuntimeError(
                "A processing script failed during the "
                f"{stage.upper()} stage.\n\n"
                f"Exit code: {return_code}\n"
                f"Command: {' '.join(command)}\n\n"
                f"Last output lines:\n{diagnostic}"
            )

    def _parse_visual_progress(
        self,
        line: str,
    ) -> None:
        match = re.search(
            r"Processed\s+(\d+)\s*/\s*(\d+)",
            line,
        )

        if not match:
            return

        current = int(match.group(1))
        total = int(match.group(2))

        if total <= 0:
            return

        stage_fraction = min(
            max(current / total, 0.0),
            1.0,
        )

        overall = (
            VISUAL_START
            + stage_fraction
            * (VISUAL_END - VISUAL_START)
        )

        self.ui_queue.put(
            ("progress", f"{overall:.3f}")
        )

    def _parse_audio_status(
        self,
        line: str,
    ) -> None:
        text = line.lower()

        if (
            "[step 1]" in text
            and "audiocnn" in text
        ):
            self.ui_queue.put(
                (
                    "status",
                    "Status: AudioCNN scanning audio windows...",
                )
            )
            self.ui_queue.put(
                ("progress", "48")
            )

        elif "cnn candidates" in text:
            self.ui_queue.put(
                ("progress", "60")
            )

        elif (
            "[step 2]" in text
            and "whisper" in text
        ):
            self.ui_queue.put(
                (
                    "status",
                    "Status: Whisper transcribing audio...",
                )
            )
            self.ui_queue.put(
                ("progress", "65")
            )
            self.ui_queue.put(
                ("whisper_wait", None)
            )

        elif "whisper bad words" in text:
            self.ui_queue.put(
                ("progress", "80")
            )

        elif (
            "[step 3]" in text
            and "combining" in text
        ):
            self.ui_queue.put(
                (
                    "status",
                    "Status: Combining AudioCNN and Whisper results...",
                )
            )
            self.ui_queue.put(
                ("progress", "83")
            )

        elif "muting segments" in text:
            self.ui_queue.put(
                (
                    "status",
                    "Status: Muting confirmed profanity segments...",
                )
            )
            self.ui_queue.put(
                ("progress", "87")
            )

    def _parse_merge_status(
        self,
        line: str,
    ) -> None:
        text = line.lower()

        if "loading censored video" in text:
            self.ui_queue.put(
                ("progress", "92")
            )
        elif "loading censored audio" in text:
            self.ui_queue.put(
                ("progress", "94")
            )
        elif "merging" in text:
            self.ui_queue.put(
                ("progress", "96")
            )

    # =================================================
    # DOWNLOAD
    # =================================================

    def download_video(self) -> None:
        if not self.final_output.exists():
            messagebox.showerror(
                "Output Missing",
                "The censored video output could not be found.",
            )
            self.download_button.configure(
                state="disabled"
            )
            return

        save_location = filedialog.asksaveasfilename(
            title="Download Censored Video",
            defaultextension=".mp4",
            initialfile="censored_output.mp4",
            filetypes=[
                ("MP4 Video", "*.mp4")
            ],
        )

        if not save_location:
            return

        try:
            destination = Path(
                save_location
            ).resolve()

            source = self.final_output.resolve()

            if source != destination:
                shutil.copy2(
                    source,
                    destination,
                )

            messagebox.showinfo(
                "Download Complete",
                "The censored video was saved successfully.\n\n"
                f"{destination}",
            )

            self._append_log(
                f"\nDownloaded output to:\n{destination}\n"
            )

        except Exception as exc:
            messagebox.showerror(
                "Download Failed",
                "The censored video could not be saved.\n\n"
                f"{exc}",
            )

    # =================================================
    # TIMER
    # =================================================

    def _update_timer(self) -> None:
        if (
            self.processing
            and self.start_time is not None
        ):
            elapsed = (
                time.monotonic()
                - self.start_time
            )

            self.elapsed_var.set(
                f"Elapsed: {format_seconds(elapsed)}"
            )

            progress_value = float(
                self.progress["value"]
            )

            if (
                elapsed >= 8
                and progress_value >= 2
            ):
                estimated_total = (
                    elapsed
                    / (progress_value / 100.0)
                )

                eta = max(
                    estimated_total - elapsed,
                    0.0,
                )

                self.eta_seconds = eta
                self.eta_var.set(
                    f"ETA: ~{format_seconds(eta)}"
                )
            else:
                self.eta_var.set(
                    "ETA: calculating..."
                )

        self.after(
            500,
            self._update_timer,
        )

    # =================================================
    # QUEUE EVENTS
    # =================================================

    def _process_ui_queue(self) -> None:
        try:
            while True:
                event_type, value = (
                    self.ui_queue.get_nowait()
                )

                if (
                    event_type == "log"
                    and value is not None
                ):
                    self._append_log(value)

                elif (
                    event_type == "status"
                    and value is not None
                ):
                    self.status_var.set(value)

                elif (
                    event_type == "progress"
                    and value is not None
                ):
                    self._set_progress(
                        float(value)
                    )

                elif event_type == "whisper_wait":
                    self.eta_var.set(
                        "ETA: Whisper-dependent"
                    )

                elif event_type == "success":
                    self._handle_success(value)

                elif event_type == "error":
                    self._handle_error(value)

        except queue.Empty:
            pass

        self.after(
            100,
            self._process_ui_queue,
        )

    def _set_progress(
        self,
        value: float,
    ) -> None:
        value = max(
            0.0,
            min(100.0, value),
        )

        self.progress["value"] = value
        self.progress_text_var.set(
            f"Progress: {value:.1f}%"
        )

    # =================================================
    # SUCCESS AND ERROR
    # =================================================

    def _handle_success(
        self,
        output_path: str | None,
    ) -> None:
        self.processing = False
        self.current_stage = "done"
        self._set_progress(100)
        self.status_var.set(
            "Status: Processing complete"
        )
        self.eta_var.set(
            "ETA: Complete"
        )

        if output_path is None:
            output_path = str(
                self.final_output
            )

        self.output_var.set(
            f"Censored video ready:\n{output_path}"
        )

        self.insert_button.configure(
            state="normal"
        )
        self.process_button.configure(
            state="normal"
        )
        self.download_button.configure(
            state="normal"
        )

        self._append_log(
            "\n"
            + "=" * 68
            + "\nPIPELINE COMPLETE\n"
            + f"Output: {output_path}\n"
            + "=" * 68
            + "\n"
        )

        messagebox.showinfo(
            "Processing Complete",
            "The video has been censored successfully.\n\n"
            "Click Download Video to choose where to save it.",
        )

    def _handle_error(
        self,
        error_message: str | None,
    ) -> None:
        self.processing = False
        self.current_stage = "failed"
        self.status_var.set(
            "Status: Processing failed"
        )
        self.eta_var.set(
            "ETA: Failed"
        )
        self.output_var.set(
            "No censored video available"
        )

        self.insert_button.configure(
            state="normal"
        )
        self.process_button.configure(
            state=(
                "normal"
                if self.selected_video is not None
                else "disabled"
            )
        )
        self.download_button.configure(
            state="disabled"
        )

        if error_message is None:
            error_message = "Unknown error"

        self._append_log(
            f"\nERROR:\n{error_message}\n"
        )

        messagebox.showerror(
            "Processing Failed",
            error_message,
        )

    # =================================================
    # LOG
    # =================================================

    def _append_log(
        self,
        text: str,
    ) -> None:
        self.log_text.configure(
            state="normal"
        )
        self.log_text.insert(
            "end",
            text,
        )
        self.log_text.see("end")
        self.log_text.configure(
            state="disabled"
        )


def main() -> None:
    app = CensorshipApp()
    app.mainloop()


if __name__ == "__main__":
    main()