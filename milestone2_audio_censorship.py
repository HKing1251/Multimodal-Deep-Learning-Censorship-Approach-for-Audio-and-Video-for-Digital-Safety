"""
Milestone 2
Audio Profanity Censorship using Hybrid AudioCNN + Whisper

Input:
    video path passed from milestone4_system.py (or run directly)

Output:
    censored_audio.wav
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

BASE_DIR = Path(__file__).resolve().parent
FFMPEG_BIN = BASE_DIR / "ffmpeg" / "bin"

if not FFMPEG_BIN.exists():
    raise FileNotFoundError(f"FFmpeg folder not found: {FFMPEG_BIN}")

os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ["PATH"]


class AudioCNN(nn.Module):
    """CNN architecture used for clean/profanity classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.Dropout2d(0.10), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.Dropout2d(0.15), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.Dropout2d(0.20), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


class AudioCensor:
    """Coordinates AudioCNN scanning, Whisper transcription, and muting."""

    def __init__(
        self,
        model_path: str = "Models/Audio/best_model.pt",
        bad_words_path: str = "bad_words.txt",
        output_audio: str = "censored_audio.wav",
        temp_audio: str = "temp_audio.wav",
        window_size: float = 1.0,
        step_size: float = 0.1,
        cnn_threshold: float = 0.50,
        whisper_model_name: str = "base",
        target_sr: int = 16000,
        n_mels: int = 64,
        n_frames: int = 101,
        n_fft: int = 512,
        hop_length: int = 160,
        fmin: int = 20,
        fmax: int = 8000,
    ) -> None:
        self.model_path = Path(model_path)
        self.bad_words_path = Path(bad_words_path)
        self.output_audio = Path(output_audio)
        self.temp_audio = Path(temp_audio)
        self.window_size = window_size
        self.step_size = step_size
        self.cnn_threshold = cnn_threshold
        self.whisper_model_name = whisper_model_name
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_samples = int(self.target_sr * self.window_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bad_words: set[str] = set()
        self.cnn_model: AudioCNN | None = None
        self.whisper_model = None

    def load_bad_words(self) -> None:
        if not self.bad_words_path.exists():
            raise FileNotFoundError(f"Bad words file not found: {self.bad_words_path}")
        with self.bad_words_path.open("r", encoding="utf8") as file:
            self.bad_words = {line.strip().lower() for line in file if line.strip()}

    def load_models(self) -> None:
        print("Loading AudioCNN...", flush=True)
        if not self.model_path.exists():
            raise FileNotFoundError(f"AudioCNN model not found: {self.model_path}")

        self.cnn_model = AudioCNN().to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_model.eval()
        print("AudioCNN loaded", flush=True)

        print(f"Loading Whisper ({self.whisper_model_name})...", flush=True)
        self.whisper_model = whisper.load_model(self.whisper_model_name)
        print("Whisper loaded", flush=True)

    @staticmethod
    def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
        return not (a_end <= b_start or a_start >= b_end)

    def audio_to_logmel(self, waveform: np.ndarray) -> np.ndarray:
        if len(waveform) < self.clip_samples:
            waveform = np.pad(waveform, (0, self.clip_samples - len(waveform)))
        else:
            waveform = waveform[:self.clip_samples]

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        if log_mel.shape[1] < self.n_frames:
            log_mel = np.pad(log_mel, ((0, 0), (0, self.n_frames - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :self.n_frames]

        return log_mel.astype(np.float32)

    def extract_audio(self, video_path: str) -> Tuple[AudioSegment, np.ndarray, float]:
        print("\nExtracting audio...", flush=True)
        video = VideoFileClip(video_path)
        try:
            if video.audio is None:
                raise RuntimeError("The selected video does not contain an audio track.")
            video.audio.write_audiofile(str(self.temp_audio), logger=None)
        finally:
            video.close()

        audio = AudioSegment.from_wav(str(self.temp_audio))
        waveform, _ = librosa.load(str(self.temp_audio), sr=self.target_sr)
        duration = len(waveform) / self.target_sr
        print(f"Duration : {duration:.2f} sec", flush=True)
        return audio, waveform, duration

    def scan_audio_windows(
        self,
        waveform: np.ndarray,
        duration: float,
    ) -> List[Tuple[float, float, float]]:
        if self.cnn_model is None:
            raise RuntimeError("AudioCNN has not been loaded.")

        print("\n[Step 1] AudioCNN scanning...", flush=True)
        candidates: List[Tuple[float, float, float]] = []
        current_time = 0.0

        while current_time + self.window_size <= duration:
            start = int(current_time * self.target_sr)
            end = int((current_time + self.window_size) * self.target_sr)
            chunk = waveform[start:end]

            spec = torch.tensor(self.audio_to_logmel(chunk)).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = float(torch.softmax(self.cnn_model(spec), dim=1)[0][1])

            if score >= self.cnn_threshold:
                candidates.append((current_time, current_time + self.window_size, score))
                print(
                    f"  CNN flagged  {current_time:.2f}s-"
                    f"{current_time + self.window_size:.2f}s  (score={score:.3f})",
                    flush=True,
                )

            current_time += self.step_size

        print(f"  CNN candidates : {len(candidates)}", flush=True)
        return candidates

    def transcribe_profanity(self) -> List[Tuple[float, float, str]]:
        if self.whisper_model is None:
            raise RuntimeError("Whisper has not been loaded.")

        print("\n[Step 2] Whisper transcribing...", flush=True)
        result = self.whisper_model.transcribe(
            str(self.temp_audio),
            word_timestamps=True,
            language="en",
        )

        detected: List[Tuple[float, float, str]] = []
        for segment in result.get("segments", []):
            for word_data in segment.get("words", []):
                word = word_data["word"].strip().lower().strip(".,!?\"'()[]")
                if word in self.bad_words:
                    detected.append((float(word_data["start"]), float(word_data["end"]), word))
                    print(
                        f"  Whisper bad  {word_data['start']:.2f}s-"
                        f"{word_data['end']:.2f}s  word='{word}'",
                        flush=True,
                    )

        print(f"  Whisper bad words : {len(detected)}", flush=True)
        return detected

    def combine_results(
        self,
        cnn_candidates: Sequence[Tuple[float, float, float]],
        whisper_bad: Sequence[Tuple[float, float, str]],
    ) -> List[List[float]]:
        print("\n[Step 3] Combining results...", flush=True)
        mute_segments: List[Tuple[float, float]] = []

        for c_start, c_end, score in cnn_candidates:
            confirmed = any(
                self.overlaps(c_start, c_end, w_start, w_end)
                for w_start, w_end, _ in whisper_bad
            )
            if confirmed:
                mute_segments.append((c_start, c_end))
                print(
                    f"  CONFIRMED  {c_start:.2f}s-{c_end:.2f}s  (CNN + Whisper)",
                    flush=True,
                )
            else:
                print(
                    f"  REJECTED   {c_start:.2f}s-{c_end:.2f}s  (CNN only, no Whisper match)",
                    flush=True,
                )

        for w_start, w_end, word in whisper_bad:
            already_covered = any(
                self.overlaps(w_start, w_end, m_start, m_end)
                for m_start, m_end in mute_segments
            )
            if not already_covered:
                mute_segments.append((w_start, w_end))
                print(
                    f"  ADDED      {w_start:.2f}s-{w_end:.2f}s  "
                    f"word='{word}'  (Whisper only)",
                    flush=True,
                )

        return self.merge_overlapping_segments(mute_segments)

    @staticmethod
    def merge_overlapping_segments(
        segments: Sequence[Tuple[float, float]],
    ) -> List[List[float]]:
        sorted_segments = sorted(segments, key=lambda item: item[0])
        merged: List[List[float]] = []

        for start, end in sorted_segments:
            if not merged:
                merged.append([start, end])
            elif start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return merged

    def mute_segments(
        self,
        audio: AudioSegment,
        segments: Sequence[Sequence[float]],
    ) -> AudioSegment:
        print("\nMuting segments...", flush=True)
        censored_audio = audio

        for start, end in segments:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            censored_audio = (
                censored_audio[:start_ms]
                + AudioSegment.silent(duration=end_ms - start_ms)
                + censored_audio[end_ms:]
            )

        return censored_audio

    def process_video(self, video_path: str) -> str:
        print("=" * 50)
        print("MILESTONE 2 — AUDIO CENSORSHIP")
        print("=" * 50)
        print(f"Input  : {video_path}")
        print(f"Output : {self.output_audio}")
        print()

        self.load_bad_words()
        self.load_models()
        audio, waveform, duration = self.extract_audio(video_path)
        cnn_candidates = self.scan_audio_windows(waveform, duration)
        whisper_bad = self.transcribe_profanity()
        merged_segments = self.combine_results(cnn_candidates, whisper_bad)
        censored_audio = self.mute_segments(audio, merged_segments)
        censored_audio.export(str(self.output_audio), format="wav")

        print("\nDONE")
        print("=" * 50)
        print(f"Profanity Segments : {len(merged_segments)}")
        for index, (start, end) in enumerate(merged_segments, start=1):
            print(f"  {index}. {start:.2f}s -> {end:.2f}s")
        print(f"Output : {self.output_audio}")

        return str(self.output_audio)


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_video_1.mp4"
    censor = AudioCensor()
    censor.process_video(video_path)


if __name__ == "__main__":
    main()