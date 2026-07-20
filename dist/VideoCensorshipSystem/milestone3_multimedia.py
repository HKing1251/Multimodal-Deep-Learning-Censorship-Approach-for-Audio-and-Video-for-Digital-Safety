"""
Milestone 3
Combine censored video and censored audio into final output.

Input:
    censored_video.mp4
    censored_audio.wav

Output:
    final_output.mp4
"""

from pathlib import Path

from moviepy.editor import AudioFileClip, VideoFileClip


class MediaMerger:
    def __init__(
        self,
        video_path: str = "censored_video.mp4",
        audio_path: str = "censored_audio.wav",
        output_path: str = "final_output.mp4",
    ) -> None:
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.output_path = Path(output_path)

    def validate_inputs(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Censored video not found: {self.video_path}"
            )

        if not self.audio_path.exists():
            raise FileNotFoundError(
                f"Censored audio not found: {self.audio_path}"
            )

    def merge(self) -> str:
        print("=" * 50)
        print("MILESTONE 3 — MERGE VIDEO + AUDIO")
        print("=" * 50)
        print(f"Video  : {self.video_path}")
        print(f"Audio  : {self.audio_path}")
        print(f"Output : {self.output_path}")
        print()

        self.validate_inputs()

        print("Loading censored video...", flush=True)
        video = VideoFileClip(str(self.video_path))

        print("Loading censored audio...", flush=True)
        audio = AudioFileClip(str(self.audio_path))

        final_video = None

        try:
            print("Merging...", flush=True)

            final_video = video.set_audio(audio)

            final_video.write_videofile(
                str(self.output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )

        finally:
            if final_video is not None:
                final_video.close()

            audio.close()
            video.close()

        print("\nDONE")
        print("=" * 50)
        print(f"Output : {self.output_path}")

        return str(self.output_path)


def main() -> None:
    merger = MediaMerger()
    merger.merge()


if __name__ == "__main__":
    main()