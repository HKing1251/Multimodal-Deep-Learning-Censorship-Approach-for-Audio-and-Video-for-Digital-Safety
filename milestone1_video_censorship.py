"""
Milestone 1
Video NSFW Censorship using YOLOv8 Classification
"""

from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


class VisualCensor:
    def __init__(
        self,
        model_path: str = "Models/Visual/best.pt",
        output_path: str = "censored_video.mp4",
        nsfw_threshold: float = 0.80,
    ) -> None:
        self.model_path = model_path
        self.output_path = output_path
        self.nsfw_threshold = nsfw_threshold
        self.model = None

    def load_model(self) -> None:
        print("Loading YOLOv8 model...")
        self.model = YOLO(self.model_path)
        print(f"Classes : {self.model.names}")

    def classify_frame(self, frame) -> float:
        if self.model is None:
            raise RuntimeError("YOLOv8 model has not been loaded.")

        result = self.model.predict(frame, verbose=False)[0]
        probabilities = result.probs.data.cpu().numpy()

        return float(probabilities[0])

    def blur_frame(self, frame, nsfw_probability: float):
        blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)

        cv2.putText(
            blurred_frame,
            f"NSFW {nsfw_probability:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        return blurred_frame

    def label_safe_frame(self, frame, nsfw_probability: float):
        cv2.putText(
            frame,
            f"SAFE {1 - nsfw_probability:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return frame

    def process_video(self, video_path: str) -> str:
        video_path = str(Path(video_path).resolve())

        print("=" * 50)
        print("MILESTONE 1 — VISUAL CENSORSHIP")
        print("=" * 50)
        print(f"Input  : {video_path}")
        print(f"Output : {self.output_path}")
        print()

        self.load_model()

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            capture.release()
            raise RuntimeError("The input video has an invalid frame rate.")

        print(f"Resolution : {width}x{height}")
        print(f"FPS        : {fps}")
        print(f"Frames     : {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                f"Cannot create output video: {self.output_path}"
            )

        frame_count = 0
        blurred_count = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                frame_count += 1
                nsfw_probability = self.classify_frame(frame)

                if nsfw_probability >= self.nsfw_threshold:
                    frame = self.blur_frame(
                        frame,
                        nsfw_probability,
                    )
                    blurred_count += 1
                else:
                    frame = self.label_safe_frame(
                        frame,
                        nsfw_probability,
                    )

                writer.write(frame)

                if frame_count % 100 == 0:
                    print(
                        f"  Processed {frame_count}/{total_frames}",
                        flush=True,
                    )

        finally:
            capture.release()
            writer.release()

        print("\nDONE")
        print("=" * 50)
        print(f"Frames Processed : {frame_count}")
        print(f"Blurred Frames   : {blurred_count}")
        print(f"Output           : {self.output_path}")

        return self.output_path


def main() -> None:
    video_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "sample_video_1.mp4"
    )

    censor = VisualCensor()
    censor.process_video(video_path)


if __name__ == "__main__":
    main()