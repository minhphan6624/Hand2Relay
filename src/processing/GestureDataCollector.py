import cv2
import os
from src.common.models import HandLandmarksDetector, label_dict_from_config_file
from src.processing.HandDatasetWriter import HandDatasetWriter


class GestureDataCollector:
    def __init__(self, config_path="config.yaml"):
        labels = label_dict_from_config_file(config_path)
        if not labels:
            raise RuntimeError("No gestures found in config.yaml")
        self.labels = labels
        self.detector = HandLandmarksDetector()

    def collect(self, mode="train"):
        os.makedirs("./src/data", exist_ok=True)
        writer = HandDatasetWriter(
            f"./src/data/landmarks_{mode}.csv")
        cap = cv2.VideoCapture(0)
        current_lbl, recording = None, False
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect the landmarks
            lmks_list, annotated = self.detector.detect_hand(frame)

            # If landmarks are detected and recording is turned on, write to file
            if recording and lmks_list:
                writer.add(lmks_list[0], current_lbl)

            cv2.imshow(
                "Gesture Collector (press a–f to toggle, q to quit)", annotated)

            # Listen for key pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):         # Quit if "q" is pressed
                break

            # Otherwise, if key is between "a" and "f" (alphabetically), label the frame
            if ord("a") <= key <= ord("f"):
                lbl = key - ord("a")
                if lbl in self.labels:
                    # Toggle Recording for this label
                    recording = (lbl == current_lbl) ^ recording
                    current_lbl = lbl

        cap.release()
        cv2.destroyAllWindows()
        writer.close()


if __name__ == "__main__":
    # choose "train" / "val" / "test"
    GestureDataCollector(config_path="config.yaml").collect(mode="train")
