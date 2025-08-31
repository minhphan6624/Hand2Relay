import cv2
import os
import argparse
from src.common.models import HandLandmarksDetector, label_dict_from_config_file
from src.processing.data_writer import HandDatasetWriter


class GestureDataCollector:
    def __init__(self, config_path="config.yaml"):
        # Construct the absolute path to config.yaml relative to this script's location
        script_dir = os.path.dirname(__file__)
        abs_config_path = os.path.join(script_dir, "..", "..", config_path)

        labels = label_dict_from_config_file(abs_config_path)
        if not labels:
            raise RuntimeError(f"No gestures found in {abs_config_path}")

        self.labels = labels
        self.detector = HandLandmarksDetector()

    def collect(self):

        # Construct the absolute path for the data directory and CSV file
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, "..", "..", "src", "data")
        os.makedirs(data_dir, exist_ok=True)

        # Construct csv filename
        writer = HandDatasetWriter(
            os.path.join(data_dir, "landmarks_all.csv"))

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

            # Add HUD for recording status and gesture
            gesture_name = self.labels.get(
                current_lbl, "Unknown") if current_lbl is not None else "None"
            status_text = f"Recording: {gesture_name}" if recording else "Idle"
            color = (0, 255, 0) if recording else (
                0, 0, 255)  # Green if recording, Red if idle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            cv2.putText(annotated, status_text, (10, 30), font,
                        font_scale, color, thickness, cv2.LINE_AA)

            cv2.imshow(
                "Gesture Collector (press a–f to toggle, q to quit)", annotated)

            # Listen for key pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):         # Quit if "q" is pressed
                break
            elif ord("a") <= key <= ord("f"):
                lbl = key - ord("a")
                if lbl in self.labels:
                    # Toggle Recording for this label
                    recording = (lbl == current_lbl) ^ recording
                    current_lbl = lbl

        cap.release()
        cv2.destroyAllWindows()
        writer.close()


if __name__ == "__main__":
    GestureDataCollector(config_path="config.yaml").collect()
