import cv2
import os
from pathlib import Path

import yaml

from ..common.landmark_detector import HandLandmarksDetector
from ..common.load_label_dict import load_label_dict

from .data_writer import HandDatasetWriter

class GestureDataCollector:
    def __init__(self, config_path="config.yaml"):
        self.labels = load_label_dict(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            confidence_threshold = yaml.safe_load(f).get("sensor_settings", {}).get("confidence_threshold", 0.7)
        self.detector = HandLandmarksDetector(confidence_threshold=confidence_threshold)

    def collect(self, data_dir="src/data"):
        os.makedirs(data_dir, exist_ok=True)

        csv_path = Path(data_dir) / "landmarks_all.csv" 
        writer = HandDatasetWriter(csv_path)

        # Start video capture
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        current_lbl = None
        is_recording = False
        frame_count = 0

        # Down-sampling
        frame_id = 0
        sample_every = 3 # 10Hz at 30fps

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read from camera")
                break
            
            frame_id += 1
            # Pre-process the frame
            frame = cv2.flip(frame, 1) # Mirror the frame
            lmks_list, annotated_frame = self.detector.detect_hand(frame) # Detect hand landmarks

            # If recording is turned on and landmarks are detected, write to file
            if is_recording and lmks_list and current_lbl is not None:
                if frame_id % sample_every == 0:
                    writer.add(lmks_list[0], current_lbl)
                    frame_count+=1

                cv2.putText(annotated_frame, f"Recording: {frame_count}",
                            (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), 1)
                cv2.circle(annotated_frame, (50,120), 20, (0,0,255),-1)

            # Show whether a hand is detected
            if lmks_list:
                cv2.putText(annotated_frame, "Hand detected", 
                           (10, annotated_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display current label and recording status
            if current_lbl is not None:
                gesture_name = self.labels.get(current_lbl, "Unknown")
                status = f"Gesture {gesture_name} | Recording: {'ON' if is_recording else 'OFF'}"
                cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Listen for key pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):         # Quit if "q" is pressed
                break
            elif ord("a") <= key <= ord("f"):

                new_label = key - ord('a')
                if new_label in self.labels:
                    if current_lbl == new_label:
                        # Toggle recording for current label
                        is_recording = not is_recording
                        
                        gesture_name = self.labels[new_label]

                        if is_recording:
                            frame_count = 0
                            print(f"Start recording for: {gesture_name}")
                        else:
                            print(f"Recording stopped. Wrote {frame_count} samples")
                    else:
                        # Pick new gesture
                        current_lbl = new_label
                        is_recording = False
                        frame_count = 0
                        gesture_name = self.labels[new_label]
                        print(f"Gesture picked: {gesture_name}")

            cv2.imshow(
                "Gesture Collector (press a to f to toggle, q to quit)", annotated_frame)

        cap.release()
        cv2.destroyAllWindows()
        writer.close()


if __name__ == "__main__":
    GestureDataCollector(config_path="config.yaml").collect(data_dir='src/data')