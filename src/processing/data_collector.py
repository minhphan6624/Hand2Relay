import cv2
import os
import argparse

# Add project root to path for cross-folder imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import HandLandmarksDetector, label_dict_from_config_file
from processing.data_writer import HandDatasetWriter

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        current_lbl = None
        is_recording = False
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read from camera")
                break

            # Detect the landmarks
            lmks_list, annotated_frame = self.detector.detect_hand(frame)

            # If landmarks are detected and recording is turned on, write to file
            if is_recording and lmks_list and current_lbl is not None:
                writer.add(lmks_list[0], current_lbl)
                frame_count+=1

                cv2.putText(annotated_frame, f"Recording: {frame_count}",
                            (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), -1)
                cv2.circle(annotated_frame, (50,120), 20, (0,0,255),-1)

            # Display current label and recording status
            if current_lbl is not None:
                gesture_name = self.labels.get(current_lbl, "Unknown")
                status = f"Gesture {gesture_name} | Recording: {'ON' if is_recording else 'OFF'}"
                cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show whether a hand is detected
            if lmks_list:
                cv2.putText(annotated_frame, "Hand detected", 
                           (10, annotated_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(
                "Gesture Collector (press a–f to toggle, q to quit)", annotated_frame)

            # Listen for key pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):         # Quit if "q" is pressed
                break
            elif ord("a") <= key <= ord("f"):

                new_label = key - ord('a')
                if new_label in self.gesture_labels:
                    if current_label == new_label:
                        # Toggle recording for current label
                        is_recording = not is_recording
                        
                        gesture_name = self.gesture_labels[new_label]

                        if is_recording:
                            frame_count = 0
                            print(f"Start recording for: {gesture_name}")
                        else:
                            print(f"Recording stopped. Wrote {frame_count} samples")
                    else:
                        # Pick new gesture
                        current_label = new_label
                        is_recording = False
                        frame_count = 0
                        gesture_name = self.gesture_labels[new_label]
                        print(f"Gesture picked: {gesture_name}")

        cap.release()
        cv2.destroyAllWindows()
        writer.close()


if __name__ == "__main__":
    GestureDataCollector(config_path="config.yaml").collect()
