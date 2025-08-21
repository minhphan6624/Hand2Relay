import cv2, os
from common import HandLandMarkDetector, label_dict_from_yaml
from data_utils import HandDatasetWriter

class GestureDataCollector:
    def __init__(self, config="../config.yaml"):
        self.labels = label_dict_from_yaml(config)
        self.detector = HandLandMarkDetector()

    def collect(self, mode="train"):
        writer = HandDatasetWriter()
        cap = cv2.VideoCapture(0)
        current_lbl, recording = None, False
        while True:
            ok, frame = cap.read()

            # Break if not ok 
            if not ok:
                break
            
            # Detect the landmarks
            landmarks, _ = self.detector.detect_hand(frame)

            # If landmarks are detected and recording is turned on, write to file
            if recording and landmarks:
                writer.add(landmarks(0), current_lbl)
            
            # Listen for key pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):         # Quit if "q" is pressed
                break

            # Otherwise, if key is between "a" and "f" (alphabetically), label the frame
            if ord("a") <= key <= ord("f"):
                lbl = key - ord("a")
                if lbl in self.labels:
                    recording = (lbl == current_lbl) ^ recording
                    current_lbl = lbl

        cap.relese()
        cv2.destroyAllWindows()
        writer.close()