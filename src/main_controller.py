import os
import cv2
import time
import argparse
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from common.landmark_detector import HandLandmarksDetector
from common.gesture_classifier import HandGestureClassifier
from common.load_label_dict import load_label_dict
from common.normalize_landmarks import normalize_landmarks

from common.gesture_action_mapper import GestureActionMapper

from hardware.base_controller import HardwareController
from hardware.esp32_controller import ESP32Controller
from hardware.modbus_controller import ModbusController

class GestureControlSystem:
    def __init__(self, model_path: str, config_path: str = 'config.yaml', port: str = None, simulation: bool = True, mode: str = 'esp32'):
        
        self.detector = HandLandmarksDetector()
        self.labels = load_label_dict(config_path)
        self.num_classes = len(self.labels)
        if self.num_classes == 0:
            raise ValueError("No gestures found in config file or config file not found.")

        # Load model using trained weights
        self.model = HandGestureClassifier(60, self.num_classes) 
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                "cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model loaded from {model_path}")

        self.simulation = simulation
        self.mode = mode.lower()
        
        hardware_controller: HardwareController = None
        if not self.simulation:
            if port is None:
                raise ValueError("Serial port must be specified when not in simulation mode.")
            elif self.mode == "esp32":
                hardware_controller = ESP32Controller(port)
            elif self.mode == "modbus":
                hardware_controller = ModbusController(port)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        
        self.action_mapper = GestureActionMapper(hardware_controller, self.labels)

        # For debouncing gesture actions
        self.last_executed_gesture = "None"
        self.last_execution_time = time.time()
        self.debounce_delay = 1.0 # seconds to prevent rapid re-execution of the same gesture

    def _execute_gesture_action(self, gesture_name: str):
        """
        Executes the action mapped to the recognized gesture, with debouncing.
        """

        # Debounce: ignore if same gesture within delay period
        if gesture_name == self.last_executed_gesture and (time.time() - self.last_execution_time) < self.debounce_delay:
            return 

        print(f"[ACTION] Detected: {gesture_name}")
        self.action_mapper.execute_action(gesture_name, self.simulation)

        self.last_executed_gesture = gesture_name
        self.last_execution_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("[INFO] Starting real-time gesture recognition. Press 'q' to quit.")
        print(f"[INFO] Simulation mode: {self.simulate}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1) # Mirror image
                lmks_list, annotated_frame = self.detector.detect_hand(frame)

                predicted_gesture_name = "None"
                if lmks_list:
                    raw_landmarks = lmks_list[0]
                    normalized_lmks = normalize_landmarks(raw_landmarks) # Apply the same normalization as during training
                    
                    # Convert to tensor and drop x0, y0, z0 (wrist) features
                    landmarks_tensor = torch.tensor(normalized_lmks[3:]).float().unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction_id = self.model.predict(landmarks_tensor).item()
                    
                    if prediction_id != -1: # -1 means confidence threshold not met
                        predicted_gesture_name = self.labels.get(prediction_id, "Unknown")
                        self._execute_gesture_action(predicted_gesture_name)    
                    else:
                        predicted_gesture_name = "Low Confidence"

                # Display current gesture on frame
                cv2.putText(annotated_frame, f"Gesture: {predicted_gesture_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('Hand Gesture Recognition', annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.action_mapper.controller: # Check if a controller was initialized
                self.action_mapper.controller.all_off() # Ensure all devices are off on exit
                self.action_mapper.controller.close()
            print("[INFO] Application closed.")


def main():
    parser = argparse.ArgumentParser(description="Real-time Hand Gesture Control System")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model .pth file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--port', type=str, default=None,
                        help='Serial port for hardware (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux/macOS)')
    parser.add_argument('--simulation', action='store_true',
                        help='Run in simulation mode (print commands instead of sending to hardware)')
    parser.add_argument('--mode', type=str, default='esp32', choices=['esp32', 'modbus'],
                        help='Hardware control mode: "esp32" or "modbus"')

    args = parser.parse_args()

    # Ensure model and config files exist
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        return

    try:
        system = GestureControlSystem(
            model_path=args.model,
            config_path=args.config,
            port=args.port,
            simulation=args.simulation,
            mode=args.mode
        )
        system.run()
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
