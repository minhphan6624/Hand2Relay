import cv2
import torch
import argparse
import os
import sys
import time

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from common.models import HandLandmarksDetector, HandGestureClassifier, label_dict_from_config_file, normalize_landmarks

# Placeholder for Arduino communication (will be implemented later)
class ArduinoController:
    def __init__(self, port: str):
        print(f"[INFO] Initializing ArduinoController on port {port} (simulated).")
        self.port = port
        # In a real scenario, you would open a serial connection here
        # self.ser = serial.Serial(port, 9600, timeout=1)

    def send_command(self, command: str):
        print(f"[SIM] Sending command '{command}' to Arduino on {self.port}")
        # In a real scenario, you would send the command over serial
        # self.ser.write(command.encode())

    def all_off(self):
        print(f"[SIM] Sending ALL_OFF command to Arduino on {self.port}")
        # self.ser.write(b'ALL_OFF')

    def close(self):
        print(f"[SIM] Closing Arduino connection on {self.port}")
        # if self.ser: self.ser.close()


class GestureControlSystem:
    def __init__(self, model_path: str, config_path: str, port: str = None, simulate: bool = True):
        self.detector = HandLandmarksDetector()
        self.labels = label_dict_from_config_file(config_path)
        self.num_classes = len(self.labels)
        if self.num_classes == 0:
            raise ValueError("No gestures found in config file or config file not found.")

        self.model = HandGestureClassifier(60, self.num_classes) # Updated input_size to 60
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print(f"[INFO] Model loaded from {model_path}")

        self.simulate = simulate
        self.arduino_controller = None
        if not self.simulate:
            if port is None:
                raise ValueError("Serial port must be specified when not in simulation mode.")
            self.arduino_controller = ArduinoController(port) # Will be replaced with actual pyserial logic

        self.current_gesture = "None"
        self.last_executed_gesture = "None"
        self.last_execution_time = time.time()
        self.debounce_delay = 1.0 # seconds to prevent rapid re-execution of the same gesture

    def _execute_gesture_action(self, gesture_name: str):
        if gesture_name == self.last_executed_gesture and (time.time() - self.last_execution_time) < self.debounce_delay:
            return # Debounce: prevent rapid re-execution

        print(f"[ACTION] Detected: {gesture_name}")
        if self.simulate:
            print(f"[SIMULATION] Executing action for: {gesture_name}")
        else:
            # This is where you'd map gesture_name to Arduino commands
            # For now, we'll use a simple mapping, assuming Arduino understands these strings
            self.arduino_controller.send_command(gesture_name)

        self.last_executed_gesture = gesture_name
        self.last_execution_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)
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
                    # Assuming only one hand is detected (max_num_hands=1)
                    raw_landmarks = lmks_list[0]
                    
                    # Apply the same normalization as during training
                    normalized_lmks = normalize_landmarks(raw_landmarks)
                    
                    # Convert to tensor and drop x0, y0, z0 (wrist) features
                    # The first 3 elements (x0, y0, z0) are now redundant after normalization
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
            if self.arduino_controller:
                self.arduino_controller.all_off() # Ensure all devices are off on exit
                self.arduino_controller.close()
            print("[INFO] Application closed.")


def main():
    parser = argparse.ArgumentParser(description="Real-time Hand Gesture Control System")
    parser.add_argument('--model', type=str, default='src/train/models/hand_gesture_model.pth',
                        help='Path to the trained model .pth file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--port', type=str, default=None,
                        help='Serial port for Arduino (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux/macOS)')
    parser.add_argument('--simulation', action='store_true',
                        help='Run in simulation mode (print commands instead of sending to Arduino)')

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
            simulate=args.simulation
        )
        system.run()
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
