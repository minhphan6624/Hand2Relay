# Hand2Relay: Touchless Relay Control System via Hand Gesture Recognition

## Project Overview

This project develops a touchless relay control system using real-time hand gesture recognition, computer vision, and machine learning. Users can control electrical devices (lights, fans, pumps) remotely by performing specific hand gestures in front of a camera. The system supports two hardware approaches: an ESP32 microcontroller and an RS485 Modbus relay module.

## Features

*   **Real-time Hand Gesture Recognition**: Utilizes MediaPipe for accurate hand landmark detection and a trained Neural Network model for gesture classification.
*   **Dual Hardware Support**: Seamlessly integrates with both ESP32 microcontrollers (via UART) and RS485 Modbus RTU relay modules.
*   **Configurable Gestures**: Gesture-to-action mappings are defined in `config.yaml`, allowing for easy customization and expansion.
*   **Simulation Mode**: Run the system without physical hardware to test gesture recognition and action mapping logic.

## Project Structure

```
.
├── config.yaml                 # System configuration, gesture mappings
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── firmware/
│   └── esp32.ino               # Arduino firmware for ESP32
├── src/
│   ├── common/
│   │   ├── models.py           # HandLandmarksDetector, HandGestureClassifier, utilities
│   │   └── gesture_action_mapper.py # Maps gestures to hardware actions
│   ├── data/                   # Stores collected hand landmark datasets (CSV)
│   ├── hardware/
│   │   ├── base_controller.py  # Abstract base class for hardware controllers
│   │   ├── esp32_controller.py # PC-side controller for ESP32 via UART
│   │   └── modbus_controller.py# PC-side controller for Modbus RTU via RS485
│   ├── processing/
│   │   ├── data_collector.py   # Script for collecting hand gesture data
│   │   └── ...                 # Other data processing scripts
│   ├── train/
│   │   ├── models/             # Stores trained PyTorch models (.pth)
│   │   └── trainer.py          # Script for training the gesture classification model
│   └── main_controller.py      # Main entry point for real-time gesture control
└── AIO2025-AIoT-GestureRecognition.pdf # Project documentation (Vietnamese)
```

## Setup

### Prerequisites

*   Python 3.10
*   Webcam
*   (Optional, for hardware control) ESP32 microcontroller or RS485 Modbus Relay Module

### Environment Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/minhphan6624/Hand2Relay.git
    cd Hand2Relay
    ```
2.  **Create a virtual environment** (recommended using `conda`):
    ```bash
    conda create -n gesture_env python=3.10
    conda activate gesture_env
    ```
    If you don't use `conda`, you can use `venv`:
    ```bash
    python -m venv gesture_env
    source gesture_env/bin/activate # On Windows: .\gesture_env\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `mediapipe`, try downgrading `opencv-python`: `pip install opencv-python==4.8.1.78`*

### Configuration (`config.yaml`)

The `config.yaml` file defines the gesture mappings and sensor settings. Ensure the gesture IDs (0-5) match your trained model's labels.

```yaml
gestures:
  0: "light1_on"
  1: "light1_off"
  2: "light2_on"
  3: "light2_off"
  4: "all_on"
  5: "all_off"

sensor_settings:
  max_hands: 1
  confidence_threshold: 0.7
  tracking_confidence: 0.5
```

## Usage

### 1. Data Collection

To collect hand landmark data for training your gesture recognition model:

```bash
python src/processing/data_collector.py
```

*   **Instructions**:
    *   Show the hand gesture you want to record data for.
    *   Press the key corresponding to the gesture ('a' to 'f') based on your `config.yaml` (e.g., 'a' for gesture ID 0, 'b' for gesture ID 1, etc.). A text overlay will indicate if the camera is idle or recording.
    *   To stop recording for that gesture, press the same key again.
    *   When finished, press 'q' to quit the recorder.
*   **Output**: Collected data will be saved in a CSV file (e.g., `src/data/landmarks_train.csv`).

### 2. Model Training

After collecting data, train your gesture classification model:

```bash
python src/train/trainer.py
```
*   This script will train a Neural Network model using the collected data and save the trained model (`hand_gesture_model.pth`) in `src/train/models/`.

### 3. Real-time Gesture Control (`main_controller.py`)

This is the main application for real-time gesture recognition and relay control.

#### Simulation Mode (No Hardware Required)

To test the gesture recognition and action mapping without physical hardware:

```bash
python src/main_controller.py --simulation
```
*   The system will print the detected gestures and corresponding actions to the console.

#### Hardware Control Mode

**Important**: Ensure your hardware (ESP32 or Modbus Relay) is correctly connected and configured before running in this mode. Refer to the "Hardware Connection (Modbus RTU)" section below for details.

*   **Using ESP32**:
    ```bash
    python src/main_controller.py --port /dev/ttyUSB0 --mode esp32
    # (Replace /dev/ttyUSB0 with your ESP32's serial port, e.g., COM3 on Windows)
    ```
*   **Using Modbus RTU Relay Module**:
    ```bash
    python src/main_controller.py --port /dev/ttyUSB0 --mode modbus
    # (Replace /dev/ttyUSB0 with your RS485 adapter's serial port, e.g., COM3 on Windows)
    ```
*   **Common Arguments**:
    *   `--model`: Path to your trained model file (default: `src/train/models/hand_gesture_model.pth`).
    *   `--config`: Path to your configuration YAML file (default: `config.yaml`).

## Hardware Connection (Modbus RTU)

For controlling the Modbus RTU relay module:

1.  **RS-485 Wiring**: Connect `A (D+)` of your USB-RS485 converter to `A` on the relay module, and `B (D-)` to `B`. Do not reverse.
2.  **Power Supply**: Provide 12V/1A power to the relay module.
3.  **USB-RS485 Converter**: Plug the converter into your PC. Identify the assigned serial port (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux/macOS).
4.  **Baud Rate**: Ensure the baud rate is 9600 (8 data bits, no parity, 1 stop bit - 8N1), which is standard for most Modbus RTU relays.

## Gesture Mapping

The `GestureActionMapper` class translates recognized gestures into hardware commands. The mappings are based on the `gestures` section in `config.yaml`. For example:
*   `"light1_on"` will activate relay 1.
*   `"all_off"` or `"turn_off"` will deactivate all relays.

## Troubleshooting

*   **Webcam not opening**: Ensure no other application is using the webcam and check camera permissions.
*   **Serial Port Errors**: Verify the correct serial port is specified and that the hardware is properly connected and powered. Check baud rate settings.
*   **Low Confidence/Incorrect Gestures**: Ensure consistent lighting, a plain background, and perform gestures clearly. Retrain the model with more diverse data if necessary.
*   **Debouncing**: The system includes a 1.0-second debounce delay to prevent rapid re-execution of the same gesture.
* For any other bugs, please excuse the author since the project is still under development nha hihi =)))