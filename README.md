# Hand2Relay - Touchless relay control system via hand recognition

## How to run the system

1. How to run the data collector:
- Install dependencies (pip install -r requirements.txt)
- Navigate to project root 
- Run python -m src.processing.data_collector.py
- Show the hand gesture you want to record data for
- Press the key corresponding to the gesture ('a' to 'f') based on the config.yaml file. A text will be displayed to indicate whether the camera is idle or recording any gesture.
- To stop recording for that gesture, press the same key again.
- When you're finished collecting data, press 'q' to quit the recorder.

**Note**: The colelcted data will be saved in a CSV file insde src/data. The filename will be landmarks_all.csv.
