import os
import csv
from typing import List
from datetime import datetime

class HandDatasetWriter:
    """Write 63 landmark features + labels to a csv file"""

    def __init__(self, csv_path: str):
        # Generate a new filename with a timestamp
        base, ext = os.path.splitext(csv_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_csv_path = f"{base}_{timestamp}{ext}"

        os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)

        self.file = open(new_csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        header = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
        self.writer.writerow(header + ["label"])
        
        self.count = 0

    def add(self, landmarks: List[float], label: int):
        """
        Validate and add a new sample to the dataset.
        Args:
            landmarks (List[float]): A list of 63 floats representing 21 3D landmarks.
            label (int): The label associated with the landmarks.
        """
        if len(landmarks) != 63:
            raise ValueError("Expect 63 floats (21*3).")
        self.writer.writerow(landmarks + [label])
        self.count += 1

    def close(self):
        self.file.close()
        print(f"[INFO] wrote {self.count} samples.")
