import os
import csv
from typing import List


class HandDatasetWriter:
    """Write 63 landmark features + labels to a csv file"""

    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.f = open(csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        header = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
        self.w.writerow(header + ["label"])
        self.count = 0

    def add(self, landmarks: List[float], label: int):
        if len(landmarks) != 63:
            raise ValueError("Expect 63 floats (21*3).")
        self.w.writerow(landmarks + [label])
        self.count += 1

    def close(self):
        self.f.close()
        print(f"[INFO] wrote {self.count} samples.")
