import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import yaml
from typing import List, Tuple, Optional, Dict


class HandLandmarksDetector:
    """Detect & extract 21 hand landmarks from a frame"""

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5, max_hands=1):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self._drawer = mp.solutions.drawing_utils

    def detect_hand(self, frame: np.ndarray) -> Tuple[Optional[List[List[float]]], np.ndarray]:
        """Return (list_of_flat_63_lists or None, annotated_frame)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        annotated = frame.copy()
        lmks_list = []
        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                self._drawer.draw_landmarks(
                    annotated, h, self._mp_hands.HAND_CONNECTIONS)
                # flatten to 63 floats (x,y,z for 21 landmarks)
                lmks = [c for lm in h.landmark for c in (lm.x, lm.y, lm.z)]
                lmks_list.append(lmks)
        return (lmks_list or None), annotated


class HandGestureClassifier(nn.Module):
    """Classifier model to detect 6 gesture classes"""

    def __init__(self, input_size=60, num_classes=6): # Changed input_size from 63 to 60
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # logits
        return self.net(x)

    @torch.no_grad()
    def predict(self, x, thresh=0.7):
        probs = torch.softmax(self.forward(x), dim=1)
        max_p, pred = torch.max(probs, 1)
        return pred if max_p.item() >= thresh else torch.tensor(-1)


def label_dict_from_config_file(path: str = "config.yaml") -> Dict[int, str]:
    """Load {id: gesture_name} from YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("gestures", {})
    except FileNotFoundError:
        print(f"[WARN] File {path} not found")
        return {}
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}

def normalize_landmarks(landmarks_flat: List[float]) -> np.ndarray:
    """
    Normalizes hand landmarks relative to the wrist (landmark 0) and scales them.
    Input: a flattened list of 63 landmark coordinates (x0,y0,z0, x1,y1,z1, ...)
    Output: a numpy array of normalized and scaled 63 landmark coordinates.
    """
    landmarks = np.array(landmarks_flat).reshape(-1, 3) # Reshape to (21, 3) for x,y,z
    
    # Use the wrist (landmark 0) as the reference point
    wrist = landmarks[0]
    
    # Translate all landmarks so the wrist is at the origin
    translated_landmarks = landmarks - wrist
    
    # Calculate the scale factor based on the distance from wrist to a key point (e.g., middle finger MCP joint, landmark 9)
    # This makes the gesture size-invariant
    # Avoid division by zero if all points are identical (e.g., a single point)
    scale_factor = np.linalg.norm(translated_landmarks[9]) # Distance from wrist to middle finger base
    if scale_factor == 0:
        scale_factor = 1e-6 # Add a small epsilon to prevent division by zero
    
    normalized_landmarks = translated_landmarks / scale_factor
    
    return normalized_landmarks.flatten() # Flatten back to 63-element vector
