import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

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