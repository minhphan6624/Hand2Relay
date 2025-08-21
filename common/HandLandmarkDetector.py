import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional

class HandLandmarkDetector:
    """Detect & extract 21 hand landmarks from a frame"""

    def __init__(self, detection_conf=0.7, tracking_conf=0.5):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode = False, 
            max_num_hands = 1,
            min_detection_confidence = detection_conf,
            min_tracking_confidence = tracking_conf
        )
        self.drawer = mp.solutions.drawing_utils


    def detect_hand(self, frame: np.ndarray) -> Tuple[List[float], np.ndarray]:
        """Returns (landmarks_lists, frame_annotated)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()
        landmarks_list = []

        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(annotated, 
                                           h, self._mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in h.landmark]
                landmarks_list.append(landmarks)
        return landmarks_list, annotated