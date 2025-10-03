import numpy as np
from typing import List

def normalize_landmarks(landmarks_flat: List[float]) -> np.ndarray:
    """
    Normalizes hand landmarks relative to the wrist (landmark 0) and scales them.
    Args:
        landmarks_flat (List[float]): A flat list of 63 floats representing 21 hand landmarks (x, y, z) each.
    Returns:
        np.ndarray: A normalized array of shape (63,) representing the normalized landmarks.
    """
    landmarks = np.array(landmarks_flat).reshape(-1, 3) # Reshape to (21, 3) for x,y,z
    
    wrist = landmarks[0] # Use the wrist (landmark 0) as the reference point
    translated_landmarks = landmarks - wrist # Translate all landmarks so the wrist is at the origin
    
    # Calculate the scale factor based on the distance from wrist to a key point (e.g., middle finger MCP joint, landmark 9)
    # This makes the gesture size-invariant
    scale_factor = np.linalg.norm(translated_landmarks[9]) # Distance from wrist to middle finger base
    if scale_factor == 0:
        scale_factor = 1e-6 # Add a small epsilon to prevent division by zero
    
    normalized_landmarks = translated_landmarks / scale_factor
    
    return normalized_landmarks.flatten() # Flatten back to 63-element vector