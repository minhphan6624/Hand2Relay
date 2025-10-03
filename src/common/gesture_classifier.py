import torch
import torch.nn as nn

class HandGestureClassifier(nn.Module):
    """Classifier model to detect 6 gesture classes"""

    def __init__(self, input_size : int = 60, num_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    # Forward pass
    def forward(self, x): 
        return self.net(x)

    # Predict class with thresholding
    @torch.no_grad()
    def predict(self, x, thresh=0.7):
        probs = torch.softmax(self.forward(x), dim=1) # Run forward pass and get probabilities
        
        max_p, pred = torch.max(probs, 1) # Returns the highest probability in the probs tensor and its index (class)

        return pred if max_p.item() >= thresh else torch.tensor(-1) # Return the class index if above threshold, else -1