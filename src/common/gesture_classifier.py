import torch
import torch.nn as nn

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

    def forward(self, x): 
        return self.net(x)

    @torch.no_grad()
    def predict(self, x, thresh=0.7):
        probs = torch.softmax(self.forward(x), dim=1)
        max_p, pred = torch.max(probs, 1)
        return pred if max_p.item() >= thresh else torch.tensor(-1)