import torch, torch.nn as nn

class HandGestureClassifier(nn.Module):
    """Classifier model to detect 6 gesture classes"""
    def __init__(self, input_size=63, num_classes=6):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
    @torch.no_grad
    def predict(self, x, threshold=0.7):
        probs = torch.softmax(self(x), dim=1)
        