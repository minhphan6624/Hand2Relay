import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import yaml
import os
import argparse
from sklearn.metrics import classification_report

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from common.models import HandGestureClassifier, label_dict_from_config_file
from early_stopper import EarlyStopper
from custom_dataset import HandGestureDataset

class HandGestureTrainer:
    """Trainer class for the Hand Gesture Recognition model."""
    def __init__(self, model_path: str, config_path: str, device: str = 'cpu'):
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load configuration and determine number of classes
        self.label_map = label_dict_from_config_file(self.config_path)
        self.num_classes = len(self.label_map)
        if self.num_classes == 0:
            raise ValueError("No gestures found in config file or config file not found.")

        # Initialize model
        self.model = HandGestureClassifier(input_size=63, num_classes=self.num_classes).to(self.device)
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.early_stopper = EarlyStopper(patience=20, min_delta=0.001) # Adjusted patience and delta

    def load_datasets(self, train_csv: str, val_csv: str, test_csv: str, batch_size: int = 64):
        """Loads datasets and creates DataLoaders."""
        train_dataset = HandGestureDataset(train_csv)
        val_dataset = HandGestureDataset(val_csv)
        test_dataset = HandGestureDataset(test_csv)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Loaded datasets: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")

    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validates the model for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def train(self, epochs: int = 100):
        """Trains the model for a specified number of epochs."""
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.early_stopper(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss.")
                break

        print("Training finished.")
        self.save_model()

    def test_model(self):
        """Tests the trained model on the test set."""
        print("\n--- Evaluating model on test set ---")
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Generate classification report
        # Map numerical labels back to gesture names for the report
        gesture_names = [self.label_map.get(i, str(i)) for i in sorted(self.label_map.keys())]
        # Ensure the report uses the correct order of classes
        # We need to map the predicted/actual numerical labels to the sorted gesture names
        # For classification_report, we need the numerical labels and the target_names
        # Let's ensure the labels are sorted numerically and then map to names
        sorted_numerical_labels = sorted(self.label_map.keys())
        
        # Create a mapping from numerical label to its index in the sorted list
        label_to_index = {label: i for i, label in enumerate(sorted_numerical_labels)}
        
        # Convert predictions and labels to indices based on sorted_numerical_labels
        indexed_preds = [label_to_index.get(p, -1) for p in all_preds] # -1 for unknown predictions
        indexed_labels = [label_to_index.get(l, -1) for l in all_labels]

        # Filter out any -1 indices if they exist (though they shouldn't with proper data)
        valid_indices = [i for i, (p, l) in enumerate(zip(indexed_preds, indexed_labels)) if p != -1 and l != -1]
        
        filtered_preds = [indexed_preds[i] for i in valid_indices]
        filtered_labels = [indexed_labels[i] for i in valid_indices]

        if not filtered_preds or not filtered_labels:
            print("No valid predictions or labels to generate report.")
            return

        report = classification_report(filtered_labels, filtered_preds, target_names=gesture_names, zero_division=0)
        print("\n--- Classification Report ---")
        print(report)

        # Save the report
        report_path = os.path.join(os.path.dirname(self.model_path), 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")


    def save_model(self):
        """Saves the trained model's state dictionary."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved successfully to {self.model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument('--train_data', type=str, default='src/data/landmarks_train.csv', help='Path to training data CSV')
    parser.add_argument('--val_data', type=str, default='src/data/landmarks_val.csv', help='Path to validation data CSV')
    parser.add_argument('--test_data', type=str, default='src/data/landmarks_test.csv', help='Path to test data CSV')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_save_path', type=str, default='src/train/models/hand_gesture_model.pth', help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')

    args = parser.parse_args()

    # Ensure the data files exist before proceeding
    if not all(os.path.exists(f) for f in [args.train_data, args.val_data, args.test_data, args.config]):
        print("Error: One or more data/config files not found. Please ensure they exist.")
        print(f"Looking for: {args.train_data}, {args.val_data}, {args.test_data}, {args.config}")
        return

    trainer = HandGestureTrainer(model_path=args.model_save_path, config_path=args.config, device=args.device)
    trainer.load_datasets(args.train_data, args.val_data, args.test_data, args.batch_size)
    trainer.train(epochs=args.epochs)
    trainer.test_model()

if __name__ == "__main__":
    main()
