import time
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
import argparse
from sklearn.metrics import classification_report

# Import custom modules
from ..common.load_label_dict import load_label_dict
from ..common.gesture_classifier import HandGestureClassifier

from .early_stopper import EarlyStopper
from .get_dataloaders import get_dataloaders

class HandGestureTrainer:
    """
    Trainer class for the Hand Gesture Recognition model.
    
    Args:
        data_path (str): Path to the data directory containing CSV files.
        config_path (str): Path to the configuration file.
        experiment_path (str): Path to save the outputs from training runs.
    """
    def __init__(self, config_path: str, experiment_path: str):

        # Load configuration and determine number of classes
        self.config_path = config_path
        self.label_map = load_label_dict(self.config_path)
        self.num_classes = len(self.label_map)
        if self.num_classes == 0:
            raise ValueError("No gestures found in config file or config file not found.")
        
        self.experiment_path = experiment_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = HandGestureClassifier(input_size=60, num_classes=self.num_classes).to(self.device)  
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.early_stopper = EarlyStopper(patience=20, min_delta=0.001)

    def _train_epoch(self, train_loader):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data, labels in train_loader:
            # 1 - Move data to device
            data, labels = data.to(self.device), labels.to(self.device)

            # 2 - Forward pass, backward pass, optimize
            self.optimizer.zero_grad()              # Reset gradients
            outputs = self.model(data)              # Forward pass
            loss = self.criterion(outputs, labels)  # Compute loss
            loss.backward()                         # Backward pass
            self.optimizer.step()                   # Update weights

            # 3 - Accumulate loss and accuracy
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs.data, dim=1)
            correct_predictions += (predicted == labels).sum().item()

            total_samples += batch_size
            
        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    @torch.no_grad()
    def _validate_epoch(self, val_loader):
        """Validates the model for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data, labels in val_loader:
            # 1 - Move data to device
            data, labels = data.to(self.device), labels.to(self.device)

            # 2 - Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            # 3 - Accumulate loss and accuracy
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            total_samples += batch_size
            

        avg_loss = running_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    def fit(self, epochs: int = 100, train_loader=None, val_loader=None):
        """Trains the model for a specified number of epochs."""
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader=train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader=val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.early_stopper(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss.")
                break

        print("Training finished.")
        self.save_model()

    @torch.no_grad()
    def evaluate(self, test_loader=None):
        """Tests the trained model on the test set."""
        print("\n--- Evaluating model on test set ---")
        self.model.eval()
        all_preds = []
        all_labels = []

        for data, labels in test_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            outputs = self.model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Generate classification report
        # Map numerical labels back to gesture names for the report
        gesture_names = [self.label_map.get(i, str(i)) 
                         for i in sorted(self.label_map.keys())]
        
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

        report = classification_report(filtered_labels, filtered_preds, target_names=gesture_names, zero_division=0, output_dict=True)
        
        # Extract core metrics
        accuracy = report['accuracy']
        # For precision, recall, f1-score, we can take the weighted average
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        print("\n--- Core Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted avg): {precision:.4f}")
        print(f"Recall (weighted avg): {recall:.4f}")
        print(f"F1-Score (weighted avg): {f1_score:.4f}")

        # Convert report back to string for printing and saving
        report_str = classification_report(filtered_labels, filtered_preds, target_names=gesture_names, zero_division=0)
        print("\n--- Full Classification Report ---")
        print(report_str)

        # Save the report
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.experiment_path, f'run_{timestamp}')
        os.makedirs(run_path, exist_ok=True) # Ensure run_path exists
        report_path = os.path.join(run_path, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("--- Core Metrics ---\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision (weighted avg): {precision:.4f}\n")
            f.write(f"Recall (weighted avg): {recall:.4f}\n")
            f.write(f"F1-Score (weighted avg): {f1_score:.4f}\n")
            f.write("\n--- Full Classification Report ---\n")
            f.write(report_str)
        print(f"Classification report saved to {report_path}")

    def save_model(self):
        """Saves the trained model's state dictionary."""

        # create a new subdirectory for each training run based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.experiment_path, f'run_{timestamp}')
        model_path = os.path.join(run_path, 'model.pth')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved successfully to {model_path}")

def main():
    """
    Main function to parse arguments and initiate training.
    """
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument('--data_path', type=str, default='src/data/', help='Path to data directory containing CSV files')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--experiment_path', type=str, default='src/experiments/', help='Path to save the outputs from training runs')
    # Training-specific arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')

    args = parser.parse_args()

    trainer = HandGestureTrainer(config_path=args.config, experiment_path=args.experiment_path)

    loaders = get_dataloaders(data_path=args.data_path, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = loaders

    trainer.fit(epochs=args.epochs, train_loader=train_loader, val_loader=val_loader)
    trainer.evaluate(test_loader=test_loader)

if __name__ == "__main__":
    main()
