from pathlib import Path
from .custom_dataset import HandGestureDataset
from torch.utils.data import DataLoader

def get_dataloaders(data_path: str, batch_size: int = 64):
    """Loads datasets and creates DataLoaders."""

    data_path = Path(data_path)
    train_csv = data_path / "landmarks_train.csv"
    val_csv = data_path / "landmarks_val.csv"
    test_csv = data_path / "landmarks_test.csv"

    train_dataset = HandGestureDataset(train_csv)
    val_dataset = HandGestureDataset(val_csv)
    test_dataset = HandGestureDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded datasets: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")

    return train_loader, val_loader, test_loader