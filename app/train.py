import cv2
import torch
from app.model import SymbolCNN
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

class SymbolDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(Path(folder).glob("*/*.png"))
        if not self.paths:
            raise FileNotFoundError(
                f"No symbol images in {folder}. Run python -m app.generate_dataset."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        class_id = int(path.parent.name.split("_", 1)[0])

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (64, 64):
            raise ValueError(f"Expected a readable 64 x 64 image at {path}")

        image = torch.from_numpy(gray).float() / 255.0
        return image.unsqueeze(0), class_id


def main():
    torch.manual_seed(42)

    train_data = SymbolDataset("data/symbols/train")
    val_data = SymbolDataset("data/symbols/val")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    model = SymbolCNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                predicted = logits.argmax(dim=1)
                correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_data)
        val_accuracy = correct / len(val_data)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"val accuracy: {val_accuracy:.1%} ({correct}/{len(val_data)})"
        )

    checkpoint_path = Path("artifacts/symbol_cnn.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model weights to {checkpoint_path}")

if __name__ == "__main__":
    main()
