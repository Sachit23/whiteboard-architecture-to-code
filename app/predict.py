import argparse
import cv2
import torch
from app.model import SymbolCNN
CLASS_NAMES = ("service", "database", "cache", "broker")

def load_crop(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {path}")
    return prepare_crop(gray)

def prepare_crop(gray):
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = torch.from_numpy(resized).float() / 255.0
    return pixels.unsqueeze(0)

def classify_batch(batch):
    model = SymbolCNN()
    weights = torch.load(
        "artifacts/symbol_cnn.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(weights)
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        return logits.argmax(dim=1).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Paths to one or more symbol crops")
    args = parser.parse_args()
    
    crops = [load_crop(path) for path in args.images]
    batch = torch.stack(crops, dim=0)
    class_ids = classify_batch(batch)

    for path, class_id in zip(args.images, class_ids):
        print(f"{path} → {CLASS_NAMES[class_id]}")

if __name__ == "__main__":
    main()
