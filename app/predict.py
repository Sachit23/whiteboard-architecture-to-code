import argparse
import cv2
import torch
from app.model import SymbolCNN
CLASS_NAMES = ("service", "database", "cache", "broker")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a single symbol crop")
    args = parser.parse_args()
    
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(gray).float() / 255.0
    batch = image.unsqueeze(0).unsqueeze(0)
    
    model = SymbolCNN()
    weights = torch.load("artifacts/symbol_cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    
    with torch.no_grad():
        logits = model(batch)
        class_id = logits.argmax(dim=1).item()

    print(f"Predicted symbol: {CLASS_NAMES[class_id]}")

if __name__ == "__main__":
    main()