import cv2
import torch

class SymbolCNN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = torch.nn.Linear(8*16*16, 4)
    
    def forward(self, images):
        features = self.conv(images)
        activated = torch.relu(features)
        pooled = self.pool(activated)
        pooled_again = self.pool(pooled)
        flattened = torch.flatten(pooled_again, start_dim=1)
        logits = self.classifier(flattened)
        return logits

def main():
    gray = cv2.imread("box_1.png", cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError("Could not load box_1.png. Run preprocessing first.")

    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = torch.from_numpy(resized).float() / 255.0

    with_channel = pixels.unsqueeze(0)
    batch = with_channel.unsqueeze(0)

    print("Original crop:", gray.shape)
    print("Resized:", tuple(pixels.shape))
    print("With channel:", tuple(with_channel.shape))
    print("CNN input:", tuple(batch.shape))

    model = SymbolCNN()
    logits = model(batch)

    print(logits)
    print("Class score shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
