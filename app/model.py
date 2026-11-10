import cv2
import torch

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