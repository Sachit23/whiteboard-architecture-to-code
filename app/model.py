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

conv = torch.nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1
)

features = conv(batch)
print("After convolution:", tuple(features.shape))

activated = torch.relu(features)

pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(activated)

print("After ReLU:", tuple(activated.shape))
print("After max pooling:", tuple(pooled.shape))

pooled_again = pool(pooled)
flattened = torch.flatten(pooled_again, start_dim=1)

classifier = torch.nn.Linear(in_features=8 * 16 * 16, out_features=4)
logits = classifier(flattened)

print("After second pooling:", tuple(pooled_again.shape))
print("After flattening:", tuple(flattened.shape))
print("Class scores:", logits)
print("Class score shape:", tuple(logits.shape))