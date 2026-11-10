import cv2
import numpy as np

# gray = np.array(
#     [[245, 220, 240],
#      [30, 127, 180]],
#     dtype=np.uint8
# )

# Generate a sample diagram.
canvas = np.full((180, 360), 245, dtype=np.uint8)
cv2.rectangle(canvas, (25, 45), (145, 135), 30, 3)
cv2.rectangle(canvas, (215, 45), (335, 135), 180, 3)
cv2.imwrite("sample_diagram.png", canvas)

# Load the PNG and create two masks.
gray = cv2.imread("sample_diagram.png", cv2.IMREAD_GRAYSCALE)

for cutoff in (127, 200):
    _, mask = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"mask_{cutoff}.png", mask)

# Use the mask with both outlines, created in the final loop iteration.
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

boxes = [cv2.boundingRect(contour) for contour in contours]
boxes.sort(key=lambda box: box[0])

for number, (x, y, width, height) in enumerate(boxes, start=1):
    crop = gray[y:y + height, x:x + width]
    cv2.imwrite(f"box_{number}.png", crop)
    print(f"Box {number}: x={x}, y={y}, width={width}, height={height}")