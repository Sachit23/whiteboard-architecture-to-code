import argparse

import cv2
import torch

from app.predict import CLASS_NAMES, classify_batch, prepare_crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a full diagram image")
    args = parser.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = sorted(
        (cv2.boundingRect(contour) for contour in contours),
        key=lambda box: box[0],
    )
    
    symbol_boxes = []
    arrow_candidates = []
    
    for box in boxes:
        x, y, width, height = box
        if width/height >= 2.0:
            arrow_candidates.append(box)
        else:
            symbol_boxes.append(box)
            
    for box in arrow_candidates:
        print(f"Arrow candidate at {box}")
    
    boxes = symbol_boxes
    
    if not boxes:
        print("No symbols found.")
        return

    crops = []
    for x, y, width, height in boxes:
        gray_crop = gray[y:y + height, x:x + width]
        crops.append(prepare_crop(gray_crop))

    batch = torch.stack(crops, dim=0)
    class_ids = classify_batch(batch)

    for box, class_id in zip(boxes, class_ids):
        print(f"{CLASS_NAMES[class_id]} at {box}")


if __name__ == "__main__":
    main()