import cv2
import numpy as np
from pathlib import Path

def main():
    output_dir = Path("data/prototypes")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ("service", "database", "cache", "broker")
    background = 245
    ink = 30
    thickness = 2

    for class_id, label in enumerate(labels):
        image = np.full((64, 64), background, dtype=np.uint8)

        if label == "service":
            cv2.rectangle(image, (12, 16), (52, 48), ink, thickness)

        elif label == "database":
            cv2.ellipse(image, (32, 18), (20, 7), 0, 0, 360, ink, thickness)
            cv2.line(image, (12, 18), (12, 46), ink, thickness)
            cv2.line(image, (52, 18), (52, 46), ink, thickness)
            cv2.ellipse(image, (32, 46), (20, 7), 0, 0, 180, ink, thickness)

        elif label == "cache":
            corners = np.array(
                [(32, 10), (54, 32), (32, 54), (10, 32)], dtype=np.int32
            )
            cv2.polylines(image, [corners], True, ink, thickness)

        else:  # broker
            corners = np.array(
                [(22, 16), (54, 16), (42, 48), (10, 48)], dtype=np.int32
            )
            cv2.polylines(image, [corners], True, ink, thickness)

        path = output_dir / f"{class_id}_{label}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
        print(path)

if __name__ == "__main__":
    main()