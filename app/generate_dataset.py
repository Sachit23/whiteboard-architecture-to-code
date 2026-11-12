import cv2
import numpy as np
from pathlib import Path

def make_variant(prototype, rng):
    # The prototype has dark ink on a light background.
    # Convert it to a mask with white ink on black.
    ink_mask = (prototype < 128).astype(np.uint8) * 255

    # Thicken the strokes in roughly half the examples.
    if rng.random() < 0.5:
        kernel = np.ones((3, 3), dtype=np.uint8)
        ink_mask = cv2.dilate(ink_mask, kernel)

    scale = rng.uniform(0.75, 1.05)
    dx, dy = rng.integers(-4, 5, size=2)

    transform = cv2.getRotationMatrix2D((32, 32), 0, scale)
    transform[0, 2] += dx
    transform[1, 2] += dy

    moved_mask = cv2.warpAffine(
        ink_mask, transform, (64, 64), flags=cv2.INTER_NEAREST
    )

    background = rng.integers(210, 256)
    ink = rng.integers(15, 161)
    image = np.full((64, 64), background, dtype=np.uint8)
    image[moved_mask > 0] = ink
    return image


def main():
    prototypes = sorted(Path("data/prototypes").glob("*.png"))
    if len(prototypes) != 4:
        raise ValueError("Expected four prototypes. Run app.generate_symbols first.")

    splits = (("train", 100, 42), ("val", 20, 43))

    for split, examples_per_class, seed in splits:
        rng = np.random.default_rng(seed)

        for path in prototypes:
            prototype = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if prototype is None:
                raise OSError(f"Could not read {path}")

            output_dir = Path("data/symbols") / split / path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            for index in range(examples_per_class):
                image = make_variant(prototype, rng)
                output_path = output_dir / f"{index:03d}.png"
                if not cv2.imwrite(str(output_path), image):
                    raise OSError(f"Could not write {output_path}")

        print(f"{split}: {len(prototypes) * examples_per_class} images")


if __name__ == "__main__":
    main()