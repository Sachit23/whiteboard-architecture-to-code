import argparse
import json
import cv2
import torch
from pathlib import Path
from app.predict import CLASS_NAMES, classify_batch, prepare_crop

def find_horizontal_neighbors(arrow_box, nodes, max_gap=30):
    arrow_x, arrow_y, arrow_width, arrow_height = arrow_box
    arrow_right = arrow_x + arrow_width
    arrow_center_y = arrow_y + arrow_height / 2

    left_node = None
    right_node = None
    best_left_gap = max_gap + 1
    best_right_gap = max_gap + 1

    for node_id, (x, y, width, height) in nodes:
        # Check whether the arrow's centerline crosses the symbol's height.
        if not (y <= arrow_center_y < y + height):
            continue

        left_gap = arrow_x - (x + width)
        if 0 <= left_gap < best_left_gap:
            left_node = node_id
            best_left_gap = left_gap

        right_gap = x - arrow_right
        if 0 <= right_gap < best_right_gap:
            right_node = node_id
            best_right_gap = right_gap

    return left_node, right_node

def horizontal_arrow_direction(mask, box):
    x, y, width, height = box
    arrow = mask[y:y + height, x:x + width]
    middle = width // 2

    # Take equally wide slices from both ends of the crop.
    left = arrow[:, :middle]
    right = arrow[:, -middle:]

    left_ink = cv2.countNonZero(left)
    right_ink = cv2.countNonZero(right)

    if left_ink == right_ink:
        return "unknown"

    return "right" if right_ink > left_ink else "left"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a full diagram image")
    parser.add_argument("--output", type=Path, help="Where to save the graph JSON")
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

    nodes = []
    graph = {"nodes": [], "edges": [], "unresolved_arrows": []}

    for index, (box, class_id) in enumerate(zip(boxes, class_ids), start=1):
        symbol_type = CLASS_NAMES[class_id]
        node_id = f"{symbol_type}_{index}"

        nodes.append((node_id, box))
        graph["nodes"].append({
            "id": node_id,
            "type": symbol_type,
            "box": list(box),
        })
        print(f"{node_id} at {box}")

    for arrow_box in arrow_candidates:
        direction = horizontal_arrow_direction(mask, arrow_box)
        left_node, right_node = find_horizontal_neighbors(arrow_box, nodes)

        if direction == "unknown" or left_node is None or right_node is None:
            graph["unresolved_arrows"].append(list(arrow_box))
            print(f"Unresolved arrow at {arrow_box}")
            continue

        if direction == "right":
            source, target = left_node, right_node
        else:
            source, target = right_node, left_node

        graph["edges"].append({"source": source, "target": target})
        print(f"Connection: {source} → {target}")

    if args.output is not None:
        args.output.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        print(f"Saved graph to {args.output}")


if __name__ == "__main__":
    main()