"""
coco_to_yolo.py
Converts COCO Segmentation JSON annotations to YOLOv8 segmentation format.

Usage:
    python src/coco_to_yolo.py
"""

import json
import os


def coco_to_yolo(json_path: str, output_dir: str) -> None:
    """
    Convert COCO segmentation annotations to YOLOv8 polygon format.

    Each output .txt file contains one line per annotation:
        <class_id> <x1> <y1> <x2> <y2> ... (normalized 0-1)

    Args:
        json_path:   Path to _annotations.coco.json
        output_dir:  Directory where .txt label files will be saved
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Map COCO category_id -> sequential YOLO class index
    # Skip the top-level supercategory (supercategory == 'none') which is
    # Roboflow's project name, not a real class.
    real_cats = [cat for cat in data["categories"] if cat["supercategory"] != "none"]
    categories = {cat["id"]: i for i, cat in enumerate(real_cats)}

    # Map image_id -> image metadata
    images = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    annotations_by_image: dict = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    os.makedirs(output_dir, exist_ok=True)

    for img_id, anns in annotations_by_image.items():
        img_info = images[img_id]
        w, h = img_info["width"], img_info["height"]
        base_name = os.path.splitext(img_info["file_name"])[0]
        out_path = os.path.join(output_dir, base_name + ".txt")

        with open(out_path, "w") as f:
            for ann in anns:
                # Skip annotations whose category was filtered out
                if ann["category_id"] not in categories:
                    continue

                cat_id = categories[ann["category_id"]]

                # COCO polygons: [x1, y1, x2, y2, ...]
                segmentation = ann["segmentation"][0]

                # Normalize coordinates by image dimensions
                normalized = []
                for i in range(0, len(segmentation), 2):
                    normalized.append(str(segmentation[i] / w))
                    normalized.append(str(segmentation[i + 1] / h))

                f.write(f"{cat_id} " + " ".join(normalized) + "\n")

    print(f"Labels saved to: {output_dir}")


if __name__ == "__main__":
    coco_to_yolo(
        json_path="data/train/_annotations.coco.json",
        output_dir="data/labels",
    )