"""
visualize.py
Run YOLOv8-seg inference, calculate mask area percentages, and overlay custom labels.

Usage:
    python src/visualize.py --source data/test_image.jpg
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


def run_visualization(
    weights: str = "runs/segment/train/weights/best.pt",
    source: str = "data/test_image.jpg",
    conf: float = 0.5,
    output_dir: str = "runs/visualization",
) -> None:
    """
    Perform segmentation, calculate area percentage, and save annotated images.

    Args:
        weights: Path to trained model weights.
        source: Image path or folder.
        conf: Confidence threshold.
        output_dir: Folder to save visualized results.
    """
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(weights)
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = model.predict(source=source, conf=conf, save=False)

    for i, r in enumerate(results):
        img = r.orig_img.copy()
        h, w = img.shape[:2]
        total_pixels = h * w
        
        # Overlay canvas
        overlay = img.copy()

        if r.masks is not None:
            for j, mask in enumerate(r.masks.data):
                # 1. Convert mask to numpy and resize to image dimensions
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h))
                
                # 2. Calculate Area Percentage
                mask_area = np.sum(mask_resized > 0.5)
                percentage = (mask_area / total_pixels) * 100
                
                # 3. Get label metadata
                box = r.boxes[j]
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                class_name = model.names[cls_id]
                
                # 4. Color logic (Generate color based on class ID)
                np.random.seed(cls_id)
                color = np.random.randint(0, 255, size=(3,)).tolist()

                # 5. Apply Mask Overlay
                # Create a colored mask
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask_resized > 0.5] = color
                
                # Blend with original image
                cv2.addWeighted(colored_mask, 0.4, overlay, 0.6, 0, overlay)

                # 6. Draw HUD (Class Name, Conf, Area %)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} ({conf_score:.2f}) | {percentage:.1f}%"
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    overlay, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # Save result
        output_filename = save_path / f"viz_{i}.jpg"
        cv2.imwrite(str(output_filename), overlay)
        print(f"Saved visualization to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-seg Area Visualization")
    parser.add_argument("--weights", default="runs/segment/train/weights/best.pt")
    parser.add_argument("--source", required=True, help="Path to image or folder")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output_dir", default="runs/visualization")
    
    args = parser.parse_args()

    run_visualization(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        output_dir=args.output_dir,
    )