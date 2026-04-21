"""
inference.py
Run YOLOv8-seg inference on an image or folder and display/save results.

Usage (CPU-friendly — no GPU needed):
    python src/inference.py --source path/to/image.jpg
    python src/inference.py --source path/to/folder/
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def run_inference(
    weights: str = "runs/segment/train/weights/best.pt",
    source: str = "data/test_image.jpg",
    conf: float = 0.5,
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Run instance segmentation inference and visualize results.

    Args:
        weights:  Path to trained model weights (.pt file)
        source:   Image path, folder, or video path
        conf:     Confidence threshold (0.0 – 1.0)
        save:     Save annotated results to runs/segment/predict*/
        show:     Display results inline (use in Jupyter/Colab)
    """
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(weights)
    print(f"Model loaded: {weights}")

    results = model.predict(source=source, conf=conf, save=save)

    for i, r in enumerate(results):
        # Show class + confidence for each detection
        if r.boxes is not None:
            for box, cls_id, conf_score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_name = model.names[int(cls_id)]
                print(f"  Detected: {class_name} ({conf_score:.2f})")

        # Display inline (Colab / Jupyter)
        if show:
            try:
                from IPython.display import display as ipy_display
                im_array = r.plot(masks=True, boxes=True)
                im = Image.fromarray(im_array[..., ::-1])  # BGR → RGB
                ipy_display(im)
            except ImportError:
                print("IPython not available — use save=True to view results.")

    if save:
        print(f"\nResults saved to: {Path('runs/segment/predict').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-seg inference")
    parser.add_argument("--weights", default="runs/segment/train/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image or folder path")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--show", action="store_true", default=False)
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show,
    )