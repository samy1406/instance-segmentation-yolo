"""
evaluate.py
Evaluate a trained YOLOv8-seg model and print per-class metrics.

Usage:
    python src/evaluate.py
"""

from ultralytics import YOLO


def evaluate(
    weights: str = "runs/segment/train/weights/best.pt",
    data_yaml: str = "dataset/data.yaml",
    split: str = "val",             # 'val' or 'test'
    imgsz: int = 640,
    device: str = "cpu",            # CPU-friendly for Codespace
) -> None:
    """
    Evaluate model and print detection + segmentation metrics.

    Key metrics:
        mAP50    — mean Average Precision at IoU 0.50
        mAP50-95 — mean AP averaged over IoU thresholds 0.50:0.95 (stricter)
        Precision — of all predicted positives, how many are correct?
        Recall    — of all actual positives, how many were found?

    Args:
        weights:   Path to trained model weights
        data_yaml: Path to dataset config yaml
        split:     Dataset split to evaluate on ('val' or 'test')
        imgsz:     Image size used during evaluation
        device:    'cpu' or GPU index (e.g. 0)
    """
    model = YOLO(weights)

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=device,
    )

    print("\n=== Detection Metrics (Bounding Box) ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    print("\n=== Segmentation Metrics (Mask) ===")
    print(f"  mAP50:    {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")

    print("\n=== Per-Class Segmentation mAP50 ===")
    class_names = model.names
    for i, (name, ap) in enumerate(zip(class_names.values(), metrics.seg.ap)):
        print(f"  {name:10s}: {ap:.4f}")


if __name__ == "__main__":
    evaluate()