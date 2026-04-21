"""
train.py
Train YOLOv8-seg on the Indian Currency Segmentation dataset.

Run on Google Colab (GPU required):
    python src/train.py

Results saved to: runs/segment/train*/
"""

from ultralytics import YOLO


def train(
    data_yaml: str = "dataset/data.yaml",
    model_size: str = "n",          # n=nano, s=small, m=medium
    epochs: int = 100,
    imgsz: int = 640,
    patience: int = 20,             # early stopping
    device: int = 0,                # 0 = first GPU; 'cpu' for CPU
) -> None:
    """
    Train YOLOv8 instance segmentation model.

    Args:
        data_yaml:   Path to data.yaml config file
        model_size:  YOLOv8 variant — 'n', 's', 'm', 'l', 'x'
        epochs:      Maximum training epochs
        imgsz:       Input image size (pixels)
        patience:    Early stopping patience (epochs without improvement)
        device:      Training device (GPU index or 'cpu')
    """
    model_name = f"yolov8{model_size}-seg.pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        device=device,
    )

    print("\nTraining complete.")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    train()