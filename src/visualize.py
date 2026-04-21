import cv2
import numpy as np
from ultralytics import YOLO

def visualize_currency(image_path, model_path, output_path='result.jpg'):
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)[0]
    img = results.orig_img
    h, w = img.shape[:2]
    total_area = h * w
    
    # Setup colors for classes
    classes = results.names
    colors = {i: np.random.randint(0, 255, size=3).tolist() for i in classes}
    
    # Create overlay for masks
    overlay = img.copy()
    
    # Process detections
    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            # 1. Get Mask & calculate area
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h))
            mask_area = np.sum(mask_resized > 0.5)
            percentage = (mask_area / total_area) * 100
            
            # 2. Get metadata
            box = results.boxes[i]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = classes[cls_id]
            color = colors[cls_id]
            
            # 3. Apply color mask
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[mask_resized > 0.5] = color
            cv2.addWeighted(colored_mask, 0.4, overlay, 0.6, 0, overlay)
            
            # 4. Draw Label + Percentage
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{name} {conf:.2f} | {percentage:.1f}% Area"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, overlay)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_currency('test_img.jpg', '/content/runs/segment/train-5/weights/best.pt')