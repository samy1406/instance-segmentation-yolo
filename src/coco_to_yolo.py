import json
import os

def coco_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create mapping: COCO category_id -> YOLO class_index
    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    # Create mapping: image_id -> image_info (width, height, file_name)
    images = {img['id']: img for img in data['images']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)

    for img_id, anns in annotations_by_image.items():
        img_info = images[img_id]
        w, h = img_info['width'], img_info['height']
        filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            for ann in anns:
                cat_id = categories[ann['category_id']]
                # COCO polygons are [x1, y1, x2, y2, ...]. 
                # Assumes single polygon list structure.
                segmentation = ann['segmentation'][0]
                
                # Normalize coordinates
                normalized_coords = []
                for i in range(0, len(segmentation), 2):
                    normalized_coords.append(str(segmentation[i] / w))
                    normalized_coords.append(str(segmentation[i+1] / h))
                
                line = f"{cat_id} " + " ".join(normalized_coords)
                f.write(line + "\n")

if __name__ == "__main__":
    coco_to_yolo('_annotations.coco.json', 'labels')