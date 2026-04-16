import cv2
from ultralytics import YOLO
class AssistiveVisionSystem:
    def __init__(self, model_path='model/best.pt'):
        self.model = YOLO(model_path)
    def analyze_scene(self, image_path):
        raw_img = cv2.imread(image_path)
        if raw_img is None: return [], None 
        processed_img = raw_img 
        results = self.model(processed_img, conf=0.32, iou=0.45, verbose=False)
        spatial_facts = []
        if len(results) > 0 and results[0].boxes:
            for box in results[0].boxes:
                name = self.model.names[int(box.cls[0])]
                coords = box.xyxy[0].tolist()
                cx = (coords[0] + coords[2]) / 2  
                cy_base = coords[3]               
                spatial_facts.append({'name': name, 'x': cx, 'y': cy_base})
        return spatial_facts, results[0]