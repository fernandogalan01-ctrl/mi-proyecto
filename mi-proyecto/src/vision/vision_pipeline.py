import cv2
import numpy as np
import json
from ultralytics import YOLO
class AssistiveVisionSystem:
    def __init__(self, model_path='model/yolov8n.pt', spatial_threshold=60, ontology_path='data/ontology.json'):
        print("Initializing Assistive Vision System")
        self.model = YOLO(model_path)
        self.threshold = spatial_threshold
        try:
            with open(ontology_path, 'r', encoding='utf-8') as f:
                self.ontology = json.load(f)
        except FileNotFoundError:
            self.ontology = {}
    def _preprocess(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    def analyze_scene(self, image_path):
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return [], None     
        processed_img = self._preprocess(raw_img)
        results = self.model(processed_img, verbose=False)
        detected_entities = []
        if len(results) > 0:
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                coords = box.xyxy[0].tolist()
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                detected_entities.append({'id': i, 'name': name, 'cx': cx, 'cy': cy})
        knowledge_base = []
        for obj_a in detected_entities:
            name_a = obj_a['name']
            knowledge_base.append(f"object({name_a})") 
            for category, items in self.ontology.items():
                if name_a in items:
                    knowledge_base.append(f"is_a({name_a}, {category})")
            for obj_b in detected_entities:
                if obj_a['id'] == obj_b['id']: continue
                name_b = obj_b['name']
                if obj_a['cx'] < obj_b['cx'] - self.threshold:
                    knowledge_base.append(f"left_of({name_a}, {name_b})")
                elif obj_a['cx'] > obj_b['cx'] + self.threshold:
                    knowledge_base.append(f"right_of({name_a}, {name_b})")
                if obj_a['cy'] > obj_b['cy'] + self.threshold:
                    knowledge_base.append(f"in_front_of({name_a}, {name_b})")
        return list(set(knowledge_base)), results[0]