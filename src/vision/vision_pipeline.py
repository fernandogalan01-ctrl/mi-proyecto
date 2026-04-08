import cv2
import numpy as np
import json 
from ultralytics import YOLO
class AssistiveVisionSystem:
    def __init__(self, model_path='model/yolov8n.pt', spatial_threshold=50, ontology_path='data/ontology.json'):
        print("Inicializando Pipeline de Visión y Conocimiento")
        self.model = YOLO(model_path)
        self.threshold = spatial_threshold
        try:
            with open(ontology_path, 'r') as f:
                self.ontology = json.load(f)
            print("Ontología cargada correctamente.")
        except FileNotFoundError:
            print("No se encontró ontology.json. Se usará conocimiento básico.")
            self.ontology = {}
    def analyze_scene(self, image_path):
        knowledge_base = []
        results = self.model(image_path)
        detected_entities = []
        for r in results:
            for box in r.boxes:
                detected_entities.append({
                    'name': r.names[box.cls.item()],
                    'cx': box.xyxy[0][0].item(),
                    'cy': box.xyxy[0][1].item()
                })
        for i, obj_a in enumerate(detected_entities):
            # Basic entity recognition
            knowledge_base.append(f"object({obj_a['name']})")
            # Ontology-based reasoning
            for category, items in self.ontology.items():
                if obj_a['name'] in items:
                    knowledge_base.append(f"is_a({obj_a['name']}, {category})")
            # Spatial relationships
            for j, obj_b in enumerate(detected_entities):
                if i == j: continue   
                if obj_a['cx'] < obj_b['cx'] - self.threshold:
                    knowledge_base.append(f"left_of({obj_a['name']}, {obj_b['name']})")
                if obj_a['cy'] < obj_b['cy'] - self.threshold:
                    knowledge_base.append(f"above({obj_a['name']}, {obj_b['name']})")
                elif obj_a['cy'] > obj_b['cy'] + self.threshold:
                    knowledge_base.append(f"below({obj_a['name']}, {obj_b['name']})")
        return list(set(knowledge_base)), results[0]
