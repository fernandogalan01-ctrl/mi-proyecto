import cv2
import math  
from ultralytics import YOLO
class AssistiveVisionSystem:
    def __init__(self, mode="hybrid"):
        """
        mode="hybrid" -> ¡Ejecuta AMBOS modelos a la vez sin superponerse!
        mode="custom" -> Solo el modelo entrenado
        mode="coco"   -> Solo el modelo global
        """
        self.mode = mode
        # CARGAMOS LOS MODELOS NECESARIOS
        if self.mode in ["custom", "hybrid"]:
            print("Cargando modelo CUSTOM (best.pt)...")
            self.model_custom = YOLO('C:\\Users\\ferga\\mi-proyecto\\mi-proyecto-1\\mi-proyecto\\model\\best.pt')
            self.clases_custom = [0, 1]  
        if self.mode in ["coco", "hybrid"]:
            print("Cargando modelo COCO (yolov8n.pt)...")
            self.model_coco = YOLO('C:\\Users\\ferga\\mi-proyecto\\mi-proyecto-1\\mi-proyecto\\model\\yolov8n.pt')
            # 56:silla, 57:sofa, 60:mesa, 61:inodoro, 62:tv, 72:nevera, 79:lavabo 
            self.clases_coco = [56, 57, 60, 61, 62, 72, 79] 
            self.traductor_coco = {
                "chair": "chair", "couch": "couch","dining table": "table", "toilet": "toilet", "tv": "tv", 
                "refrigerator": "refrigerator", "sink": "sink"
            }
    def analyze_scene(self, image_path):
        raw_img = cv2.imread(image_path)
        if raw_img is None: 
            return [], None 
        img_dibujada = raw_img.copy() 
        facts_custom = []
        facts_coco = [] 
        if self.mode in ["custom", "hybrid"]:
            res_custom = self.model_custom(raw_img, conf=0.60, iou=0.40, classes=self.clases_custom, verbose=False) 
            if len(res_custom) > 0 and res_custom[0].boxes:
                for box in res_custom[0].boxes:
                    name = self.model_custom.names[int(box.cls[0])]
                    confianza = float(box.conf[0]) 
                    coords = box.xyxy[0].tolist()
                    cx = (coords[0] + coords[2]) / 2  
                    cy_base = coords[3]               
                    facts_custom.append({'name': name, 'conf': confianza, 'x': cx, 'y': cy_base, 'coords': coords})
        if self.mode in ["coco", "hybrid"]:
            res_coco = self.model_coco(raw_img, conf=0.45, iou=0.90, classes=self.clases_coco, verbose=False)
            if len(res_coco) > 0 and res_coco[0].boxes:
                for box in res_coco[0].boxes:
                    raw_name = self.model_coco.names[int(box.cls[0])]
                    if raw_name in self.traductor_coco:
                        final_name = self.traductor_coco[raw_name]
                        confianza = float(box.conf[0]) 
                        coords = box.xyxy[0].tolist()
                        cx = (coords[0] + coords[2]) / 2  
                        cy_base = coords[3]               
                        facts_coco.append({'name': final_name, 'conf': confianza, 'x': cx, 'y': cy_base, 'coords': coords})
        spatial_facts = []
        for f_coco in facts_coco:
            spatial_facts.append(f_coco)
            c = f_coco['coords']
            etiqueta = f"{f_coco['name']} {f_coco['conf']:.2f}"
            cv2.rectangle(img_dibujada, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
            cv2.putText(img_dibujada, etiqueta, (int(c[0]), int(c[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        for f_cust in facts_custom:
            es_duplicado = False
            for f_coco in facts_coco:
                if f_coco['name'] == 'toilet':
                    distancia = math.hypot(f_cust['x'] - f_coco['x'], f_cust['y'] - f_coco['y'])
                    if distancia < 150: 
                        es_duplicado = True
                        print(f"Árbitro: Ignorando falso '{f_cust['name']}' porque es un 'toilet'.")
                        break
            if not es_duplicado:
                spatial_facts.append(f_cust)
                c = f_cust['coords']
                etiqueta = f"{f_cust['name']} {f_cust['conf']:.2f}"
                cv2.rectangle(img_dibujada, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 2)
                cv2.putText(img_dibujada, etiqueta, (int(c[0]), int(c[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        return spatial_facts, img_dibujada