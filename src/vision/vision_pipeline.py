import cv2
import numpy as np
from ultralytics import YOLO

class AssistiveVisionSystem:
    def __init__(self, model_path='yolov8n.pt', spatial_threshold=50):
        """
        Inicializa el sistema de visión asistencial.
        :param model_path: Ruta del modelo YOLO preentrenado.
        :param spatial_threshold: Píxeles de tolerancia para evitar ambigüedad espacial.
        """
        print("--- Inicializando Pipeline de Visión ---")
        self.model = YOLO(model_path)
        self.threshold = spatial_threshold

    def _preprocess(self, frame):
        """
        Etapa de Robustez: Mejora la imagen antes de la inferencia.
        Aplica CLAHE para ecualizar el histograma y mejorar el contraste en sombras.
        """
        # Convertir a espacio de color LAB para procesar solo la luminancia
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Recombinar canales y volver a BGR
        limg = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    def analyze_scene(self, image_path):
        """
        Procesa la imagen, detecta objetos y construye la base de conocimiento.
        """
        # 1. Carga y Preprocesamiento
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise FileNotFoundError(f"No se pudo encontrar la imagen: {image_path}")
            
        processed_img = self._preprocess(raw_img)
        
        # 2. Inferencia (Detección de objetos)
        results = self.model(processed_img)
        detected_entities = []

        for r in results:
            for box in r.boxes:
                # Extraer datos técnicos de la Bounding Box
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                # Calcular Centroide
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                
                detected_entities.append({
                    'name': name,
                    'cx': cx,
                    'cy': cy,
                    'conf': conf
                })

        # 3. Razonamiento Espacial (Generación de Predicados)
        knowledge_base = []
        
        for i, obj_a in enumerate(detected_entities):
            # Añadir existencia a la base de conocimiento
            knowledge_base.append(f"object({obj_a['name']})")
            
            for j, obj_b in enumerate(detected_entities):
                if i == j: continue # No compararse con uno mismo
                
                # Relación Horizontal: Izquierda / Derecha
                # Usamos un umbral (tau) para mayor estabilidad
                if obj_a['cx'] < obj_b['cx'] - self.threshold:
                    knowledge_base.append(f"left_of({obj_a['name']}, {obj_b['name']})")
                
                # Relación Vertical: Encima / Debajo
                # Nota: En imagen, Y crece hacia abajo
                if obj_a['cy'] < obj_b['cy'] - self.threshold:
                    knowledge_base.append(f"above({obj_a['name']}, {obj_b['name']})")
                elif obj_a['cy'] > obj_b['cy'] + self.threshold:
                    knowledge_base.append(f"below({obj_a['name']}, {obj_b['name']})")

        return list(set(knowledge_base)), results[0]

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    # Instanciamos el sistema
    vision_system = AssistiveVisionSystem(spatial_threshold=60)
    
    # Analizamos la imagen
    img_test = 'mesa.jpg' # Asegúrate de que este archivo exista
    facts, result_img = vision_system.analyze_scene(img_test)
    
    print("\n--- ESTRUCTURA LÓGICA DE LA ESCENA ---")
    for f in sorted(facts):
        print(f"✔️ {f}")
        
    # Guardar el resultado visual para la defensa
    result_img.save('pipeline_output.jpg')
    print("\nImagen de detección guardada como 'pipeline_output.jpg'")