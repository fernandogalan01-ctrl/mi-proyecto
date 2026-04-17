import os
from ultralytics import YOLO
def train_ultimate_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.abspath(os.path.join(base_dir, 'data', 'custom_ds.yaml'))
    model_path = os.path.join(base_dir, 'model', 'yolov8n.pt') 
    print("Iniciando entrenamiento...")
    model = YOLO(model_path)
    model.train(
        data=yaml_path,
        epochs=150,          
        patience=30,         
        imgsz=640,
        device='cpu',        
        batch=8,             
        mosaic=1.0,          
        degrees=10.0,        
        fliplr=0.5,          
        mixup=0.1,           
        hsv_h=0.015,         
        hsv_s=0.5,           
        hsv_v=0.2,           
        name='guia_espacial_definitivo',
        exist_ok=True
    )
    print("\n¡Entrenamiento listo!")
if __name__ == "__main__":
    train_ultimate_model()