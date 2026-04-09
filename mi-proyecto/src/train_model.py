from ultralytics import YOLO
import os
def train_custom_model():
    """
    Fine-tunes the YOLOv8 model using the custom dataset configuration.
    """
    print(" Starting Model Fine-Tuning ")
    # Load the pre-trained YOLOv8 model
    model = YOLO('model/yolov8n.pt')
    # Train the model on the custom dataset
    results = model.train(
        data='data/custom_ds.yaml', 
        epochs=50, 
        imgsz=640, 
        device='cpu',
        workers=0,
        exist_ok=True,
        name='train'
    )
    print("Training Complete")
    print("Your new model is saved in: runs/detect/train/weights/best.pt")
if __name__ == "__main__":
    train_custom_model()