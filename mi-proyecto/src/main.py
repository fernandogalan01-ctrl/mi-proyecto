import os
import sys
import cv2
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from src.vision.vision_pipeline import AssistiveVisionSystem
from src.reasoning.inference_engine import InferenceEngine
from src.nlp.dialog_manager import DialogManager
def main():
    custom_model = os.path.join(base_dir, 'model', 'best.pt')
    model_path = custom_model if os.path.exists(custom_model) else os.path.join(base_dir, 'model', 'yolov8n.pt')
    vision = AssistiveVisionSystem(model_path=model_path)
    engine = InferenceEngine()
    dialog = DialogManager()
    test_path = os.path.join(base_dir, "my-dataset", "test", "images")
    if not os.path.exists(test_path): 
        print("Ruta de test no encontrada")
        return 
    images = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.png'))]
    for img in images:
        path = os.path.join(test_path, img)
        facts, res = vision.analyze_scene(path)
        report = engine.get_scene_report(facts)  
        scene_description = report[0]
        dialog.speak(scene_description)
        cv2.imshow("Asistente Visual", res.plot())
        cv2.waitKey(100)
        test_audio_path = os.path.join(base_dir, 'my-dataset', 'test', 'audio', 'pregunta_prueba.wav')
        if os.path.exists(test_audio_path):
             user_question = dialog.listen_and_transcribe(test_audio_path)
             print(f"Usuario preguntó: {user_question}")
             dialog.answer_question_about_scene(scene_description, user_question)
        else:
             print(f"AVISO: No se encontró el audio de prueba en: {test_audio_path}")
        cv2.waitKey(0) 
if __name__ == "__main__":
    main()