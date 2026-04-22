import os
import sys
import cv2
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from src.vision.vision_pipeline import AssistiveVisionSystem
from src.reasoning.inference_engine import InferenceEngine
from src.nlp.dialog_manager import DialogManager
def main():
    vision = AssistiveVisionSystem(mode="hybrid") 
    print("Iniciando Sistema de Asistencia Multimodal...")
    engine = InferenceEngine()
    dialog = DialogManager()
    test_path = os.path.join(base_dir, "my-dataset", "test", "images")  
    if not os.path.exists(test_path): 
        print(f"Error: Ruta de test no encontrada en {test_path}")
        return 
    images = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.png'))]
    for img in images:
        path = os.path.join(test_path, img)
        print(f"\nAnalizando escena: {img}")
        facts, res = vision.analyze_scene(path)
        if not facts:
            print("El modelo no detectó objetos seguros con suficiente confianza.")
            continue
        report = engine.get_scene_report(facts)  
        scene_description = report[0]
        print(f"Descripción generada: {scene_description}")
        dialog.speak(scene_description)
        cv2.imshow("Asistente Visual (Modo Hibrido)", res)
        cv2.waitKey(100)
        test_audio_path = os.path.join(base_dir, 'my-dataset', 'test', 'audio', 'pregunta_prueba.wav')
        if os.path.exists(test_audio_path):
             user_question = dialog.listen_and_transcribe(test_audio_path)
             print(f" Usuario preguntó: {user_question}")
             dialog.answer_question_about_scene(facts, scene_description, user_question)
        else:
             print(f"AVISO: No se encontró audio del usuario en: {test_audio_path}")
        print("Presiona cualquier tecla en la ventana de la imagen para continuar...")
        cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print("Demostración finalizada.")
if __name__ == "__main__":
    main()