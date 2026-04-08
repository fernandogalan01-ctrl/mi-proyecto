import os
import sys
import pyttsx3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.vision_pipeline import AssistiveVisionSystem 
from reasoning.inference_engine import InferenceEngine
from nlp.dialog_manager import DialogManager
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, 'test_images', 'mesa.jpg')
    model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
    vision = AssistiveVisionSystem(model_path='model/best.pt')
    reasoner = InferenceEngine()
    dialog = DialogManager()
    print("\n[PERCEPTION] Analyzing scene...")
    try:
        facts, _ = vision.analyze_scene(image_path)
        scene_info = reasoner.get_scene_report(facts)
        voice_report = dialog.generate_full_report(scene_info)
        print("[OUTPUT] Generated Report:")   
        print(voice_report)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) 
        engine.say(voice_report)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during system execution: {e}")
if __name__ == "__main__":
    main() 
 

