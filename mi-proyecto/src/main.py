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
    model_path = os.path.join(base_dir, 'model', 'best.pt')
    vision = AssistiveVisionSystem(model_path=model_path)
    reasoner = InferenceEngine()
    dialog = DialogManager(voice_speed=150)
    print("\n[PERCEPTION] Analyzing scene...")
    try:
        if not os.path.exists(image_path):
            print(f"Error: The image was not found at {image_path}")
            return
        facts, _ = vision.analyze_scene(image_path)
        scene_info = reasoner.get_scene_report(facts)
        voice_report = dialog.generate_full_report(scene_info)
        print("-" * 30)
        print("[OUTPUT] Generated Report:")   
        print(voice_report)
        print("-" * 30)
    except Exception as e:
        print(f"Error during system execution: {e}")
if __name__ == "__main__":
    main()

