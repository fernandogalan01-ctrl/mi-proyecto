import pyttsx3
class DialogManager:
    def __init__(self, voice_speed=150):
        # Initialize the Text-to-Speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', voice_speed)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "EN-US" in voice.id.upper() or "ENGLISH" in voice.name.upper():
                self.engine.setProperty('voice', voice.id)
                break
    def generate_full_report(self, scene_data):
        """
        Formats the scene data into a polished narrative.
        """
        if not scene_data["objects"]:
            msg = "The camera doesn't see any objects clearly. Please scan your surroundings slowly."
            self.speak(msg)
            return msg
        intro = f"Scene analysis complete. I detect {len(scene_data['objects'])} items: {', '.join(scene_data['objects'])}. "
        layout = ""
        if scene_data["relations"]:
            layout = "Regarding the layout: " + ". ".join(scene_data["relations"][:2]) + ". "
        safety = ""
        if scene_data["hazards"]:
            safety = "Important notices: " + " ".join(scene_data["hazards"])
        else:
            safety = "The path ahead appears to be clear."
        full_report = f"{intro}{layout}{safety}"
        self.speak(full_report)
        return full_report
    def speak(self, text):
        """
        Converts text to audible speech.
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")