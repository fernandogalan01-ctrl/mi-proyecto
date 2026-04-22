import whisper
import pyttsx3 
class DialogManager:
    def __init__(self):
        print("Cargando modelo de voz Whisper (esto puede tardar unos segundos)...")
        self.stt_model = whisper.load_model("base")
        self.intents_db = {
            "describe_scene": ["what do you see", "describe", "what is here", "scan"],
            "check_safety": ["is it safe", "hazards", "danger", "stairs", "obstacles", "clear", "path"],
            "find_object": ["where is", "find", "locate", "search for"],
            "check_location": ["where am i", "which room", "identify room", "location"]
        }
        self.templates = {
            "alerts": {
                "hazard": "Caution: {item} detected. Please proceed carefully.",
                "safe": "The immediate path appears to be clear of hazards."
            },
            "location": {
                "found": "Based on the objects, you are in the {room}.",
                "unknown": "I cannot identify the specific room yet. Keep scanning."
            }
        }
        self.knowledge_graph = {
            "bathroom": ["toilet", "bathtub", "shower", "sink", "brush", "towels", "mirror"],
            "kitchen": ["refrigerator", "countertop", "garbage"],
            "living_room": ["couch", "tv", "tv stand", "chair", "table", "plant", "window"],
            "bedroom": ["bed", "door", "plant"],
            "danger_zones": ["stairs", "garbage"]
        }
    def speak(self, text):
        print(f"Asistente dice: {text}")
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) 
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    def listen_and_transcribe(self, audio_file_path):
        print("Procesando audio de pregunta...")
        result = self.stt_model.transcribe(audio_file_path)
        transcript = result["text"].strip().lower()
        return transcript
    def _get_intent(self, question):
        for intent, patterns in self.intents_db.items():
            if any(pattern in question for pattern in patterns):
                return intent
        return "find_object" 
    def answer_question_about_scene(self, facts, scene_description, user_question):
        question = user_question.lower()
        intent = self._get_intent(question)
        detected_objects = [fact['name'] for fact in facts]
        llm_answer = ""
        if intent == "describe_scene":
            llm_answer = scene_description
        elif intent == "check_safety":
            hazards = [obj for obj in detected_objects if obj in self.knowledge_graph["danger_zones"]]
            if hazards:
                llm_answer = self.templates["alerts"]["hazard"].format(item=hazards[0])
            else:
                llm_answer = self.templates["alerts"]["safe"]
        elif intent == "check_location":
            room_scores = {"bathroom": 0, "kitchen": 0, "living_room": 0, "bedroom": 0}
            for obj in detected_objects:
                for room, items in self.knowledge_graph.items():
                    if room != "danger_zones" and obj in items:
                        room_scores[room] += 1
            best_room = max(room_scores, key=room_scores.get) 
            if room_scores[best_room] > 0:
                room_name = best_room.replace("_", " ") 
                llm_answer = self.templates["location"]["found"].format(room=room_name)
            else:
                llm_answer = self.templates["location"]["unknown"]
        elif intent == "find_object":
            all_known_items = sum([items for room, items in self.knowledge_graph.items()], [])
            target = next((item for item in all_known_items if item in question), None)
            if target:
                if target in detected_objects:
                    llm_answer = f"Yes, I detect a {target} in front of you."
                else:
                    llm_answer = f"I'm sorry, I don't detect any {target} right now."
            else:
                llm_answer = "I'm not sure what object you are looking for."
        self.speak(llm_answer)
        return llm_answer