import json

class InferenceEngine:
    def __init__(self):
        self.display_names = {
            "bathtub": "bathtub",
            "bed": "bed",
            "brush": "brush",
            "chair": "chair",
            "closed door": "closed door",
            "couch": "couch",
            "countertop": "countertop",
            "garbage": "trash can",
            "plant": "plant",
            "refrigerator": "refrigerator",
            "shower": "shower",
            "sink": "sink",
            "stairs": "stairs",
            "table": "table",
            "toilet": "toilet",
            "towels": "towels",
            "tv": "television",
            "tv stand": "TV stand"
        }
    def get_scene_report(self, knowledge_base):
        """
        Processes the knowledge base and generates a structured English report.
        """
        data = {
            "objects": [],
            "relations": [],
            "hazards": []
        }
        for fact in knowledge_base:
            if fact.startswith("object("):
                obj_name = fact.split("(")[1].replace(")", "")
                readable_name = self.display_names.get(obj_name, obj_name)
                data["objects"].append(readable_name)
            elif any(rel in fact for rel in ["left_of", "behind", "in_front_of"]):
                rel_type = fact.split("(")[0]
                args = fact.split("(")[1].replace(")", "").split(", ")
                obj_a = self.display_names.get(args[0], args[0])
                obj_b = self.display_names.get(args[1], args[1])
                rel_map = {
                    "left_of": "to the left of",
                    "behind": "further back than",
                    "in_front_of": "closer than"
                }
                label = rel_map.get(rel_type, "near")
                data["relations"].append(f"the {obj_a} is {label} the {obj_b}")
        raw_objects = [f.split("(")[1].replace(")", "") for f in knowledge_base if f.startswith("object(")]
        unique_raw = set(raw_objects)
        if "stairs" in unique_raw:
            data["hazards"].append("CAUTION: Stairs detected ahead. Use your cane to locate the first step.")
        if "closed door" in unique_raw:
            data["hazards"].append("There is a closed door in front of you.")
        if "garbage" in unique_raw:
            data["hazards"].append("Watch out, there is a trash can on the floor that might be in your way.")
        if any(item in unique_raw for item in ["toilet", "bathtub", "shower", "sink"]):
            data["hazards"].append("You are currently in the bathroom area.")
        if "refrigerator" in unique_raw or "countertop" in unique_raw:
            data["hazards"].append("You are in the kitchen or near a work surface.")
        data["objects"] = list(set(data["objects"]))
        return data
    def generate_natural_language(self, scene_data):
        """
        Converts structured data into a natural English narrative for TTS.
        """
        if not scene_data["objects"]:
            return "I don't see any clear objects right now. Try scanning the area again."
        num = len(scene_data["objects"])
        intro = f"I detect {num} items: {', '.join(scene_data['objects'])}. "
        layout = ""
        if scene_data["relations"]:
            layout = "Regarding the layout: " + ". ".join(scene_data["relations"][:2]) + ". "
        safety = ""
        if scene_data["hazards"]:
            safety = "Important notices: " + " ".join(scene_data["hazards"])
        else:
            safety = "The path ahead appears to be clear."
        return f"{intro}{layout}{safety}"