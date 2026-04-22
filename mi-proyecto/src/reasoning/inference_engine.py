import json
class InferenceEngine:
    def __init__(self):
        # Diccionario para nombres 
        self.display_names = {
            "bathtub": "bathtub", "bed": "bed", "brush": "toothbrush", "chair": "chair", "couch": "sofa",
            "countertop": "countertop", "garbage": "trash can", "plant": "plant",
            "refrigerator": "fridge", "shower": "shower", "sink": "sink",
            "stairs": "stairs", "table": "table", "toilet": "toilet",
            "towels": "towels", "tv": "television", 
            "door": "door", "window": "window", 
        }
    def get_scene_report(self, knowledge_base):
        if not knowledge_base:
            return ["I don't see any clear objects. Try scanning the area again."]
        parsed_objects = []
        for fact in knowledge_base:
            if isinstance(fact, dict):
                 parsed_objects.append({'name': fact['name'], 'x': fact['x'], 'y': fact['y']})
            elif fact.startswith("spatial_info("):
                parts = fact.replace("spatial_info(", "").replace(")", "").split(",")
                parsed_objects.append({'name': parts[0], 'x': float(parts[1]), 'y': float(parts[2])})
        if not parsed_objects:
             return ["I don't see any clear objects."]
        sorted_objs = sorted(parsed_objects, key=lambda o: o['x'], reverse=True)
        unique_names = {obj['name'] for obj in sorted_objs}
        context = ""
        if any(item in unique_names for item in ["toilet", "bathtub", "shower", "sink"]):
            context = "You are in the bathroom. "
        elif "refrigerator" in unique_names or "countertop" in unique_names or "oven" in unique_names:
            context = "You are in the kitchen area. "
        narrativa = f"{context}Scanning from right to left. I detect {len(sorted_objs)} items. "
        for i, obj in enumerate(sorted_objs):
            name = self.display_names.get(obj['name'], obj['name'])
            article = "" if name.endswith('s') else "a "
            if obj['y'] > 350: depth = "very close to you"
            elif obj['y'] > 150: depth = "a few steps away"
            else: depth = "further back in the room"
            if i == 0:
                narrativa += f"On your far right, there is {article}{name} {depth}. "
            else:
                prev_name = self.display_names.get(sorted_objs[i-1]['name'], sorted_objs[i-1]['name'])
                narrativa += f"Just to the left of the {prev_name}, you will find {article}{name} {depth}. "
        hazards = []
        if "stairs" in unique_names: hazards.append("CAUTION: Stairs detected.")
        if "closed door" in unique_names: hazards.append("Note: there is a closed door ahead.")
        safety = " " + " ".join(hazards) if hazards else " The path ahead appears clear."
        return [narrativa + safety]