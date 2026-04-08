class InferenceEngine:
    def __init__(self):
        pass
    def get_scene_report(self, knowledge_base):
        data = {"objects": [], "relations": [], "hazards": []}
        for fact in knowledge_base:
            if fact.startswith("object("):
                obj = fact.split("(")[1].replace(")", "")
                data["objects"].append(obj)
            elif any(rel in fact for rel in ["left_of", "behind", "in_front_of"]):
                rel_type = fact.split("(")[0]
                parts = fact.split("(")[1].replace(")", "").split(", ")
                rel_map = {
                    "left_of": "to the left of", 
                    "behind": "further back than", 
                    "in_front_of": "closer than"
                }
                if rel_type in rel_map:
                    data["relations"].append(f"the {parts[0]} is {rel_map[rel_type]} the {parts[1]}")
        for obj in data["objects"]:
            if obj == "cup":
                data["hazards"].append("The cup is near other items, be careful not to spill it.")
        data["objects"] = list(set(data["objects"]))
        return data