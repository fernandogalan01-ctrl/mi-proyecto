class DialogManager:
    def __init__(self):
        pass

    def generate_full_report(self, scene_data):
        if not scene_data["objects"]:
            return "The camera doesn't see any objects clearly. Please adjust your position."

        intro = f"I see {len(scene_data['objects'])} items: {', '.join(scene_data['objects'])}. "
        
        layout = ""
        if scene_data["relations"]:
            # We filter to show only 2 clear relations so we don't overwhelm the user
            layout = "Layout: " + ". ".join(scene_data["relations"][:2]) + ". "

        safety = ""
        if scene_data["hazards"]:
            safety = "Warning: " + " ".join(scene_data["hazards"])
        else:
            safety = "The area looks safe."

        return f"{intro}{layout}{safety}"