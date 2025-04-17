



class ObjectDector3D:
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.load_model()