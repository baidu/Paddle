class PickWeight:
    def __init__(self, weight: float):
        self.weight = max(0.0, min(weight, 1.0))