import math
import torch

class Combiner:
    def __init__(self, model, s_prev, s_cur, n = None) -> None:
        self.model = model
        self.cache = []
        self.n = math.floor(s_prev/s_cur) if n is None else n

    def combine_and_compute(self, x):
        self.cache.append(x)
        if len(self.cache) < self.n:
            return None
        input = torch.cat(self.cache, dim=2)
        self.cache = []
        return self.model(input)