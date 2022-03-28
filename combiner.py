import math
import torch

class Combiner:
    def __init__(self, model, s_prev, s_cur, n = None) -> None:
        self.model = model
        self.cache = []
        self.n = math.floor(s_prev/s_cur) if n is None else n

    def combine_and_compute(self, x):
        import time
        self.cache.append(x)
        if len(self.cache) < self.n:
            return None, 0 
        input = torch.cat(self.cache, dim=2)
        self.cache = []
        start_time = time.time()
        res = self.model(input)
        time_run = time.time() - start_time
        return res, time_run