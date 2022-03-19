import torch

class Spliter:
    def __init__(self, model, input_size, s, n) -> None:
        '''
        model:
        input_size: shape of input data, size should be like [b, c, t] 
        s: stride size, depends on the model
        n: number of splited data
        '''
        # input_size // n = data length in each block which should more than stride.
        assert n <= input_size // s
        self.model = model
        self.s = s
        self.n = n
        self.size = input_size // n

    def compute_once(self, x):
        b, c, t = x.shape
        left = t % self.s
        if left != 0:
            zeros_pad = torch.zeros([b, c, self.s - left], dtype=torch.float32).cuda()
            x = torch.cat([x, zeros_pad], dim=2)
        return self.model(x)

    def split_and_compute(self, x):
        if self.n == 1:
            return self.model(x)
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        split_list = torch.split(x, self.size, dim=2)
        ans = []
        for sub_input in split_list:
            out = self.compute_once(sub_input)
            '''
            此处可以添加发送的逻辑
            '''
            ans.append(out)
        return ans