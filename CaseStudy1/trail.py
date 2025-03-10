import torch
a=torch.tensor([[
    [1],[2],[3],[4],[5]
]])
b=torch.reshape(a,(5,))
print(b)
print(a.shape,b.shape)