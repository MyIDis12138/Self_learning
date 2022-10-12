import torch
import numpy as np

a = torch.randn(3,4)
print(a)
print(a.shape)
b = np.array(a)
print(b)
print(b.shape)