import numpy as np
import matplotlib.pyplot as plt
import torch

# x = np.arange(1, 201)
# y = []
# for i in range(len(x)):
#     if x[i] > 130:
#         y.append(1)
#     else:
#         y.append(0)
# plt.plot(x, y)
# plt.show()

# x = torch.arange(-10., 10., step=0.01)
# plt.plot(x, torch.sigmoid(x))
# plt.show()

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
def forward(t):
    return torch.sigmoid( a * t + b )