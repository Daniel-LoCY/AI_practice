import torch

# x = torch.tensor([3.0])  # 蘋果單價
# y = torch.tensor([18.0]) # 我們的預算
# a = torch.tensor([1.0], requires_grad=True)  # 追蹤導數
# print('grad:', a.grad)
# loss = y - (a * x)  # loss function ( 中文稱 損失函數 )
# loss.backward()
# print('grad:', a.grad)

# for _ in range(100):
#     a.grad.zero_()
#     loss = y - (a * x)
#     loss.backward()
#     with torch.no_grad():
#         a -= a.grad * 0.01 * loss
# print('a:', a)
# print('loss:', (y - (a * x)))
# print('result:', (a * x))

# x = torch.tensor([3.0, 5.0, 6.0,])   # 不同種蘋果售價
# y = torch.tensor([18.0, 18.0, 18.0]) # 我們的預算
# a = torch.tensor([1.0, 1.0, 1.0], requires_grad=True) # 先假設都只能買一顆
# loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.SGD([a], lr=0.01)
# for _ in range(1000):
#     optimizer.zero_grad()
#     loss = loss_func(y, a * x)
#     loss.backward()
#     optimizer.step()
#     print(a*x)
# print('a:', a)

from sklearn.datasets import make_regression
np_x, np_y = make_regression(n_samples=500, n_features=10)
x = torch.from_numpy(np_x).float()
y = torch.from_numpy(np_y).float()

w = torch.randn(10, requires_grad=True)
b = torch.randn(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.01)
def model(x):
    return x @ w + b
predict_y = model(x)
print(predict_y)
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(y)
plt.plot(predict_y.detach().numpy())
plt.subplot(1, 2, 2)
plt.scatter(y.detach().numpy(), predict_y.detach().numpy())
plt.show()