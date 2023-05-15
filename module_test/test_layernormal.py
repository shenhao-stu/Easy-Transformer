import numpy as np

class LayerNormalization:
    def __init__(self, gamma, beta, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.mean = None
        self.var = None
        self.x_hat = None

    def forward(self, x):
        # 计算均值和方差
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)

        # 计算标准化后的输入
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)

        # 计算输出
        out = self.gamma * self.x_hat + self.beta

        return out

    def backward(self, dout):
        # 计算输入的维度
        _, D = dout.shape

        # 计算 dL/dx_hat, dL/dvar, dL/dmean
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.x_hat - self.mean) * (-0.5) * np.power(self.var + self.eps, -1.5), axis=1, keepdims=True)
        dmean = np.sum(dx_hat * (-1 / np.sqrt(self.var + self.eps)), axis=1, keepdims=True) + dvar * np.mean(-2 * (self.x_hat - self.mean), axis=1, keepdims=True)

        # 计算 dL/dx
        dx = dx_hat * (1 / np.sqrt(self.var + self.eps)) + dvar * (2 * (self.x_hat - self.mean) / D) + dmean / D

        # 计算 dL/dgamma, dL/dbeta
        dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        return dx, dgamma, dbeta
    
# 测试代码
x = np.array([[1, 2, 3], [4, 5, 6]])
gamma = np.array([1, 1, 1])
beta = np.array([0, 0, 0])

# 前向传播
ln = LayerNormalization(gamma, beta)
y = ln.forward(x)
print("y = ", y)

# 反向传播
dy = np.zeros_like(y)
dy[1, 0] = 1
dx, dgamma, dbeta = ln.backward(dy)
print("dx = ", dx)
print("dgamma = ", dgamma)
print("dbeta = ", dbeta)