import cupy as cp

class LayerNormalization():
    # 2-dimension
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=cp.float32):
        self.layer_name = "layernorm"
        self.optimizer = optimizer
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_type = data_type
        self.gamma = None
        self.beta = None
        self.mean = None
        self.var = None
        self.x_hat = None
        self.grad_gamma = None
        self.grad_beta = None
        self.init_weights()
        self.zero_grad()
        self.register()

    def init_weights(self):
        self.gamma = cp.ones(self.normalized_shape).astype(self.data_type)
        self.beta = cp.zeros(self.normalized_shape).astype(self.data_type)

    def zero_grad(self):
        self.grad_gamma = cp.zeros_like(self.gamma)
        self.grad_beta = cp.zeros_like(self.beta)

    def register(self):
        self.layer_id = self.optimizer.count_layers(self.layer_name) // 2
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.register_params("{}.gamma".format(self.register_name), self.gamma)
        self.optimizer.register_params("{}.beta".format(self.register_name), self.beta)

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)
        y = self.gamma * self.x_hat + self.beta
        return y

    # def backward(self, grad):
    #     x = self.x
    #     x_mean = x.mean(axis=-1, keepdims=True)
    #     x_var = x.var(axis=-1, keepdims=True)
    #     lnorm = (x - x_mean) / cp.sqrt(x_var + self.eps)
    #     batch_size, seq_len, d = x.shape
    #     self.grad_beta += grad.sum(axis=tuple(range(grad.ndim - 1)))
    #     self.grad_gamma += cp.sum(grad * lnorm, axis=tuple(range(grad.ndim - 1)))
    #     grad_lnorm = grad * self.gamma
    #     grad_x = (
    #         d * grad_lnorm
    #         - grad_lnorm.sum(axis=-1, keepdims=True)
    #         - lnorm * (grad_lnorm * lnorm).sum(axis=-1, keepdims=True)
    #         ) / (d * cp.sqrt(x_var + self.eps))
    #     return grad_x
    
    def backward(self, grad):
        _, D = grad.shape

        # 计算 dL/dx_hat, dL/dvar, dL/dmean
        dx_hat = grad * self.gamma
        dvar = cp.sum(dx_hat * (self.x_hat - self.mean) * (-0.5) * cp.power(self.var + self.eps, -1.5), axis=1, keepdims=True)
        dmean = cp.sum(dx_hat * (-1 / cp.sqrt(self.var + self.eps)), axis=1, keepdims=True) + dvar * cp.mean(-2 * (self.x_hat - self.mean), axis=1, keepdims=True)

        # 计算 dL/dx
        dx = dx_hat * (1 / cp.sqrt(self.var + self.eps)) + dvar * (2 * (self.x_hat - self.mean) / D) + dmean / D

        # 计算 dL/dgamma, dL/dbeta
        self.grad_gamma += cp.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.grad_beta += cp.sum(grad, axis=0, keepdims=True)

        return dx


    def release_memory(self):
        del self.grad_gamma, self.grad_beta

    def update_weights(self):
        self.gamma = self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.beta = self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))
        # self.release_memory()
        self.zero_grad()