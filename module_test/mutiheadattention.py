import numpy as np


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scale_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = np.matmul(Q, np.transpose(K, (0, 1, 3, 2)))
    dk = K.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = softmax(scaled_attention_logits, axis=-1)
    output = np.matmul(attention_weights, V)
    return output

def split_heads(x, num_heads):
    batch_size = x.shape[0]
    depth = x.shape[-1]
    assert depth % num_heads == 0
    depth_per_head = depth // num_heads
    x = np.reshape(x, (batch_size, -1, num_heads, depth_per_head))
    return np.transpose(x, (0, 2, 1, 3))

def multi_head_attention(Q, K, V, num_heads, mask=None):
    batch_size = Q.shape[0]
    depth = Q.shape[-1]

    WQ = np.random.randn(depth, depth)
    WK = np.random.randn(depth, depth)
    WV = np.random.randn(depth, depth)

    Q = np.matmul(Q, WQ)
    K = np.matmul(K, WK)
    V = np.matmul(V, WV)

    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    scaled_attention = scale_dot_product_attention(Q, K, V, mask)
    scaled_attention = np.transpose(scaled_attention, (0, 2, 1, 3))

    concat_attention = np.reshape(scaled_attention, (batch_size, -1, depth))
    WO = np.random.randn(depth, depth)
    output = np.matmul(concat_attention, WO)
    return output

def residual_connection(x, sub_layer_output):
    return x + sub_layer_output

batch_size = 2
sequence_length = 6
embedding_dim = 64
num_heads = 8

X = np.random.randn(batch_size, sequence_length, embedding_dim)

output = multi_head_attention(X, X, X, num_heads)
residual_output = residual_connection(X, output)

print("多头注意力层输出形状：", output.shape)
print("残差连接层输出形状：", residual_output.shape)