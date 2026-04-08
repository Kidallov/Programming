import numpy as np
from numba import njit

# Параметры из вашего условия
d_embedding = 4
d_key = d_value = d_query = 3
d_feed_forward = 8
n_attention_heads = 2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def attention(x, WQ, WK, WV):
    K = x @ WK
    V = x @ WV
    Q = x @ WQ
    scores = (Q @ K.T) / np.sqrt(d_key)
    return softmax(scores) @ V

def multi_head_attention(x, WQs, WKs, WVs):
    heads = []
    for i in range(n_attention_heads):
        heads.append(attention(x, WQs[i], WKs[i], WVs[i]))
    
    attentions = np.concatenate(heads, axis=1)
    # Генерируем веса проекции (фиксируем сид для воспроизводимости)
    np.random.seed(42)
    W_out = np.random.randn(n_attention_heads * d_value, d_embedding)
    return attentions @ W_out

def encoder_block(x, WQs, WKs, WVs, W1, b1, W2, b2):
    Z = multi_head_attention(x, WQs, WKs, WVs)
    # Feed Forward часть
    res = relu(Z @ W1 + b1) @ W2 + b2
    return res

# --- Версия для noGIL (Numba) ---
# Numba не очень любит списки массивов разной формы, 
# поэтому для примера оптимизируем самую тяжелую часть — Feed Forward
@njit(nogil=True)
def encoder_block_nogil(x, W1, b1, W2, b2):
    # Упрощенная имитация блока для демонстрации работы вне GIL
    Z = x @ W1
    # Аналог ReLU + Linear
    out = np.maximum(0.0, Z + b1) @ W2 + b2
    return out