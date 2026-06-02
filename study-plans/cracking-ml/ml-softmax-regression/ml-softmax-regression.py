import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int) 
    
    n, d = X.shape
    
    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    
    y_one_hot = np.zeros((n, n_classes))
    y_one_hot[np.arange(n), y] = 1

    for _ in range(n_iters):
        z = X @ w + b

        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        

        p = exp_z / np.sum(exp_z, axis=1, keepdims=True)


        dw = (1.0 / n) * (X.T @ (p - y_one_hot))
        db = (1.0 / n) * np.sum(p - y_one_hot, axis=0)

        w -= lr * dw
        b -= lr * db

    return (w, b)
