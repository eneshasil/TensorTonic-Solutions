import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for i in range(n_iters):
        z = X @ w + b
        y_hat = 1 / (1+np.e**(-z))
        loss = (-1/n) * np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y))
        dw = (1/n) * X.T @ (y_hat-y)
        db = (1/n) * np.sum(y_hat-y)

        w -= lr * dw
        b -= lr * db

    return (w, b)
        
