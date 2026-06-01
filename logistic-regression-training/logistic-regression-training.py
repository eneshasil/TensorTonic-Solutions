import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for i in range(steps):
        z = np.dot(X,w) + b
        p = _sigmoid(z)
        
        loss = np.sum(y * np.log(p) + (1-y) * np.log(1-p))
        loss /= n
        
        dl_dw = 1/m* np.dot(X.T, (p-y))
        dl_db = 1/m * np.sum(p-y)
        
        w = w - lr * dl_dw
        b = b - lr * dl_db

    return w, b
        