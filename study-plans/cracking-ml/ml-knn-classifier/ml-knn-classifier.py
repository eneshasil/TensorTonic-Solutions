import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, dtype=float)

    preds = []
    
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train-test_point)**2, axis=1))
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        most_common = unique_labels[np.argmax(counts)]

        preds.append(most_common)

    return preds
    

        