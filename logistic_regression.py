import numpy as np

def _sigmoid(z):
    #Numerically stable sigmoid implementation
    return np.where(z>=0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def compute_loss(y, p):
    """Compute the binary cross-entropy loss."""
    eps = 1e-15  # To prevent log(0)
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + ( 1 - y) * np.log(1 - p))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N,D = X.shape

    # print(N,D)

    w = np.zeros(D)
    # print(w)
    b = 0.0

    for step in range(steps):

        #Forward pass
        z = X @ w + b
        # print(z)
        p = _sigmoid(z)

        #Compute gradients
        error = p - y
        grad_w = (X.T @ error) / N
        grad_b = np.mean(error)


        #Update parameters
        w -= lr * grad_w
        b -= lr * grad_b

        if step % 100 == 0:
            loss = compute_loss(y, p)
            print(f'Step {step:4d}: Loss = {loss:.4f}')

    return w, b


def predict(X, w, b):
    probs = _sigmoid(X @ w + b)
    return (probs >= 0.5).astype(int)


def accuracy(y, y_pred):
    return np.mean(y == y_pred)



if __name__ == '__main__':

    X = np.array([
        [1, 2],
        [2, 1],
        [2, 3],
        [3, 2],
        [3, 4],
        [4, 3]
    ])

    y = np.array([0,0,0,1,1,1])

    w, b = train_logistic_regression(X, y, lr=0.1, steps=1000)

    print("\nLearned parameters:")
    print("w =", w)
    print("b =", b)

    #Predictions
    y_pred = predict(X, w, b)
    print("\nPredicted probabilities:", y_pred)

    acc = accuracy(y, y_pred)
    print("Accuracy:", acc)
