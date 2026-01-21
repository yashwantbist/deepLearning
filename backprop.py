import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T   # (2,4)
d = np.array([[0, 1, 1, 0]])                       # (1,4)  <-- reshaped

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_network_parameters():
    inputSize = 2
    hiddenSize = 2
    outputSize = 1
    lr = 0.1
    epochs = 180000

    # Optional: make results reproducible
    np.random.seed(0)

    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # (2,2)
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1          # (2,1)
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1 # (1,2)
    b2 = np.random.rand(outputSize, 1) * 2 - 1          # (1,1)

    return w1, b1, w2, b2, lr, epochs

w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

error_list = []

for epoch in range(epochs):
    # ---------- Forward pass ----------
    z1 = w1 @ X + b1           # (2,4)
    a1 = sigmoid(z1)           # (2,4)

    z2 = w2 @ a1 + b2          # (1,4)
    a2 = sigmoid(z2)           # (1,4)

    # ---------- Backprop ----------
    # Using MSE-style derivative: dL/da2 = (a2 - d)
    # (This matches the standard gradient descent update w -= lr * grad)
    delta2 = (a2 - d) * (a2 * (1 - a2))   # (1,4)

    delta1 = (w2.T @ delta2) * (a1 * (1 - a1))  # (2,4)

    # ---------- Gradient descent updates ----------
    w2 -= lr * (delta2 @ a1.T)                 # (1,2)
    b2 -= lr * np.sum(delta2, axis=1, keepdims=True)  # (1,1)

    w1 -= lr * (delta1 @ X.T)                  # (2,2)
    b1 -= lr * np.sum(delta1, axis=1, keepdims=True)  # (2,1)

    if (epoch + 1) % 10000 == 0:
        avg_err = np.mean(np.abs(d - a2))
        print(f"Epoch: {epoch+1}, Average error: {avg_err:.5f}")
        error_list.append(avg_err)

# ---------- Testing AFTER training ----------
z1 = w1 @ X + b1
a1 = sigmoid(z1)
z2 = w2 @ a1 + b2
a2 = sigmoid(z2)

error = d - a2

print("\nFinal output after training:\n", np.round(a2, 4))
print("Ground truth:\n", d)
print("Error after training:\n", np.round(error, 4))
print("Average error:", np.mean(np.abs(error)))

plt.plot(error_list)
plt.title("Average absolute error (every 10k epochs)")
plt.xlabel("Checkpoints (x10k epochs)")
plt.ylabel("Average absolute error")
plt.show()
