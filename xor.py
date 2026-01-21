import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data: XOR
# -----------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float64).T   # (2,4)
d = np.array([[0, 1, 1, 0]], dtype=np.float64)  # (1,4)

# -----------------------------
# Activations
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(a):
    # derivative using output a
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def dtanh(a):
    # derivative using output a = tanh(z)
    return 1 - a**2

# -----------------------------
# Initialize
# -----------------------------
def init_params(inputSize=2, hiddenSize=2, outputSize=1, seed=0):
    np.random.seed(seed)
    # Xavier-ish init helps tanh
    w1 = np.random.randn(hiddenSize, inputSize) * np.sqrt(1 / inputSize)
    b1 = np.zeros((hiddenSize, 1))
    w2 = np.random.randn(outputSize, hiddenSize) * np.sqrt(1 / hiddenSize)
    b2 = np.zeros((outputSize, 1))
    return w1, b1, w2, b2

w1, b1, w2, b2 = init_params(hiddenSize=2, seed=0)

# -----------------------------
# Training config (SPEED + EARLY STOP)
# -----------------------------
lr = 0.5                    # higher works well with tanh on XOR
max_epochs = 50000          # much smaller than 180k
print_every = 2000

target_error = 0.01         # stop when average abs error below this
patience = 20               # allow some checks without improvement
min_delta = 1e-5            # ignore tiny improvements

error_history = []
best_err = float("inf")
bad_checks = 0

# -----------------------------
# Train
# -----------------------------
for epoch in range(1, max_epochs + 1):
    # Forward
    z1 = w1 @ X + b1         # (2,4)
    a1 = tanh(z1)            # tanh hidden

    z2 = w2 @ a1 + b2        # (1,4)
    a2 = sigmoid(z2)         # sigmoid output (probability-ish)

    # Error metric (for reporting & early stopping)
    avg_err = np.mean(np.abs(d - a2))

    # Backprop (MSE-style)
    delta2 = (a2 - d) * dsigmoid(a2)     # (1,4)
    delta1 = (w2.T @ delta2) * dtanh(a1) # (2,4)

    # Gradient descent update
    w2 -= lr * (delta2 @ a1.T)
    b2 -= lr * np.sum(delta2, axis=1, keepdims=True)

    w1 -= lr * (delta1 @ X.T)
    b1 -= lr * np.sum(delta1, axis=1, keepdims=True)

    # Logging + store history
    if epoch % print_every == 0:
        print(f"Epoch: {epoch}, Avg error: {avg_err:.5f}")
        error_history.append(avg_err)

        # Early stopping checks happen on these checkpoints
        if avg_err < best_err - min_delta:
            best_err = avg_err
            bad_checks = 0
        else:
            bad_checks += 1

        if avg_err <= target_error:
            print(f"✅ Early stop: reached target error ({avg_err:.5f} <= {target_error}) at epoch {epoch}")
            break

        if bad_checks >= patience:
            print(f"✅ Early stop: no meaningful improvement for {patience} checks. Best={best_err:.5f}")
            break

# -----------------------------
# Final evaluation on XOR points
# -----------------------------
z1 = w1 @ X + b1
a1 = tanh(z1)
z2 = w2 @ a1 + b2
a2 = sigmoid(z2)

print("\nFinal output after training:\n", np.round(a2, 4))
print("Ground truth:\n", d)
print("Average error:", np.mean(np.abs(d - a2)))

# -----------------------------
# Plot training error curve
# -----------------------------
plt.figure()
plt.plot(np.arange(len(error_history)) * print_every, error_history)
plt.title("Average Absolute Error (checkpoints)")
plt.xlabel("Epoch")
plt.ylabel("Avg |d - a2|")
plt.show()

# -----------------------------
# Visualize decision boundary
# -----------------------------
# Create a grid in [0,1]x[0,1]
grid_step = 0.01
xx, yy = np.meshgrid(np.arange(0, 1 + grid_step, grid_step),
                     np.arange(0, 1 + grid_step, grid_step))
grid = np.vstack([xx.ravel(), yy.ravel()])  # (2, N)

# Forward pass on grid
z1g = w1 @ grid + b1
a1g = tanh(z1g)
z2g = w2 @ a1g + b2
a2g = sigmoid(z2g).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, a2g, levels=50)  # probability surface
plt.colorbar(label="Predicted output (sigmoid)")
plt.scatter([0, 0, 1, 1], [0, 1, 0, 1], c=[0, 1, 1, 0], edgecolors="k")
plt.title("XOR Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
