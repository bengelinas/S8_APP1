import numpy as np
import matplotlib.pyplot as plt

# BA = I
# We want to get B which is the inverse of A, so B = AI

A = np.array([[3, 4, 1], [5, 2, 3], [6, 2, 2]])
I = np.identity(3)
B = np.identity(3)

# Gradient descent optimization formula is as follows: L = ||BA-I||^2_2 which is close to doing Frobenius Norm
# without the square root. The derivative of it is 2(BA-I)*A.T

mu = 0.005  # Learning rate, choose between 0.001, 0.005, 0.01

iterations = 1000
L = []

for _ in range(iterations):
    gradient_descent = 2 * A.T @ (A @ B - I)
    B -= mu * gradient_descent
    L.append(np.sum((B @ A - I) ** 2))

plt.plot(L, np.arange(iterations))

# The only good mu is 0.005

# --------------------------------------------------------------------------------------------------------------

A_6 = np.array([[3, 4, 1, 1, 2, 5], [5, 2, 3, 2, 2, 1], [6, 2, 2, 6, 4, 5], [1, 2, 1, 3, 1, 2], [1, 5, 2, 3, 3, 3],
                [1, 2, 2, 4, 2, 1]])
I_6 = np.identity(6)
B_6 = np.identity(6)

mu_6 = 0.000099  # Learning rate, choose between 0.001, 0.005, 0.01

iterations_6 = 10000000
L_6 = []

for _ in range(iterations_6):
    gradient_descent = 2 * A_6.T @ (A_6 @ B_6 - I_6)
    B_6 -= mu_6 * gradient_descent
    L_6.append(np.sum((B_6 @ A_6 - I_6) ** 2))

print(A_6 @ B_6)

plt.plot(L_6, np.arange(iterations_6))

# --------------------------------------------------------------------------------------------------------------

A_4 = np.array([[2, 1, 1, 2], [1, 2, 3, 2], [2, 1, 1, 2], [3, 1, 4, 1]])
I_4 = np.identity(4)
B_4 = np.identity(4)

mu_4 = 0.001  # Learning rate, choose between 0.001, 0.005, 0.01

iterations_4 = 5000
L_4 = []

for _ in range(iterations_4):
    gradient_descent = 2 * A_4.T @ (A_4 @ B_4 - I_4)
    B_4 -= mu_4 * gradient_descent
    L_4.append(np.sum((B_4 @ A_4 - I_4) ** 2))

print(A_4 @ B_4)

plt.plot(L_4, np.arange(iterations_4))
plt.show()

# ---------------- Polynomial Regression -----------------------------------

