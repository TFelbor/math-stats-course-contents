import numpy as np


def gradient_descent(grad_J, x0, rho, k_max, epsilon, J=None):
    """
    Implements the gradient descent algorithm.

    Based on the steps:
    1) x^0 is given (x0)
    2) Loop k <= k_max: x^(k+1) = x^(k) - rho * grad_J(x^(k))
    3) Stop if ||grad_J(x^(k))|| < epsilon

    Parameters:
    ----------
    grad_J : callable
        The gradient of the cost function, ∇J(x).
        It must take a numpy array x and return a numpy array of the same shape.
    x0 : numpy.ndarray
        The initial guess (x^0).
    rho : float
        The learning rate (ϱ_k > 0), assumed constant here.
    k_max : int
        The maximum number of iterations.
    epsilon : float
        The tolerance for the gradient norm (ε).
    J : callable, optional
        The original cost function J(x). If provided, the history
        of the cost will be tracked and returned.

    Returns:
    -------
    x_final : numpy.ndarray
        The final optimized parameters.
    x_history : list
        A list containing the parameters at each iteration.
    cost_history : list
        A list containing the cost J(x) at each iteration.
        (Only if J is provided, otherwise returns empty list).
    """

    # 1) x^0 belongs to R^D given
    # Ensure x is a float array for calculations
    x = np.array(x0, dtype=float)

    # Optional: Store history of x and cost
    x_history = [x]
    cost_history = []

    if J is not None:
        cost_history.append(J(x))

    print(f"Starting Gradient Descent from x0 = {x}")
    print(f"Params: rho={rho}, k_max={k_max}, epsilon={epsilon}\n")

    # 2) do loop: while k <= k_max
    for k in range(k_max):
        # Calculate the gradient: ΔJ(x^(k))
        gradient = grad_J(x)

        # 3) Stop if: ||ΔJ(x^(k))|| < ε
        grad_norm = np.linalg.norm(gradient)

        if grad_norm < epsilon:
            print(f"--- Convergence reached! ---")
            print(f"Iteration {k + 1}: Gradient norm {grad_norm:.2e} < {epsilon:.2e}")
            break

        # 2) ... x^(k+1) = x^(k) - ϱ_k * ΔJ(x^(k))
        x = x - rho * gradient

        # Store history
        x_history.append(x)
        if J is not None:
            cost_history.append(J(x))

        # Optional: Log progress
        if (k + 1) % 10 == 0 or k == 0:
            cost_val = J(x) if J is not None else "N/A"
            print(f"Iter: {k + 1:03d} | Grad Norm: {grad_norm:9.4f} | Cost: {cost_val}")

    else:
        # This 'else' block executes if the loop finishes without 'break'
        print(f"--- Max iterations reached ({k_max}) ---")

    print(f"\nFinal Result: x = {x}")
    if J is not None:
        print(f"Final Cost: J(x) = {J(x)}")

    return x, x_history, cost_history

    # --- 1. Define our cost function and its gradient ---


def J(x):
    """Cost function: J(x) = x_1^2 + x_2^2"""
    return x[0] ** 2 + x[1] ** 2


def grad_J(x):
    """Gradient function: ∇J(x) = [2*x_1, 2*x_2]"""
    return np.array([2 * x[0], 2 * x[1]])


# --- 2. Set optimization parameters ---

# Step 1: x^0 given
initial_x = np.array([25.0, 7.0])

# Step 2: Parameters
learning_rate = 0.25  # ϱ_k
max_iterations = 100  # k_max

# Step 3: Stop condition
tolerance = 1e-6  # ε

# --- 3. Run the algorithm ---

x_final, x_hist, cost_hist = gradient_descent(
    grad_J=grad_J,
    x0=initial_x,
    rho=learning_rate,
    k_max=max_iterations,
    epsilon=tolerance,
    J=J,  # Pass the cost function to track history
)

# You can now analyze x_final, x_hist, and cost_hist
# print(f"\nHistory of x values:\n{x_hist}")
