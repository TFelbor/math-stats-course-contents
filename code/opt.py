import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Test Functions from Wikipedia ---
def sphere(x):
    return np.sum(x**2, axis=-1)

def rastrigin(x):
    d = x.shape[-1]
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

def ackley(x):
    d = x.shape[-1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=-1) / d))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=-1) / d)
    return term1 + term2 + 20 + np.e

# --- 1. Particle Swarm Optimization (PSO) ---
def pso_optimize(func, d=2, n=50, a=-5, b=5, max_iter=200, tol=1e-6):
    x = np.random.uniform(a, b, (n, d))
    v = np.random.uniform(-1, 1, (n, d))
    p_best = x.copy()
    p_best_val = func(x)
    g_best = p_best[np.argmin(p_best_val)]
    g_best_val = np.min(p_best_val)

    c1, c2, w = 1.5, 1.5, 0.7
    budget = n
    start_time = time.time()

    for k in range(max_iter):
        prev_baricenter = np.mean(x, axis=0)
        r1, r2 = np.random.rand(n, d), np.random.rand(n, d)
        # Velocity update based on personal and global bests
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = x + v
        # Boundary constraint handling
        x = np.clip(x, a, b)

        vals = func(x)
        budget += n

        mask = vals < p_best_val
        p_best[mask] = x[mask]
        p_best_val[mask] = vals[mask]

        if np.min(vals) < g_best_val:
            g_best = x[np.argmin(vals)]
            g_best_val = np.min(vals)

        curr_baricenter = np.mean(x, axis=0)
        if np.linalg.norm(curr_baricenter - prev_baricenter) < tol:
            break

    return g_best_val, budget, time.time() - start_time, x

# --- 2. Consensus Based Optimization (CBO) ---
def cbo_optimize(func, d=2, n=50, a=-5, b=5, max_iter=200, tol=1e-6):
    alpha, lambd, dt, sigma = 50.0, 1.0, 0.1, 0.5
    x = np.random.uniform(a, b, (n, d))
    budget = 0
    start_time = time.time()

    for k in range(max_iter):
        fx = func(x)
        budget += n

        # Calculate consensus moment m_alpha
        weights = np.exp(-alpha * (fx - np.min(fx)))
        m_alpha = np.sum(x * weights[:, None], axis=0) / np.sum(weights)

        prev_x = x.copy()
        drift = lambd * (m_alpha - x) * dt
        # Brownian motion term
        noise = sigma * np.sqrt(dt) * np.linalg.norm(x - m_alpha, axis=1)[:, None] * np.random.randn(n, d)
        x = x + drift + noise
        x = np.clip(x, a, b)

        if np.linalg.norm(np.mean(x, axis=0) - np.mean(prev_x, axis=0)) < tol:
            break

    return func(m_alpha), budget, time.time() - start_time, x

# --- Visualization Functions ---
def plot_functions(functions):
    fig = plt.figure(figsize=(15, 5))
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    for i, (name, f) in enumerate(functions.items()):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        Z = np.array([f(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=False)
        ax.set_title(f"Surface: {name}")

    plt.suptitle("Objective Function Landscapes")
    plt.tight_layout()
    plt.show()

def plot_convergence(functions, results_data):
    fig = plt.figure(figsize=(18, 10))
    x_grid = np.linspace(-5, 5, 100)
    y_grid = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    for i, (name, func) in enumerate(functions.items()):
        Z = np.array([func(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        # PSO Plot
        ax_pso = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax_pso.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
        final_x_pso = results_data[name]['pso_x']
        ax_pso.scatter(final_x_pso[:,0], final_x_pso[:,1], func(final_x_pso), color='red', s=20)
        ax_pso.set_title(f"PSO Final State: {name}")

        # CBO Plot
        ax_cbo = fig.add_subplot(2, 3, i + 4, projection='3d')
        ax_cbo.plot_surface(X, Y, Z, cmap='plasma', alpha=0.3)
        final_x_cbo = results_data[name]['cbo_x']
        ax_cbo.scatter(final_x_cbo[:,0], final_x_cbo[:,1], func(final_x_cbo), color='blue', s=20)
        ax_cbo.set_title(f"CBO Final State: {name}")

    plt.suptitle("Final Population Distribution at Convergence")
    plt.tight_layout()
    plt.show()

# --- Main Execution Loop ---
functions = {"Sphere": sphere, "Rastrigin": rastrigin, "Ackley": ackley}
results_store = {}

print("--- Project Analysis ---")
for name, f in functions.items():
    # Unpack all 4 values to avoid ValueError
    pso_val, pso_bug, pso_t, pso_x = pso_optimize(f)
    cbo_val, cbo_bug, cbo_t, cbo_x = cbo_optimize(f)

    results_store[name] = {'pso_x': pso_x, 'cbo_x': cbo_x}

    print(f"\nResults for {name}:")
    print(f"  PSO -> Val: {pso_val:.4e} | Budget: {pso_bug} | Time: {pso_t:.4f}s")
    print(f"  CBO -> Val: {cbo_val:.4e} | Budget: {cbo_bug} | Time: {cbo_t:.4f}s")

# Show pure function shapes
plot_functions(functions)

# Show convergence with data points
plot_convergence(functions, results_store)
