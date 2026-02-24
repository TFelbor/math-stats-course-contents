# Comparison of Gradient-Free Optimization Algorithms: PSO vs. CBO

**Advanced Maths & Stats | Université Côte d’Azur**

This repository contains the mathematical implementation and comparative analysis of gradient-free optimization techniques. The project focuses on the trade-offs between numerical precision and computational efficiency when navigating non-convex cost landscapes.

---

### 📂 Repository Overview

The repository is divided into foundational course explorations and a comprehensive final research project.

#### 1. Exploratory Course Resources

Before diving into meta-heuristics, I implemented traditional optimization frameworks to understand the limitations of gradient-based methods.

* **`gradient_descent.py`**: A from-scratch implementation of the fixed-step gradient descent algorithm, including gradient norm tolerance and cost history tracking.
* **`optimization_playground.ipynb`**: A Jupyter Notebook used for manual gradient calculation and iterative testing. It explores how the learning rate ($\alpha$) affects convergence speed and stability.

#### 2. Final Project: PSO vs. CBO

The core of this repository is a comparative study of two bio-inspired, stochastic optimization frameworks used to find global minima in complex landscapes where gradients are unavailable or unreliable.

**Algorithms Implemented:**

* **Particle Swarm Optimization (PSO):** Mimics the social behavior of flocks, where agents update positions based on personal bests and the global swarm best.
* **Consensus-Based Optimization (CBO):** A collective dynamics approach where the population moves toward a weighted baricenter (consensus moment).

**Benchmark Functions:**
The algorithms were validated against standard optimization test functions:

* **Sphere Function:** A simple convex test.
* **Rastrigin Function:** A highly non-convex function with many local minima.
* **Ackley Function:** Characterized by a nearly flat outer region and a deep central hole.

---

### 📊 Visualizations & Results

The final project utilizes 3D surface plots and scatter maps to visualize population convergence.

<img width="1520" height="950" alt="convergence" src="https://github.com/user-attachments/assets/82786604-04b0-497e-976f-8682e2628d30" />

*Figure 1: Comparison of PSO and CBO final states across Sphere, Rastrigin, and Ackley functions.*

**Key Research Findings:**

* **Numerical Precision:** PSO proved to be the more precise optimizer, consistently finding deeper global minima with near-zero error.
* **Computational Efficiency:** CBO was the more efficient choice, requiring 20-30% fewer function evaluations to reach the vicinity of the global optimum.
* **Robustness:** Both algorithms successfully avoided "local minima traps" that typically hinder standard gradient-based methods.

---

### 🚀 Getting Started

1. **Dependencies:**
* `numpy`
* `matplotlib`


2. **Execution:**
Run the comparative analysis script to generate results and visualizations:
```bash
python opt.py
```



### 📜 License

This project is licensed under the **MIT License**.
