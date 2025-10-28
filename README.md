# DS223: Multi-Armed Bandit Algorithms — Arevik Melikyan

## 📘 Overview
This project implements and compares multiple algorithms for the **multi-armed bandit problem**, a classic reinforcement-learning setup used to model decision-making under uncertainty.

The assignment explores:
- **Epsilon-Greedy** (ε = 1/t decaying exploration rate)
- **Thompson Sampling** (Gaussian case, known precision)
- **UCB1** (Upper Confidence Bound algorithm — bonus implementation)

Each algorithm learns the optimal arm through trial and error and is evaluated in terms of **learning convergence**, **cumulative reward**, and **regret**.

---

## 🧩 Repository Structure

├── Bandit.py # Abstract base class (provided)
├── main.py # Implementation of all algorithms & experiments
├── requirements.txt # Dependencies for reproducibility
├── Homework 2-2.pdf # Assignment instructions/report
└── .gitignore # Excludes cache & generated artifacts


---

## ⚙️ Algorithms Implemented

### 1. Epsilon-Greedy
- Exploration probability decays as ε = 1/t.
- Selects random arm with probability ε, otherwise the estimated best arm.
- Simple yet effective baseline.

### 2. Thompson Sampling (Gaussian, known precision)
- Bayesian approach with Normal–Normal conjugacy.
- Maintains posterior parameters (μ, τ) for each arm.
- Samples from each posterior to choose the next arm.

### 3. UCB1 (Bonus)
- Deterministic exploration using confidence bounds.
- Selects arm maximizing `μ_i + sqrt(2 ln t / n_i)`.

---

## 🧠 Key Concepts
- **Exploration vs. Exploitation:** Algorithms balance exploring new arms vs. exploiting known good ones.
- **Regret:**  
  \( R(T) = T \cdot \max_i \mu_i - \sum_{t=1}^T r_t \)
- **Cumulative Reward:** Measures total performance over time.

---

## 📊 Outputs
The code automatically produces the following visualizations:

| Plot | Description |
|------|--------------|
| `learning_eg.png` | Epsilon-Greedy: estimated mean per arm |
| `learning_ts.png` | Thompson Sampling: posterior mean evolution |
| `learning_ucb1.png` | UCB1: mean + confidence bound |
| `cumulative_rewards.png` | Cumulative rewards comparison |
| `optimal_arm_rate.png` | Probability of selecting the best arm |
| `rewards_log.csv` | Logged rewards and actions per trial |

---

## 🧰 Dependencies
Install all dependencies from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
