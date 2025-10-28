# Multi-armed bandit homework:
#  - Epsilon-Greedy (epsilon_t = 1/t)
#  - Thompson Sampling (Gaussian, known precision)
#  - UCB1 (bonus)
#
# Outputs:
#  - rewards_log.csv   (t, Algorithm, Bandit(chosen arm), Reward)
#  - learning_eg.png, learning_ts.png, learning_ucb1.png
#  - cumulative_rewards.png
#  - optimal_arm_rate.png
#  - prints cumulative reward and regret per algorithm
# ------------------------------------------------------------
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from Bandit import Bandit  # abstract base class with __repr__, experiment, pull, report

# ------------------------------
# Utility
# ------------------------------
def cumulative_regret(rewards: np.ndarray, best_mean: float) -> float:
    """Regret(T) = T * max(mean) - sum_{t=1..T} r_t"""
    T = len(rewards)
    return T * best_mean - rewards.sum()


# ============================================================
# Epsilon-Greedy (epsilon_t = 1/t)
# ============================================================
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy with epsilon(t) = 1 / t.
    Maintains sample-mean estimates; picks argmax w.p. 1-eps_t, explore otherwise.
    """

    def __init__(self, p: float, n_arms: int):
        super().__init__(p)
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.estimates = np.zeros(n_arms, dtype=float)
        self._history_estimates: Dict[int, List[float]] = {a: [] for a in range(n_arms)}
        self.t = 0

    def epsilon(self, t: int) -> float:
        return 1.0 / max(1, t)

    def select_arm(self) -> int:
        self.t += 1
        if random.random() < self.epsilon(self.t):
            return random.randrange(self.n_arms)
        return int(np.argmax(self.estimates))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean
        self.estimates[arm] += (reward - self.estimates[arm]) / n
        # track for plotting
        for a in range(self.n_arms):
            self._history_estimates[a].append(self.estimates[a])

    # === ABC-required methods (Bandit.py) ===
    def __repr__(self):
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"

    def pull(self, true_means, noise_std):
        """One interaction step: choose, observe reward, update -> (arm, reward)."""
        a = self.select_arm()
        r = np.random.normal(loc=true_means[a], scale=noise_std)
        self.update(a, r)
        return a, r

    def experiment(self, true_means, noise_std, T):
        """Run T steps and return (rewards, actions)."""
        rewards = np.zeros(T)
        actions = np.zeros(T, dtype=int)
        for t in range(T):
            a, r = self.pull(true_means, noise_std)
            rewards[t] = r
            actions[t] = a
        return rewards, actions

    def report(self):
        return {"t": self.t, "counts": self.counts.tolist(), "estimates": self.estimates.tolist()}


# ============================================================
# Thompson Sampling (Gaussian with KNOWN precision)
# ============================================================
@dataclass
class NormalPosterior:
    mu: float   # posterior mean
    tau: float  # posterior precision (1/variance)

class ThompsonSampling(Bandit):
    """
    Rewards r | μ ~ N(μ, 1/tau), tau known (i.e., known observation variance σ² = 1/tau).
    Prior per arm: μ ~ N(mu0, 1/tau0).
      tau_n = tau0 + n_i * tau
      mu_n  = (tau0*mu0 + tau * sum(r_i)) / tau_n
    Sample θ_i ~ N(mu_n, 1/tau_n); pick argmax θ_i.
    """

    def __init__(self, p: float, n_arms: int, mu0: float = 0.0, tau0: float = 1e-2, tau: float = 1.0):
        super().__init__(p)
        self.n_arms = n_arms
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau
        self.post: List[NormalPosterior] = [NormalPosterior(mu0, tau0) for _ in range(n_arms)]
        self.counts = np.zeros(n_arms, dtype=int)
        self.sum_rewards = np.zeros(n_arms, dtype=float)
        self._history_estimates: Dict[int, List[float]] = {a: [] for a in range(n_arms)}

    def select_arm(self) -> int:
        samples = [np.random.normal(loc=pp.mu, scale=1.0 / math.sqrt(pp.tau)) for pp in self.post]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        n_i = self.counts[arm]
        tau_n = self.tau0 + n_i * self.tau
        mu_n = (self.tau0 * self.mu0 + self.tau * self.sum_rewards[arm]) / tau_n
        self.post[arm] = NormalPosterior(mu_n, tau_n)
        for a in range(self.n_arms):
            self._history_estimates[a].append(self.post[a].mu)

    # === ABC-required methods (Bandit.py) ===
    def __repr__(self):
        return f"{self.__class__.__name__}(n_arms={self.n_arms}, mu0={self.mu0}, tau0={self.tau0})"

    def pull(self, true_means, noise_std):
        # Keep observation precision consistent with noise_std
        self.tau = 1.0 / (noise_std ** 2)
        a = self.select_arm()
        r = np.random.normal(loc=true_means[a], scale=noise_std)
        self.update(a, r)
        return a, r

    def experiment(self, true_means, noise_std, T):
        rewards = np.zeros(T)
        actions = np.zeros(T, dtype=int)
        for t in range(T):
            a, r = self.pull(true_means, noise_std)
            rewards[t] = r
            actions[t] = a
        return rewards, actions

    def report(self):
        return {
            "counts": self.counts.tolist(),
            "posterior_mu": [p.mu for p in self.post],
            "posterior_tau": [p.tau for p in self.post],
        }


# ============================================================
# UCB1 (bonus)
# ============================================================
class UCB1(Bandit):
    """
    Classic UCB1:
      - Pull each arm once to initialize.
      - For t > n_arms: select argmax_i [ mu_i + sqrt(2 ln t / n_i) ].
    """

    def __init__(self, p: float, n_arms: int):
        super().__init__(p)
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.estimates = np.zeros(n_arms, dtype=float)
        self.t = 0
        self._history_estimates: Dict[int, List[float]] = {a: [] for a in range(n_arms)}

    def select_arm(self) -> int:
        self.t += 1
        # ensure each arm is tried once
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        ucb = self.estimates + np.sqrt(2.0 * math.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n
        for a in range(self.n_arms):
            self._history_estimates[a].append(self.estimates[a])

    # === ABC-required methods (Bandit.py) ===
    def __repr__(self):
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"

    def pull(self, true_means, noise_std):
        a = self.select_arm()
        r = np.random.normal(loc=true_means[a], scale=noise_std)
        self.update(a, r)
        return a, r

    def experiment(self, true_means, noise_std, T):
        rewards = np.zeros(T)
        actions = np.zeros(T, dtype=int)
        for t in range(T):
            a, r = self.pull(true_means, noise_std)
            rewards[t] = r
            actions[t] = a
        return rewards, actions

    def report(self):
        return {"t": self.t, "counts": self.counts.tolist(), "estimates": self.estimates.tolist()}


# ============================================================
# Plot helpers
# ============================================================
def plot_learning(hist: Dict[int, List[float]], means: np.ndarray, best_mean: float, title: str, outfile: str) -> None:
    plt.figure(figsize=(10, 6))
    T_hist = len(next(iter(hist.values()))) if hist else 0
    x = np.arange(1, T_hist + 1)
    for a, series in hist.items():
        plt.plot(x, series, label=f"Arm {a} (true μ={means[a]:.1f})")
    plt.axhline(best_mean, linestyle="--", linewidth=1, label="Best μ")
    plt.xlabel("t")
    plt.ylabel("Estimated mean")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


# ============================================================
# Experiment runner
# ============================================================
def run_experiment(
    bandit_means: List[float] = [1.0, 2.0, 3.0, 4.0],
    trials: int = 20000,
    noise_std: float = 1.0,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    means = np.array(bandit_means, dtype=float)
    best_mean = float(np.max(means))
    n_arms = len(means)

    # Optional file logger:
    logger.add(
        "run.log",
        rotation="5 MB",
        retention=5,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )

    results = []

    # Epsilon-Greedy
    logger.info("Running Epsilon-Greedy (epsilon_t = 1/t)")
    eg = EpsilonGreedy(p=0.0, n_arms=n_arms)
    eg_rewards, eg_actions = eg.experiment(true_means=means, noise_std=noise_std, T=trials)
    eg_hist = eg._history_estimates
    results.append(("EpsilonGreedy", eg_rewards, eg_actions, eg_hist))

    # Thompson Sampling
    logger.info("Running Thompson Sampling (Gaussian, known precision)")
    ts = ThompsonSampling(p=0.0, n_arms=n_arms, mu0=0.0, tau0=1e-2)
    ts_rewards, ts_actions = ts.experiment(true_means=means, noise_std=noise_std, T=trials)
    ts_hist = ts._history_estimates
    results.append(("ThompsonSampling", ts_rewards, ts_actions, ts_hist))

    # UCB1 (bonus)
    logger.info("Running UCB1")
    ucb = UCB1(p=0.0, n_arms=n_arms)
    ucb_rewards, ucb_actions = ucb.experiment(true_means=means, noise_std=noise_std, T=trials)
    ucb_hist = ucb._history_estimates
    results.append(("UCB1", ucb_rewards, ucb_actions, ucb_hist))

    # ---------------- CSV ----------------
    logger.info("Saving rewards_log.csv")
    frames = []
    for name, rewards, actions, _ in results:
        frames.append(pd.DataFrame({
            "t": np.arange(1, trials + 1),
            "Algorithm": name,
            "Bandit": actions,      # chosen arm at time t
            "Reward": rewards
        }))
    pd.concat(frames, ignore_index=True).to_csv("rewards_log.csv", index=False)

    # ------------- LEARNING PLOTS -------------
    logger.info("Plotting learning trajectories per algorithm")
    plot_learning(eg_hist, means, best_mean, "Learning (Epsilon-Greedy, ε=1/t)", "learning_eg.png")
    plot_learning(ts_hist, means, best_mean, "Learning (Thompson Sampling, Gaussian)", "learning_ts.png")
    plot_learning(ucb_hist, means, best_mean, "Learning (UCB1)", "learning_ucb1.png")

    # ------------- CUMULATIVE REWARDS -------------
    logger.info("Plotting cumulative rewards")
    plt.figure(figsize=(10, 6))
    for name, rewards, _, _ in results:
        plt.plot(np.arange(1, trials + 1), np.cumsum(rewards), label=name)
    plt.xlabel("t")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cumulative_rewards.png", dpi=150)
    plt.close()

    # ------------- OPTIMAL-ARM RATE -------------
    logger.info("Plotting optimal-arm rate")
    optimal = int(np.argmax(means))
    plt.figure(figsize=(10, 6))
    for name, _, actions, _ in results:
        picks = (actions == optimal).astype(int)
        cum_rate = np.cumsum(picks) / np.arange(1, trials + 1)
        plt.plot(np.arange(1, trials + 1), cum_rate, label=name)
    plt.xlabel("t")
    plt.ylabel("P(select best arm up to t)")
    plt.title("Optimal-Arm Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_arm_rate.png", dpi=150)
    plt.close()

    # ------------- PRINT STATS -------------
    logger.info("Final stats:")
    for name, rewards, _, _ in results:
        cum = float(np.cumsum(rewards)[-1])
        regret = cumulative_regret(rewards, best_mean)
        logger.info(f"[{name}]  Cumulative Reward: {cum:.2f} | Regret: {regret:.2f}")

    logger.success("Done. Files: rewards_log.csv, learning_eg.png, learning_ts.png, learning_ucb1.png, cumulative_rewards.png, optimal_arm_rate.png")


# Hook if Bandit.py wants to import and call comparison()
def comparison():
    run_experiment()


if __name__ == "__main__":
    # Default per assignment:
    # Bandit_Reward = [1,2,3,4], NumberOfTrials = 20000
    run_experiment(bandit_means=[1, 2, 3, 4], trials=20000, noise_std=1.0, seed=42)
