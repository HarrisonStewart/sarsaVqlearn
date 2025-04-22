import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple

sns.set_theme()

# Parameters
class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    seed: int
    n_runs: int
    savefig_folder: Path
    epsilon_decay: float
    min_epsilon: float

params = Params(
    total_episodes=2000,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.8,  # Start more explorative
    seed=123,
    n_runs=10,
    savefig_folder=Path("taxi_imgs/"),
    epsilon_decay=0.995,
    min_epsilon=0.05
)


params.savefig_folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(params.seed)

# Environment
env = gym.make("Taxi-v3")
env.action_space.seed(params.seed)
state_size = env.observation_space.n
action_size = env.action_space.n

# Q-learning Model
class QModel:
    def __init__(self):
        self.q = np.zeros((state_size, action_size))

    def reset_qtable(self):
        self.q[:] = 0

    def update(self, s, a, r, s_):
        best_next = np.max(self.q[s_])
        delta = r + params.gamma * best_next - self.q[s, a]
        self.q[s, a] += params.learning_rate * delta

# SARSA Model
class SARSAModel:
    def __init__(self):
        self.q = np.zeros((state_size, action_size))

    def reset_qtable(self):
        self.q[:] = 0

    def update(self, s, a, r, s_, a_):
        delta = r + params.gamma * self.q[s_, a_] - self.q[s, a]
        self.q[s, a] += params.learning_rate * delta

# Epsilon-greedy strategy
class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, state, q):
        if rng.uniform(0, 1) < self.epsilon:
            return rng.integers(action_size)
        else:
            return rng.choice(np.flatnonzero(q[state] == np.max(q[state])))

    def decay(self):
        self.epsilon = max(params.min_epsilon, self.epsilon * params.epsilon_decay)

# Training loop
def run_env(learner, model_type):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    successes = np.zeros((params.total_episodes, params.n_runs))

    for run in range(params.n_runs):
        explorer = EpsilonGreedy(params.epsilon)
        learner.reset_qtable()

        for episode in tqdm(range(params.total_episodes), desc=f"{model_type.upper()} Run {run+1}"):
            run_seed = params.seed + run
            state = env.reset(seed=run_seed)[0]
            action = explorer.choose(state, learner.q)

            done = False
            total_reward = 0
            successful_dropoff = False

            while not done:
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                new_action = explorer.choose(new_state, learner.q)

                if model_type == "q_learning":
                    learner.update(state, action, reward, new_state)
                else:
                    learner.update(state, action, reward, new_state, new_action)

                if reward == 20:
                    successful_dropoff = True

                state = new_state
                action = new_action
                total_reward += reward

            rewards[episode, run] = total_reward
            successes[episode, run] = 1 if successful_dropoff else 0

            explorer.decay()  # decay epsilon after each episode

    return rewards, successes, learner.q

# Run models
q_model = QModel()
sarsa_model = SARSAModel()

q_rewards, q_successes, q_qtable = run_env(q_model, "q_learning")
sarsa_rewards, sarsa_successes, sarsa_qtable = run_env(sarsa_model, "sarsa")

# Convert to DataFrame
episodes = np.arange(params.total_episodes)
df_q = pd.DataFrame({
    "Episode": np.tile(episodes, params.n_runs),
    "Reward": q_rewards.flatten(),
    "Success": q_successes.flatten(),
    "Model": "Q-Learning"
})
df_sarsa = pd.DataFrame({
    "Episode": np.tile(episodes, params.n_runs),
    "Reward": sarsa_rewards.flatten(),
    "Success": sarsa_successes.flatten(),
    "Model": "SARSA"
})
df_all = pd.concat([df_q, df_sarsa])

# Rolling averages
window = 20
df_all["RollingAvg"] = df_all.groupby("Model")["Reward"].transform(lambda x: x.rolling(window, min_periods=1).mean())
df_all["RollingSuccess"] = df_all.groupby("Model")["Success"].transform(lambda x: x.rolling(window, min_periods=1).mean())

# Plot: Reward
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_all, x="Episode", y="RollingAvg", hue="Model", linewidth=2)
plt.title(f"Taxi-v3: Rolling Avg Reward (Window={window})")
plt.xlabel("Episode")
plt.ylabel("Rolling Avg Reward")
plt.axhline(8, color='gray', linestyle='--', label="Baseline Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(params.savefig_folder / "taxi_q_vs_sarsa_reward.png")
plt.show()

# Plot: Success Rate
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_all, x="Episode", y="RollingSuccess", hue="Model", linewidth=2)
plt.title(f"Taxi-v3: Rolling Success Rate (Window={window})")
plt.xlabel("Episode")
plt.ylabel("Rolling Success Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig(params.savefig_folder / "taxi_q_vs_sarsa_success_rate.png")
plt.show()

def visualize_policy(q_table, title, filename):
    grid_size = 5
    policy_grid = np.full((grid_size, grid_size), "", dtype=object)

    for row in range(grid_size):
        for col in range(grid_size):
            # fixed passenger location = 0, destination = 4
            state = env.unwrapped.encode(row, col, 0, 4)
            action = np.argmax(q_table[state])
            action_map = {0: "↓", 1: "↑", 2: "→", 3: "←", 4: "⛟", 5: "✓"}
            policy_grid[row, col] = action_map[action]

    plt.figure(figsize=(6, 5))
    sns.heatmap(np.zeros_like(policy_grid, dtype=float), annot=policy_grid, fmt="",
                cbar=False, linewidths=0.5, linecolor="black", cmap="Blues")
    plt.title(title)
    plt.xticks(np.arange(grid_size) + 0.5, labels=[f"Col {c}" for c in range(grid_size)])
    plt.yticks(np.arange(grid_size) + 0.5, labels=[f"Row {r}" for r in range(grid_size)], rotation=0)
    plt.tight_layout()
    plt.savefig(params.savefig_folder / filename)
    plt.show()

visualize_policy(q_qtable, "Q-Learning Policy (Pickup=0, Dropoff=4)", "taxi_q_policy.png")
visualize_policy(sarsa_qtable, "SARSA Policy (Pickup=0, Dropoff=4)", "taxi_sarsa_policy.png")

print("Success counts (Q-Learning):", np.sum(q_successes))
print("Success counts (SARSA):", np.sum(sarsa_successes))

env.close()
