import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import NamedTuple

sns.set_theme()

# --- Parameters ---
class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    n_runs: int
    seed: int
    savefig_folder: Path

params = Params(
    total_episodes=1000,
    learning_rate=0.5,
    gamma=0.99,
    epsilon=0.1,
    n_runs=10,
    seed=42,
    savefig_folder=Path("cliff_imgs/")
)

params.savefig_folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(params.seed)

# --- Environment ---
env = gym.make("CliffWalking-v0")
env.action_space.seed(params.seed)
state_size = env.observation_space.n
action_size = env.action_space.n

# --- Epsilon-greedy strategy ---
class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, state, q):
        if rng.random() < self.epsilon:
            return rng.integers(action_size)
        else:
            return rng.choice(np.flatnonzero(q[state] == np.max(q[state])))
        
# --- Q-Learning ---
class QModel:
    def __init__(self):
        self.q = np.zeros((state_size, action_size))

    def reset_qtable(self):
        self.q[:] = 0

    def update(self, s, a, r, s_):
        best_next = np.max(self.q[s_])
        self.q[s, a] += params.learning_rate * (r + params.gamma * best_next - self.q[s, a])

# --- SARSA ---
class SARSAModel:
    def __init__(self):
        self.q = np.zeros((state_size, action_size))

    def reset_qtable(self):
        self.q[:] = 0

    def update(self, s, a, r, s_, a_):
        self.q[s, a] += params.learning_rate * (r + params.gamma * self.q[s_, a_] - self.q[s, a])

# --- Training Loop ---
def run_env(model, model_type):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    falls = np.zeros((params.total_episodes, params.n_runs))
    step_penalty = 0.05  # Discourage long paths
    goal_reward_bonus = 100  # Extra reward for reaching goal without falling

    for run in range(params.n_runs):
        model.reset_qtable()
        explorer = EpsilonGreedy(params.epsilon)

        for episode in tqdm(range(params.total_episodes), desc=f"{model_type.upper()} Run {run+1}"):
            state, _ = env.reset(seed=params.seed + run)
            action = explorer.choose(state, model.q)
            done = False
            total_reward = 0

            while not done:
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if reward == -100:
                    falls[episode, run] += 1

                # Base reward shaping: penalize every step
                shaped_reward = reward - step_penalty

                # Bonus if the goal is reached (reward==0 and not a cliff fall)
                if reward == 0 and done:
                    shaped_reward += goal_reward_bonus

                new_action = explorer.choose(new_state, model.q)

                if model_type == "q_learning":
                    model.update(state, action, shaped_reward, new_state)
                else:
                    model.update(state, action, shaped_reward, new_state, new_action)

                state = new_state
                action = new_action
                total_reward += shaped_reward

            rewards[episode, run] = total_reward

    return rewards, model.q, falls

# --- Run models ---
q_model = QModel()
sarsa_model = SARSAModel()

q_rewards, q_qtable, q_falls = run_env(q_model, "q_learning")
sarsa_rewards, sarsa_qtable, sarsa_falls = run_env(sarsa_model, "sarsa")

# --- DataFrame for plotting ---
episodes = np.arange(params.total_episodes)
df_q = pd.DataFrame({
    "Episode": np.tile(episodes, params.n_runs),
    "Reward": q_rewards.flatten(),
    "Model": "Q-Learning"
})
df_sarsa = pd.DataFrame({
    "Episode": np.tile(episodes, params.n_runs),
    "Reward": sarsa_rewards.flatten(),
    "Model": "SARSA"
})
df_all = pd.concat([df_q, df_sarsa])
df_all["RollingReward"] = df_all.groupby("Model")["Reward"].transform(lambda x: x.rolling(20, min_periods=1).mean())

# --- Plot Rolling Reward ---
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_all, x="Episode", y="RollingReward", hue="Model")
plt.title("CliffWalking-v0: Rolling Avg Reward (Window=20)")
plt.grid(True)
plt.tight_layout()
plt.savefig(params.savefig_folder / "rewards.png")
plt.show()

# --- Visualize Cliff Falls ---
avg_q_falls = q_falls.mean(axis=1)
avg_sarsa_falls = sarsa_falls.mean(axis=1)

plt.figure(figsize=(14, 6))
plt.plot(avg_q_falls, label="Q-Learning", alpha=0.7)
plt.plot(avg_sarsa_falls, label="SARSA", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Average Cliff Falls per Episode")
plt.title("CliffWalking-v0: Average Cliff Falls")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(params.savefig_folder / "cliff_falls.png")
plt.show()

# --- Visualize Q-table Policy and Value Heatmap ---
def visualize_policy_and_value(q_table, title_prefix, filename_prefix):
    height, width = 4, 12
    action_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    grid = np.full((height, width), "", dtype=object)
    values = np.zeros((height, width))

    for state in range(state_size):
        row = state // width
        col = state % width
        best_action = np.argmax(q_table[state])
        grid[row, col] = action_map[best_action]
        values[row, col] = np.max(q_table[state])

    plt.figure(figsize=(12, 4))
    sns.heatmap(values, annot=grid, fmt="", cbar=True,
                linewidths=0.5, linecolor="black", cmap="coolwarm")
    plt.title(f"{title_prefix} Policy with State Values")
    plt.xticks(np.arange(width) + 0.5, labels=[f"{i}" for i in range(width)])
    plt.yticks(np.arange(height) + 0.5, labels=[f"{i}" for i in range(height)], rotation=0)
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"{filename_prefix}_heatmap.png")
    plt.show()

visualize_policy_and_value(q_qtable, "Q-Learning", "q_policy")
visualize_policy_and_value(sarsa_qtable, "SARSA", "sarsa_policy")

env.close()
