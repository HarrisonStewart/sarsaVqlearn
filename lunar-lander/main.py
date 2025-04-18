import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple

sns.set_theme()

class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    seed: int
    n_runs: int
    bins: int
    savefig_folder: Path

params = Params(
    total_episodes=500,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1,
    seed=123,
    n_runs=10,
    bins=10,
    savefig_folder=Path("lunar_imgs/")
)

params.savefig_folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(params.seed)

env = gym.make("LunarLander-v3")
env.action_space.seed(params.seed)

state_low = env.observation_space.low
state_high = env.observation_space.high
state_bins = [np.linspace(state_low[i], state_high[i], params.bins - 1) for i in range(len(state_low))]
n_states = tuple([params.bins] * len(state_low))
n_actions = env.action_space.n

def discretize(state):
    return tuple(
        int(np.digitize(state[i], state_bins[i])) for i in range(len(state))
    )

class QModel:
    def __init__(self):
        self.q = np.zeros(n_states + (n_actions,))

    def reset_qtable(self):
        self.q[:] = 0

    def update(self, s, a, r, s_):
        s_ = discretize(s_)
        best_next = np.max(self.q[s_])
        delta = r + params.gamma * best_next - self.q[s + (a,)]
        self.q[s + (a,)] += params.learning_rate * delta

class SARSAModel:
    def __init__(self):
        self.q = np.zeros(n_states + (n_actions,))

    def reset_qtable(self):
        self.q[:] = 0

    def update(self, s, a, r, s_, a_):
        s_ = discretize(s_)
        delta = r + params.gamma * self.q[s_ + (a_,)] - self.q[s + (a,)]
        self.q[s + (a,)] += params.learning_rate * delta

class EpsilonGreedy:
    def choose(self, state, q):
        if rng.uniform(0, 1) < params.epsilon:
            return rng.integers(n_actions)
        else:
            return rng.choice(np.flatnonzero(q[state] == np.max(q[state])))

def run_env(learner, model_type):
    rewards = np.zeros((params.total_episodes, params.n_runs))

    for run in range(params.n_runs):
        learner.reset_qtable()

        for episode in tqdm(range(params.total_episodes), desc=f"{model_type.upper()} Run {run+1}"):
            state = discretize(env.reset(seed=params.seed)[0])
            action = explorer.choose(state, learner.q)

            done = False
            total_reward = 0

            while not done:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                new_state = discretize(obs)
                new_action = explorer.choose(new_state, learner.q)

                if model_type == "q_learning":
                    learner.update(state, action, reward, obs)
                else:
                    learner.update(state, action, reward, obs, new_action)

                state = new_state
                action = new_action
                total_reward += reward

            rewards[episode, run] = total_reward

    return rewards

explorer = EpsilonGreedy()

q_model = QModel()
sarsa_model = SARSAModel()

q_rewards = run_env(q_model, "q_learning")
sarsa_rewards = run_env(sarsa_model, "sarsa")

# Plotting
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

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_all, x="Episode", y="Reward", hue="Model", ci="sd")
plt.title("LunarLander-v2: Q-Learning vs SARSA")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig(params.savefig_folder / "lunarlander_q_vs_sarsa.png")
plt.show()

env.close()
