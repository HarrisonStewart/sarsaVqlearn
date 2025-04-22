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
    total_episodes=2000,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1,
    seed=123,
    n_runs=3,
    bins=5,
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

                # === Reward Shaping ===
                shaped_reward = reward

                # Decompose state vector
                x, y, vx, vy, angle, angular_v, left_contact, right_contact = obs

                shaped_reward += -0.3 * abs(vx)      # penalize horizontal speed
                shaped_reward += -0.3 * abs(vy)      # penalize vertical speed
                shaped_reward += -0.2 * abs(angle)   # penalize tilt
                shaped_reward += -0.1 * abs(angular_v)  # penalize spin

                if left_contact or right_contact:
                    shaped_reward += 10  # reward leg contact

                # === Update with shaped reward ===
                if model_type == "q_learning":
                    learner.update(state, action, shaped_reward, obs)
                else:
                    learner.update(state, action, shaped_reward, obs, new_action)

                state = new_state
                action = new_action
                total_reward += reward  # accumulate original reward for tracking

            rewards[episode, run] = total_reward

    return rewards

def estimate_convergence(df, threshold=150):
    convergence = {}
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        avg = model_data.groupby("Episode")["Reward"].mean()
        rolling = avg.rolling(window=window, min_periods=1).mean()
        above_thresh = rolling[rolling > threshold]
        if not above_thresh.empty:
            convergence[model] = above_thresh.index[0]
        else:
            convergence[model] = None
    return convergence

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

# Compute rolling average reward
window = 50
df_all["RollingAvg"] = df_all.groupby("Model")["Reward"].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_all, x="Episode", y="RollingAvg", hue="Model", linewidth=2)
plt.title(f"LunarLander-v3: Rolling Avg Reward (Window={window})")
plt.xlabel("Episode")
plt.ylabel("Rolling Average Reward")
plt.axhline(200, color='gray', linestyle='--', label="Ideal Target (200)")
plt.axhline(0, color='gray', linestyle=':', label="Zero Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(params.savefig_folder / "lunarlander_convergence.png")
plt.show()

convergence_points = estimate_convergence(df_all, threshold=150)
print("Estimated convergence episodes:")
for model, ep in convergence_points.items():
    print(f"  {model}: {'Not converged' if ep is None else f'Episode {ep}'}")

env.close()
