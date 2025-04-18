from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sns.set_theme()

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("imgs/"),
)
params

# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

class QModel:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

class SARSAModel:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state, next_action):
        """SARSA update: Q(s,a) ← Q(s,a) + α [r + γ * Q(s',a') - Q(s,a)]"""
        q_next = self.qtable[new_state, next_action]
        delta = reward + self.gamma * q_next - self.qtable[state, action]
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # Find the indices where the Q-value equals the maximum value
            # Choose a random action from the indices where the Q-value is maximum
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = rng.choice(max_ids)
        return action

def run_env(learner, model_type):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    successes = np.zeros((params.total_episodes, params.n_runs))
    failures = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):
        learner.reset_qtable()

        for episode in tqdm(episodes, desc=f"{model_type.upper()} Run {run+1}", leave=False):
            state = env.reset(seed=params.seed)[0]
            action = explorer.choose_action(env.action_space, state, learner.qtable)
            done = False
            total_rewards = 0
            step = 0

            while not done:
                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = explorer.choose_action(env.action_space, new_state, learner.qtable)

                if model_type == "q_learning":
                    learner.qtable[state, action] = learner.update(state, action, reward, new_state)
                elif model_type == "sarsa":
                    learner.qtable[state, action] = learner.update(state, action, reward, new_state, next_action)

                state = new_state
                action = next_action
                total_rewards += reward
                step += 1

            rewards[episode, run] = total_rewards
            steps[episode, run] = step
            successes[episode, run] = 1 if total_rewards > 0 else 0
            failures[episode, run] = 1 if total_rewards == 0 else 0

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions, successes, failures

def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map_comparison(q_qtable, sarsa_qtable, env, map_size):
    def prepare_policy(qtable):
        q_val_max = qtable.max(axis=1).reshape(map_size, map_size)
        q_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
        directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        arrows = np.empty(q_best_action.size, dtype=str)
        eps = np.finfo(float).eps
        for idx, val in enumerate(q_best_action.flatten()):
            if q_val_max.flatten()[idx] > eps:
                arrows[idx] = directions[val]
        return q_val_max, arrows.reshape(map_size, map_size)

    q_max, q_dir = prepare_policy(q_qtable)
    sarsa_max, sarsa_dir = prepare_policy(sarsa_qtable)

    # Get rendered tile background (frozenlake map graphic)
    render_img = env.render()  # shape = (height, width, 3)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Q-learning heatmap on FrozenLake background
    ax[0].imshow(render_img)
    sns.heatmap(q_max, annot=q_dir, fmt="", ax=ax[0], cmap="Blues", alpha=0.6,
                linewidths=0.7, linecolor="black", cbar=False,
                xticklabels=False, yticklabels=False,
                annot_kws={"fontsize": "xx-large"})
    ax[0].set_title("Q-Learning Policy (Overlayed)")

    # SARSA heatmap on FrozenLake background
    ax[1].imshow(render_img)
    sns.heatmap(sarsa_max, annot=sarsa_dir, fmt="", ax=ax[1], cmap="Greens", alpha=0.6,
                linewidths=0.7, linecolor="black", cbar=False,
                xticklabels=False, yticklabels=False,
                annot_kws={"fontsize": "xx-large"})
    ax[1].set_title("SARSA Policy (Overlayed)")

    for a in ax:
        a.axis("off")
        for _, spine in a.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

    plt.suptitle(f"Q-Learning vs SARSA Policy (Overlayed) - {map_size}x{map_size}")
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"q_vs_sarsa_overlay_{map_size}x{map_size}.png")
    plt.show()

def plot_states_actions_distribution(q_states, q_actions, s_states, s_actions, map_size):
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    sns.histplot(q_states, ax=ax[0], color="blue", label="Q-Learning", stat="density", kde=True)
    sns.histplot(s_states, ax=ax[0], color="green", label="SARSA", stat="density", kde=True, alpha=0.6)
    ax[0].set_title("States")
    ax[0].legend()

    sns.histplot(q_actions, ax=ax[1], color="blue", label="Q-Learning", stat="density")
    sns.histplot(s_actions, ax=ax[1], color="green", label="SARSA", stat="density", alpha=0.6)
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    ax[1].legend()

    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_model_comparison(df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    sns.lineplot(data=df, x="Episodes", y="CumRewards", hue="Model", ax=ax[0])
    ax[0].set_title("Cumulative Rewards")
    
    sns.lineplot(data=df, x="Episodes", y="Steps", hue="Model", ax=ax[1])
    ax[1].set_title("Steps per Episode")
    
    for a in ax:
        a.legend(title="Model")
        a.set(xlabel="Episode")
    
    fig.tight_layout()
    plt.savefig(params.savefig_folder / f"compare_{map_size}x{map_size}.png")
    plt.show()
    
def plot_success_failure_comparison(q_successes, q_failures, sarsa_successes, sarsa_failures, map_size):
    episodes = np.arange(q_successes.shape[0])
    
    plt.figure(figsize=(12, 6))
    
    # Success rates
    plt.plot(episodes, q_successes.mean(axis=1), label="Q-Learning Success", color="green")
    plt.plot(episodes, sarsa_successes.mean(axis=1), label="SARSA Success", color="limegreen", linestyle="--")
    
    # Failure rates
    plt.plot(episodes, q_failures.mean(axis=1), label="Q-Learning Failure", color="red")
    plt.plot(episodes, sarsa_failures.mean(axis=1), label="SARSA Failure", color="orange", linestyle="--")

    plt.title(f"Success and Failure Rate Comparison - {map_size}x{map_size}")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"success_failure_comparison_{map_size}x{map_size}.png")
    plt.show()

map_sizes = [7, 9, 11]
#res_all = pd.DataFrame()
#st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(size=map_size, p=params.proba_frozen, seed=params.seed),
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(params.seed)
    
    q_learner = QModel(params.learning_rate, params.gamma, params.state_size, params.action_size)
    sarsa_learner = SARSAModel(params.learning_rate, params.gamma, params.state_size, params.action_size)
    explorer = EpsilonGreedy(params.epsilon)

    print(f"Map size: {map_size}x{map_size}")
    q_rewards, q_steps, episodes, q_qtables, q_states, q_actions, q_successes, q_failures = run_env(q_learner, "q_learning")
    sarsa_rewards, sarsa_steps, _, sarsa_qtables, sarsa_states, sarsa_actions, sarsa_successes, sarsa_failures = run_env(sarsa_learner, "sarsa")

    def make_df(model_name, rewards, steps):
        df = pd.DataFrame({
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
            "Model": model_name,
            "MapSize": f"{map_size}x{map_size}",
        })
        df["CumRewards"] = rewards.cumsum(axis=0).flatten(order="F")
        return df

    df_q = make_df("Q-Learning", q_rewards, q_steps)
    df_sarsa = make_df("SARSA", sarsa_rewards, sarsa_steps)
    df_all = pd.concat([df_q, df_sarsa], ignore_index=True)

    plot_model_comparison(df_all)
    plot_states_actions_distribution(q_states, q_actions, sarsa_states, sarsa_actions, map_size)
    plot_success_failure_comparison(q_successes, q_failures, sarsa_successes, sarsa_failures, map_size)

    qtable_q = q_qtables.mean(axis=0)
    qtable_sarsa = sarsa_qtables.mean(axis=0)
    plot_q_values_map_comparison(qtable_q, qtable_sarsa, env, map_size)

    env.close()
