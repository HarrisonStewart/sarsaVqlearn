from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from scipy.stats import sem

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
    total_episodes=3000,
    learning_rate=0.5,
    gamma=0.999,
    epsilon=0.1,
    map_size=10,
    seed=123,
    is_slippery=True,
    n_runs = 10,
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
            max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
            action = rng.choice(max_ids)
        return action

def run_env(learner, model_type):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    successes = np.zeros((params.total_episodes, params.n_runs))
    hole_failures = np.zeros((params.total_episodes, params.n_runs))
    timeout_failures = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []
    hole_falls = np.zeros((params.total_episodes, params.n_runs))


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

                # Check if the agent fell into a hole
                row, col = divmod(new_state, map_size)
                tile_type = env.unwrapped.desc[row][col]
                if tile_type == b'H':
                    hole_falls[episode, run] += 1

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
            if total_rewards > 0:
                successes[episode, run] = 1
            else:
                row, col = divmod(state, map_size)
                tile_type = env.unwrapped.desc[row][col]
                if tile_type == b'H':
                    hole_failures[episode, run] = 1
                else:
                    timeout_failures[episode, run] = 1

            qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions, successes, hole_falls, hole_failures, timeout_failures


def evaluate_policy(model, model_type, qtable, n_episodes=1000):
    total_rewards = []
    success_count = []
    hole_failure_flags = []
    timeout_failure_flags = []

    exploit_only = EpsilonGreedy(epsilon=0.0)

    for episode in range(n_episodes):
        state = env.reset(seed=params.seed + episode + 1000)[0]
        done = False
        total_reward = 0
        fell = False

        while not done:
            action = exploit_only.choose_action(env.action_space, state, qtable)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            row, col = divmod(new_state, map_size)
            tile_type = env.unwrapped.desc[row][col]
            if tile_type == b'H':
                fell = True

            total_reward += reward
            state = new_state

        total_rewards.append(total_reward)
        if total_reward > 0:
            success_count.append(1)
            hole_failure_flags.append(0)
            timeout_failure_flags.append(0)
        else:
            success_count.append(0)
            row, col = divmod(state, map_size)
            tile_type = env.unwrapped.desc[row][col]
            if tile_type == b'H':
                hole_failure_flags.append(1)
                timeout_failure_flags.append(0)
            else:
                hole_failure_flags.append(0)
                timeout_failure_flags.append(1)

    return total_rewards, success_count, hole_failure_flags, timeout_failure_flags

def plot_hole_falls(train_failures_q, train_failures_sarsa, eval_holes_q, eval_holes_sarsa, map_size):
    episodes = np.arange(len(train_failures_q))
    window = 50

    # Smoothed training hole falls per episode (averaged across runs)
    q_train_smooth = pd.Series(train_failures_q.mean(axis=1)).rolling(window, min_periods=1).mean()
    sarsa_train_smooth = pd.Series(train_failures_sarsa.mean(axis=1)).rolling(window, min_periods=1).mean()

    plt.figure(figsize=(14, 6))

    # --- Training: hole falls per episode ---
    plt.subplot(1, 2, 1)
    plt.plot(episodes, q_train_smooth, label="Q-Learning", color="steelblue")
    plt.plot(episodes, sarsa_train_smooth, label="SARSA", color="green")
    plt.title("Training Hole Falls (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Hole Falls out of 1.0")
    max_y = max(q_train_smooth.max(), sarsa_train_smooth.max()) * 1.2
    plt.ylim(0, max(0.05, max_y))
    plt.legend()
    plt.grid(True)

    # --- Evaluation: fraction of episodes that ended in a hole ---
    eval_means = [np.mean(eval_holes_q), np.mean(eval_holes_sarsa)]
    plt.subplot(1, 2, 2)
    plt.bar(["Q-Learning", "SARSA"], eval_means, color=["steelblue", "green"])
    plt.title("Evaluation Hole Failures")
    plt.ylabel("Proportion of Episodes")
    plt.ylim(0, max(0.05, max_y))
    plt.grid(True)

    plt.suptitle(f"Hole Fall Risk Comparison – {map_size}x{map_size}")
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"hole_falls_{map_size}x{map_size}.png")
    plt.show()
    
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

    def tile_labels(desc):
        """Create labels like 'S←', 'H', 'F→', 'G↑' depending on learned action and tile type."""
        tile_overlay = np.array(desc, dtype=str)
        label_grid = np.full_like(tile_overlay, "", dtype=object)
        return tile_overlay

    q_max, q_dir = prepare_policy(q_qtable)
    sarsa_max, sarsa_dir = prepare_policy(sarsa_qtable)
    tile_overlay = tile_labels(env.unwrapped.desc)

    # Combine tile type and direction
    q_labels = np.where(q_dir == "", tile_overlay, np.char.add(tile_overlay, q_dir))
    sarsa_labels = np.where(sarsa_dir == "", tile_overlay, np.char.add(tile_overlay, sarsa_dir))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(q_max, annot=q_labels, fmt="", ax=ax[0], cmap="Blues", alpha=0.7,
                linewidths=0.7, linecolor="black", cbar=False,
                xticklabels=False, yticklabels=False,
                annot_kws={"fontsize": "large", "weight": "bold"})
    ax[0].set_title("Q-Learning Policy (Tile Type + Direction)")

    sns.heatmap(sarsa_max, annot=sarsa_labels, fmt="", ax=ax[1], cmap="Greens", alpha=0.7,
                linewidths=0.7, linecolor="black", cbar=False,
                xticklabels=False, yticklabels=False,
                annot_kws={"fontsize": "large", "weight": "bold"})
    ax[1].set_title("SARSA Policy (Tile Type + Direction)")

    for a in ax:
        a.axis("off")
        for _, spine in a.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

    plt.suptitle(f"Q-Learning vs SARSA Policy – {map_size}x{map_size} Map")
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"q_vs_sarsa_tile_overlay_{map_size}x{map_size}.png")
    plt.show()


def plot_states_actions_distribution(q_states, q_actions, s_states, s_actions, map_size):
    labels = ["LEFT", "DOWN", "RIGHT", "UP"]
    q_counts = np.bincount(q_actions, minlength=4)
    sarsa_counts = np.bincount(s_actions, minlength=4)  # not sarsa_actions

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, q_counts, width, label='Q-Learning', color='steelblue')
    ax.bar(x + width/2, sarsa_counts, width, label='SARSA', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Action Distribution")
    ax.legend()
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="rewards", hue="map_size", ax=ax[0]
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
    
    sns.lineplot(data=df, x="Episodes", y="Rewards", hue="Model", ax=ax[0])
    ax[0].set_title("Cumulative Rewards")
    
    window = 50  # You can adjust this
    df["RollingSteps"] = df.groupby("Model")["Steps"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    sns.lineplot(data=df, x="Episodes", y="RollingSteps", hue="Model", ax=ax[1])
    ax[1].set_title(f"Rolling Steps per Episode (window={window})")
    
    for a in ax:
        a.legend(title="Model")
        a.set(xlabel="Episode")
    
    fig.tight_layout()
    plt.savefig(params.savefig_folder / f"compare_{map_size}x{map_size}.png")
    plt.show()
    
def plot_success_failure_comparison(successes_q, hole_failures_q, timeout_failures_q,
                                    successes_sarsa, hole_failures_sarsa, timeout_failures_sarsa,
                                    map_size, window=50):
    episodes = np.arange(successes_q.shape[0])
    smooth = lambda arr: pd.Series(arr.mean(axis=1)).rolling(window, min_periods=1).mean()

    # --- Q-Learning smoothed metrics ---
    q_success_smooth = smooth(successes_q)
    q_hole_smooth = smooth(hole_failures_q)
    q_timeout_smooth = smooth(timeout_failures_q)

    # --- SARSA smoothed metrics ---
    sarsa_success_smooth = smooth(successes_sarsa)
    sarsa_hole_smooth = smooth(hole_failures_sarsa)
    sarsa_timeout_smooth = smooth(timeout_failures_sarsa)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- Q-Learning subplot ---
    ax[0].plot(episodes, q_success_smooth, label="Success", color="forestgreen")
    ax[0].plot(episodes, q_hole_smooth, label="Hole Failure", color="royalblue")
    ax[0].plot(episodes, q_timeout_smooth, label="Timeout Failure", color="steelblue")
    ax[0].set_title("Q-Learning")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Rate")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend()
    ax[0].grid(True)

    # --- SARSA subplot ---
    ax[1].plot(episodes, sarsa_success_smooth, label="Success", color="seagreen")
    ax[1].plot(episodes, sarsa_hole_smooth, label="Hole Failure", color="darkorange")
    ax[1].plot(episodes, sarsa_timeout_smooth, label="Timeout Failure", color="firebrick")
    ax[1].set_title("SARSA")
    ax[1].set_xlabel("Episode")
    ax[1].legend()
    ax[1].grid(True)

    plt.suptitle(f"Success vs Failure Type per Algorithm – {map_size}x{map_size}")
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"success_failure_type_split_{map_size}x{map_size}.png")
    plt.show()

    
def plot_rolling_avg_rewards(df, window=50):
    df["RollingReward"] = df.groupby("Model")["Rewards"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x="Episodes", y="RollingReward", hue="Model")
    plt.title(f"Rolling Avg Reward (window={window})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"rolling_avg_reward_{params.map_size}x{params.map_size}.png")
    plt.show()
    
def plot_evaluation_curves(eval_rewards_q, eval_success_q, eval_rewards_sarsa, eval_success_sarsa, map_size, window=50):
    n_eval_episodes = len(eval_rewards_q)

    df_eval = pd.DataFrame({
        "Episode": list(range(n_eval_episodes)) * 2,
        "Reward": eval_rewards_q + eval_rewards_sarsa,
        "Success": eval_success_q + eval_success_sarsa,
        "Model": ["Q-Learning"] * n_eval_episodes + ["SARSA"] * n_eval_episodes
    })

    df_eval["RollingReward"] = df_eval.groupby("Model")["Reward"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df_eval["RollingSuccess"] = df_eval.groupby("Model")["Success"].transform(lambda x: x.rolling(window, min_periods=1).mean())

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=df_eval, x="Episode", y="RollingReward", hue="Model", ax=ax[0])
    ax[0].set_title("Evaluation: Rolling Average Reward")
    ax[0].set_ylabel("Avg Reward")

    sns.lineplot(data=df_eval, x="Episode", y="RollingSuccess", hue="Model", ax=ax[1])
    ax[1].set_title("Evaluation: Rolling Success Rate")
    ax[1].set_ylabel("Success Rate")

    for a in ax:
        a.grid(True)
        a.set_xlabel("Episode")

    plt.suptitle(f"Evaluation Performance – {map_size}x{map_size} Map")
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"eval_rolling_metrics_{map_size}x{map_size}.png")
    plt.show()

map_sizes = [10]
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
    q_rewards, q_steps, episodes, q_qtables, q_states, q_actions, q_successes, q_hole_falls, q_hole_failures, q_timeout_failures = run_env(q_learner, "q_learning")
    sarsa_rewards, sarsa_steps, _, sarsa_qtables, sarsa_states, sarsa_actions, sarsa_successes, sarsa_hole_falls, sarsa_hole_failures, sarsa_timeout_failures = run_env(sarsa_learner, "sarsa")

    def make_df(model_name, rewards, steps):
        df = pd.DataFrame({
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
            "Model": model_name,
            "MapSize": f"{map_size}x{map_size}",
        })
        df["Rewards"] = rewards.cumsum(axis=0).flatten(order="F")
        return df

    df_q = make_df("Q-Learning", q_rewards, q_steps)
    df_sarsa = make_df("SARSA", sarsa_rewards, sarsa_steps)
    df_all = pd.concat([df_q, df_sarsa], ignore_index=True)

    plot_model_comparison(df_all)
    plot_states_actions_distribution(q_states, q_actions, sarsa_states, sarsa_actions, map_size)
    plot_success_failure_comparison(q_successes, q_hole_failures, q_timeout_failures, sarsa_successes, sarsa_hole_failures, sarsa_timeout_failures, map_size)

    qtable_q = q_qtables.mean(axis=0)
    qtable_sarsa = sarsa_qtables.mean(axis=0)
    plot_q_values_map_comparison(qtable_q, qtable_sarsa, env, map_size)

    plot_rolling_avg_rewards(df_all, 20)

    # --- Evaluation (no exploration) ---
    q_eval_rewards, q_eval_success, q_eval_holes, q_eval_timeouts = evaluate_policy(q_learner, "q_learning", qtable_q)
    sarsa_eval_rewards, sarsa_eval_success, sarsa_eval_holes, sarsa_eval_timeouts = evaluate_policy(sarsa_learner, "sarsa", qtable_sarsa)

    # --- Plot evaluation performance ---
    models = ["Q-Learning", "SARSA"]
    avg_rewards = [np.mean(q_eval_rewards), np.mean(sarsa_eval_rewards)]
    reward_se = [sem(q_eval_rewards), sem(sarsa_eval_rewards)]

    plt.figure(figsize=(10, 5))
    plt.bar(models, avg_rewards, yerr=reward_se, capsize=10, color=["steelblue", "green"])
    plt.ylabel("Average Reward ± SE")
    plt.title("Evaluation: Average Reward with Error Bars")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"eval_avg_reward_{map_size}x{map_size}.png")
    plt.show()

    # --- Success Rate ---
    avg_success = [np.mean(q_eval_success), np.mean(sarsa_eval_success)]
    success_se = [sem(q_eval_success), sem(sarsa_eval_success)]

    plt.figure(figsize=(10, 5))
    plt.bar(models, avg_success, yerr=success_se, capsize=10, color=["steelblue", "green"])
    plt.ylabel("Average Success Rate ± SE")
    plt.title("Evaluation: Success Rate with Error Bars")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(params.savefig_folder / f"eval_success_rate_{map_size}x{map_size}.png")
    plt.show()

    plot_evaluation_curves(q_eval_rewards, q_eval_success, sarsa_eval_rewards, sarsa_eval_success, map_size)

    plot_hole_falls(q_hole_failures, sarsa_hole_failures, q_eval_holes, sarsa_eval_holes, map_size)

    env.close()
    