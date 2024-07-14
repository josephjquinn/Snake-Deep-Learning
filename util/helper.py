import matplotlib.pyplot as plt
import numpy as np


def plot(
    plot_scores,
    plot_mean_scores,
    episode_losses,
    step_losses,
    epsilon,
    action_distributions,
    n_games,
):
    plt.clf()

    # Subplot 1: Scores over episodes
    plt.subplot(3, 2, 1)
    plt.plot(plot_scores, label="Scores")
    plt.plot(plot_mean_scores, label="Mean Scores")
    plt.title("Scores over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    # Subplot 2: Episode losses
    plt.subplot(3, 2, 2)
    plt.plot(episode_losses)
    plt.title("Episode Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    # Subplot 3: Step losses
    plt.subplot(3, 2, 3)
    plt.plot(step_losses)
    plt.title("Step Losses")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Subplot 4: Action distributions
    plt.subplot(3, 2, 4)
    action_episodes = list(range(1, len(action_distributions) + 1))
    action_distributions = np.array(action_distributions)
    total_actions = action_distributions.sum(axis=1)
    action_percentages = (action_distributions.T / total_actions).T * 100

    plt.stackplot(
        action_episodes,
        action_percentages.T,
        labels=["Straight", "Right", "Left"],
        colors=["blue", "green", "red"],
    )
    plt.title("Action Distributions")
    plt.xlabel("Episode")
    plt.ylabel("Percentage")
    plt.legend(loc="upper left")

    # Subplot 5: Epsilon over episodes
    plt.subplot(3, 2, 5)
    plt.plot(range(1, n_games + 1), epsilon)
    plt.title("Epsilon over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")


    # Subplot 6: Total ations
    plt.subplot(3, 2, 6)
    action_episodes = list(range(1, len(action_distributions) + 1))
    total_actions = np.sum(action_distributions, axis=1)  # Total actions per episode
    plt.plot(action_episodes, total_actions, label="Total Actions", color="purple")
    plt.title("Total Actions per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Actions")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def plot_view(
    plot_scores,
    plot_mean_scores,
    episode_losses,
    epsilon,
    action_distributions,
    n_games,
):
    plt.figure(figsize=(6, 4))
    plt.plot(plot_scores, label="Scores")
    plt.plot(plot_mean_scores, label="Mean Scores")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(episode_losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

    plt.figure(figsize=(6, 4))
    action_episodes = list(range(1, len(action_distributions) + 1))
    action_distributions = np.array(action_distributions)
    total_actions = action_distributions.sum(axis=1)
    action_percentages = (action_distributions.T / total_actions).T * 100

    plt.stackplot(
        action_episodes,
        action_percentages.T,
        labels=["Straight", "Right", "Left"],
        colors=["blue", "green", "red"],
    )
    plt.xlabel("Episode")
    plt.ylabel("Action distribution")
    plt.legend(loc="upper left")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, n_games + 1), epsilon)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()

    plt.pause(0.1)
