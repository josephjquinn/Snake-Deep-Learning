import os
from util.game import gameAI
from util.helper import plot
import argparse
import csv
from util.agent import Agent
from matplotlib import pyplot as plt

OUTPUT_DIR = "./output"  # Define your output directory here

# Ensure output directory exists or create it
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train(model_path):
    plot_scores = []
    plot_mean_scores = []
    training_data = []
    epsilon_vals = []
    episode_losses = []
    step_losses = []
    action_counts = [0, 0, 0]  # [straight, right, left]
    action_distributions = []
    total_score = 0
    record = 0
    agent = Agent(model_path)
    game = gameAI()
    plt.figure(figsize=(7, 5))
    plt.ion()

    while True:
        episode_loss = 0
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        action_counts[final_move.index(1)] += 1
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        step_loss = agent.train_short_memory(
            state_old, final_move, reward, state_new, done
        )
        step_losses.append(step_loss)

        episode_loss += step_loss

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            if not model_path:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                epsilon = agent.epsilon
                epsilon_vals.append(epsilon)
                episode_losses.append(episode_loss)
                action_distributions.append(agent.action_counts.copy())
                agent.action_counts = [0, 0, 0]
                plot(
                    plot_scores,
                    plot_mean_scores,
                    episode_losses,
                    step_losses,
                    epsilon_vals,
                    action_distributions,
                    agent.n_games,
                )

                training_data.append(
                    (
                        score,
                        mean_score,
                        episode_loss,
                        step_losses,
                        epsilon,
                        action_counts.copy(),
                    )
                )
                save_data_to_csv("./output/training_data.csv", training_data)


def save_data_to_csv(filename, data):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Episode",
                "Score",
                "Mean Score",
                "Episode Loss",
                "Step Loss",
                "Epsilon",
                "Action Counts",
            ]
        )
        for episode, (
            score,
            mean_score,
            episode_loss,
            step_losses,
            epsilon,
            action_counts,
        ) in enumerate(data, start=1):
            writer.writerow(
                [
                    episode,
                    score,
                    mean_score,
                    episode_loss,
                    step_losses[-1],
                    epsilon,
                    action_counts,
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pre-trained model file",
    )
    args = parser.parse_args()

    train(model_path=args.model_path)
