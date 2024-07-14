import os
from util.game import gameAI
from util.helper import plot
import argparse
import csv
from util.agent import Agent
from matplotlib import pyplot as plt

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train(model_path, speed, num_episodes):
    episode = 0
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
    game = gameAI(speed)
    plt.figure(figsize=(7, 5))
    plt.ion()

    while True:
        if episode >= num_episodes:
            break
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
            # train long memory
            game.reset()
            episode += 1

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
                        epsilon,
                        action_counts.copy(),
                    )
                )
                save_data_to_csv("./output/training.csv", training_data)


def evaluate(model_path, speed):
    agent = Agent(model_path)
    game = gameAI(speed)
    total_score = 0
    num_games = 0
    plot_scores = []
    plt.figure(figsize=(7, 5))
    plt.ion()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        _, done, score = game.play_step(final_move)

        if done:
            game.reset()
            num_games += 1
            total_score += score
            mean_score = total_score / num_games
            plot_scores.append(score)
            print("Game", num_games, "Score", score, "Mean Score:", mean_score)

            if not model_path:
                plot(plot_scores, mean_scores=[mean_score], game_num=num_games)

        if num_games >= 10:  # evaluate 10 games
            break


def save_data_to_csv(filename, data):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Episode",
                "Score",
                "Mean Score",
                "Episode Loss",
                "Epsilon",
                "Action Counts",
            ]
        )
        for episode, (
            score,
            mean_score,
            episode_loss,
            epsilon,
            action_counts,
        ) in enumerate(data, start=1):
            writer.writerow(
                [
                    episode,
                    score,
                    mean_score,
                    episode_loss,
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
    parser.add_argument(
        "--speed",
        type=int,
        default=20,
        help="Game Speed Default: 20",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to train for",
    )
    args = parser.parse_args()

    if args.model_path:
        print("EVAL MODEL")
        evaluate(model_path=args.model_path, speed=args.speed)
    else:
        print("TRAINING MODEL")
        train(
            model_path=args.model_path, speed=args.speed, num_episodes=args.num_episodes
        )
