from util.game import gameAI
from util.helper import plot
import argparse
import csv
from util.agent import Agent


def train(fresh):
    plot_scores = []
    plot_mean_scores = []
    training_data = []
    total_score = 0
    record = 0
    agent = Agent(fresh)
    game = gameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

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

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            training_data.append((score, mean_score))
            save_data_to_csv("training_data.csv", training_data)


def save_data_to_csv(filename, data):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Score", "Mean Score"])  # Write header row
        for episode, (score, mean_score) in enumerate(data, start=1):
            writer.writerow([episode, score, mean_score])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Load fresh model")
    args = parser.parse_args()
    train(args.fresh)
