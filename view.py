import csv
import argparse

from util.helper import plot_view


def load_data_from_csv(filename):
    data = []
    with open(filename, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            episode = int(row[0])
            score = float(row[1])
            mean_score = float(row[2])
            episode_loss = float(row[3])
            epsilon = float(row[4])
            action_counts = list(map(int, row[5].strip("[]").split(",")))

            data.append(
                (
                    episode,
                    score,
                    mean_score,
                    episode_loss,
                    epsilon,
                    action_counts,
                )
            )

    return data


def display_data(data):
    for (
        episode,
        score,
        mean_score,
        episode_loss,
        epsilon,
        action_counts,
    ) in data:
        print(f"Episode: {episode}")
        print(f"  Score: {score}")
        print(f"  Mean Score: {mean_score}")
        print(f"  Episode Loss: {episode_loss}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Action Counts: {action_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="View and plot training data from CSV file."
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to the CSV file containing training data"
    )
    args = parser.parse_args()

    data = load_data_from_csv(args.csv_file)

    display_data(data)

    (
        episodes,
        plot_scores,
        plot_mean_scores,
        episode_losses,
        epsilon_vals,
        action_distributions,
    ) = zip(*data)
    n_games = len(data)

    plot_view(
        list(plot_scores),
        list(plot_mean_scores),
        list(episode_losses),
        list(epsilon_vals),
        list(action_distributions),
        n_games,
    )


if __name__ == "__main__":
    main()
