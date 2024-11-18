import argparse
from analyze import playpen_correlation_logger
from analyze.dataset_correlation import run_correlation


def main(args: argparse.Namespace) -> None:
    if args.command_name == "run":
        playpen_correlation_logger.info(f"Starting the correlation analysis. Extracting data from")
        run_correlation(data_path = args.data_path, save_results_path = args.save_results_path, consider_subtasks_of=args.consider_subtasks_of)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)

    run_parser.add_argument(
        "-d", "--data_path",
        type=str,
        default="results/playpen",
        help="Path to the folder containing the results from which to extract data for the correlation analysis."
    )
    run_parser.add_argument(
        "-p", "--save_results_path",
        type=str,
        default="results/correlation",
        help="Path to the folder where to save the results from the correlation analysis."
    )
    run_parser.add_argument(
        "--consider_subtasks_of",
        type=str,
        default="",
        help="For the specific tasks here, take the subtasks rather than the aggregation of results on subtasks."
    )
    args = parser.parse_args()
    main(args)