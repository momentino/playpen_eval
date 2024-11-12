import argparse
from analyze import playpen_correlation_logger
from analyze.dataset_correlation import run_correlation


def main(args: argparse.Namespace) -> None:
    if args.command_name == "run":
        playpen_correlation_logger.info(f"Starting the correlation analysis. Extracting data from")
        run_correlation(results_path = args.results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)

    run_parser.add_argument(
        "-p", "--results_path",
        type=str,
        default="results/playpen",
        help="Path to the folder containing the results from which to extract data for the correlation analysis."
    )

    args = parser.parse_args()
    main(args)