import argparse
from pathlib import Path
from analyze import playpen_correlation_logger
from analyze.dataset_correlation import run_correlation


def main(args: argparse.Namespace) -> None:
    if args.command_name == "run":
        playpen_correlation_logger.info(f"Starting the correlation analysis. Extracting data from")
        run_correlation(src_path = Path(args.src_path), output_path = Path(args.output_path), take_subtasks = args.take_subtasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)

    run_parser.add_argument(
        "-s", "--src_path",
        type=str,
        default="results/playpen",
        help="Path to the folder containing the results from which to extract data for the correlation analysis."
    )
    run_parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="results/correlation",
        help="Path to the folder where to save the results from the correlation analysis."
    )
    run_parser.add_argument(
        "--take_subtasks",
        nargs="+",
        default=[],
        help="For the specific tasks here, take the subtasks rather than the aggregation of results on subtasks."
    )
    args = parser.parse_args()
    main(args)