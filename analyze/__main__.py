import argparse
from pathlib import Path
from analyze import playpen_correlation_logger
from analyze.dataset_correlation import run_correlation, verify_functional_correlation_patterns


def main(args: argparse.Namespace) -> None:
    if args.command_name == "run_correlation":
        playpen_correlation_logger.info(f"Starting the correlation analysis for the experiment.")
        run_correlation(src_path = Path(args.src_path),
                        output_path = Path(args.output_path),
                        take_subtasks = args.take_subtasks,
                        divide_by_model_size = args.by_model_size,
                        name=args.name,
                        tasks_to_ignore=args.tasks_to_ignore)
    elif args.command_name == "run_verify_functional_correlation_patterns":
        playpen_correlation_logger.info(f"Starting to analyze patterns in the correlations.")
        verify_functional_correlation_patterns(src_path = Path(args.src_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    run_correlation_parser = sub_parsers.add_parser("run_correlation", formatter_class=argparse.RawTextHelpFormatter)

    run_correlation_parser.add_argument(
        "-s", "--src_path",
        type=str,
        default="results/playpen",
        help="Path to the folder containing the results from which to extract data for the correlation analysis."
    )
    run_correlation_parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="results/correlation",
        help="Path to the folder where to save the results from the correlation analysis."
    )
    run_correlation_parser.add_argument(
        "--take_subtasks",
        nargs="+",
        default=[],
        help="For the specific tasks here, take the subtasks rather than the aggregation of results on subtasks."
    )

    run_correlation_parser.add_argument(
        "--tasks_to_ignore",
        nargs="+",
        default=[],
        help="Specify tasks to ignore in the correlation analysis."
    )

    run_correlation_parser.add_argument(
        "--by_model_size",
        action="store_true",
        default=False,
        help="Whether to include also separate analysis for different sizes of models."
    )

    run_correlation_parser.add_argument(
        "--name",
        action="store_true",
        default=False,
        help="Whether to include the name of the image in the correlation plot."
    )

    run_verify_functional_correlation_patterns_parser = sub_parsers.add_parser("run_verify_functional_correlation_patterns", formatter_class=argparse.RawTextHelpFormatter)
    run_verify_functional_correlation_patterns_parser.add_argument(
        "-s", "--src_path",
        type=str,
        default="results/correlation",
        help="Path to the folder containing the results from which to extract data for the correlation analysis."
    )

    args = parser.parse_args()
    main(args)