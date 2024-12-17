import argparse
from pathlib import Path
from analyze import playpen_correlation_logger
from analyze.dataset_correlation import run_correlation, verify_functional_correlation_patterns


def main(args: argparse.Namespace) -> None:
    if args.command_name == "correlation":
        playpen_correlation_logger.info(f"Starting the correlation analysis for the experiment.")
        if args.take_functional_subtasks:
            output_path_root = Path(args.output_path) / Path(args.discriminant) / Path("unpacked") / args.correlation_method
        else:
            if args.discriminant == "benchmarks":
                raise Exception("Cannot consider overall benchmarks here. We are taking into consideration within-benchmark subtasks. Try by adding the '--take_functional_subtasks' argument.")
            output_path_root = Path(args.output_path) / Path(args.discriminant) / Path("overall") / args.correlation_method
        run_correlation(src_path = Path(args.src_path),
                        output_path_root = output_path_root,
                        tiers = args.tiers,
                        correlation_method = args.correlation_method,
                        discriminant = args.discriminant,
                        take_functional_subtasks = args.take_functional_subtasks,
                        tasks_to_ignore=args.tasks_to_ignore)
    elif args.command_name == "verify_functional_correlation_patterns":
        playpen_correlation_logger.info(f"Starting to analyze patterns in the correlations.")
        verify_functional_correlation_patterns(src_path = Path(args.src_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    run_correlation_parser = sub_parsers.add_parser("correlation", formatter_class=argparse.RawTextHelpFormatter)

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
        "--take_functional_subtasks",
        action="store_true",
        default=False,
        help="For the specific tasks here, take the subtasks rather than the aggregation of results on subtasks."
    )

    run_correlation_parser.add_argument(
        "--tasks_to_ignore",
        nargs="+",
        default=[],
        help="Specify tasks to ignore in the correlation analysis."
    )

    run_correlation_parser.add_argument(
        "--tiers",
        action="store_true",
        default=False,
        help="Whether to include also separate analysis for different sizes of models."
    )

    run_correlation_parser.add_argument(
        "--correlation_method",
        type=str,
        default="pearson",
        choices = ['pearson', 'kendall', 'spearman'],
        help="Whether to include the name of the image in the correlation plot."
    )

    run_correlation_parser.add_argument(
        "--discriminant",
        type=str,
        default="capabilities",
        choices=['capabilities', 'tasks', 'benchmarks'],
        help="The variable to consider for grouping benchmarks for the correlation analysis."
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