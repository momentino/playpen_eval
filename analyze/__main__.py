import argparse
from pathlib import Path
from analyze import playpen_correlation_logger
from analyze.correlation import run_correlation
from analyze.scatterplots import run_scatterplots


def main(args: argparse.Namespace) -> None:
    if args.command_name == "correlation":
        playpen_correlation_logger.info(f"Starting the correlation analysis for the experiment.")
        output_path_root = Path(args.output_path) / Path(args.discriminant) / args.subset / args.correlation_method
        if args.discriminant == "benchmark" and args.subset in ["main","all"]:
            raise Exception("Cannot consider overall benchmarks here. We are taking into consideration within-benchmark subtasks. Try by adding the '--take_functional_subtasks' argument.")
        run_correlation(src_path = Path(args.src_path),
                        output_path_root = output_path_root,
                        tiers = args.tiers,
                        correlation_method = args.correlation_method,
                        discriminant = args.discriminant,
                        partial=args.partial,
                        subset = args.subset,
                        ignore_tasks=args.ignore_tasks,
                        ignore_groups=args.ignore_groups,
                        take_above_baseline=args.take_above_baseline)
    elif args.command_name == "scatterplot":
        playpen_correlation_logger.info(f"Plotting results for pairs of benchmarks")
        run_scatterplots(src_path=Path(args.src_path),
                        output_path_root=Path(args.output_path),
                        ignore_groups=args.ignore_groups
                           )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")

    run_correlation_parser = sub_parsers.add_parser("correlation", formatter_class=argparse.RawTextHelpFormatter)

    run_correlation_parser.add_argument(
        "-s", "--src_path",
        type=str,
        default="results/playpen_eval",
        help="Path to the folder containing the results from which to extract data for the correlation analysis."
    )
    run_correlation_parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="results/correlation",
        help="Path to the folder where to save the results from the correlation analysis."
    )
    run_correlation_parser.add_argument(
        "--subset",
        type=str,
        default='main',
        help="Choose which subset of results you wish to consider. Admissible values: 'subtasks', 'main', 'all'."
    )

    run_correlation_parser.add_argument(
        "--ignore_tasks",
        nargs="+",
        default=[],
        help="Specify tasks to ignore in the correlation analysis."
    )

    run_correlation_parser.add_argument(
        "--ignore_groups",
        nargs="+",
        default=[],
        help="Specify groups of tasks to ignore in the correlation analysis."
    )

    run_correlation_parser.add_argument(
        "--tiers",
        action="store_true",
        default=False,
        help="Whether to include also separate analysis for different sizes of models."
    )

    run_correlation_parser.add_argument(
        "--take_above_baseline",
        action="store_true",
        default=False,
        help="Whether to consider only results above the random baseline or not."
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

    run_correlation_parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Whether to consider only results above the random baseline or not."
    )

    scatterplot_parser = sub_parsers.add_parser("scatterplot", formatter_class=argparse.RawTextHelpFormatter)
    scatterplot_parser.add_argument(
        "-s", "--src_path",
        type=str,
        default="results/playpen_eval",
        help="Path to the folder containing the results from which to extract data for the plots."
    )
    scatterplot_parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="results/scatterplots",
        help="Path to the folder where to save the plots."
    )

    scatterplot_parser.add_argument(
        "--ignore_groups",
        nargs="+",
        default=[],
        help="Specify groups of tasks to ignore in the scatterplot analysis."
    )

    args = parser.parse_args()
    main(args)