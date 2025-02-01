import argparse
from pathlib import Path
from analyze import playpen_correlation_logger
from analyze.correlation import run_correlation
from analyze.scatterplots import run_scatterplots
from analyze.barcharts import run_barcharts


def main(args: argparse.Namespace) -> None:
    if args.command_name == "correlation":
        playpen_correlation_logger.info(f"Starting the correlation analysis for the experiment.")
        if args.by == "benchmarks":
            output_path_root = Path(args.output_path) / Path(args.discriminant) / args.benchmark_subset / args.correlation_method
        elif args.by == "models":
            output_path_root = Path(args.output_path) / "models" / args.benchmark_subset / args.correlation_method
        if args.discriminant == "benchmark" and args.subset in ["main","all"]:
            raise Exception("Cannot consider overall benchmarks here. We are taking into consideration within-benchmark subtasks. Try by adding the '--take_functional_subtasks' argument.")
        run_correlation(src_path = Path(args.src_path),
                        output_path_root = output_path_root,
                        tiers = args.tiers,
                        correlation_method = args.correlation_method,
                        discriminant = args.discriminant,
                        partial=args.partial,
                        benchmark_subset = args.benchmark_subset,
                        ignore_tasks=args.ignore_tasks,
                        ignore_groups=args.ignore_groups,
                        functional_groups_to_exclude=args.functional_groups_to_exclude,
                        take_above_baseline=args.take_above_baseline,
                        by=args.by)
    elif args.command_name == "scatterplot":
        playpen_correlation_logger.info(f"Plotting results for pairs of benchmarks")
        output_path = Path(args.output_path) / args.by
        run_scatterplots(src_path=args.src_path,
                        output_path_root=output_path,
                        ignore_groups=args.ignore_groups,
                        by=args.by
                           )
    elif args.command_name == "barchart":
        output_path = Path(args.output_path) / args.by
        run_barcharts(src_path=args.src_path,
                         output_path_root=output_path,
                         ignore_groups=args.ignore_groups,
                         by=args.by
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
        "--benchmark_subset",
        type=str,
        default='main',
        help="Choose which subset of results you wish to consider. Admissible values: 'subtasks', 'main', 'all'."
    )

    run_correlation_parser.add_argument(
        "--by",
        type=str,
        default='benchmarks',
        choices=["benchmarks","models"],
        help="Choose whether you wish to compute the correlation by benchmark or model."
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
        default=None,
        choices=[None, 'capabilities', 'tasks', 'benchmarks'],
        help="The variable to consider for grouping benchmarks for the correlation analysis."
    )

    run_correlation_parser.add_argument(
        "--functional_groups_to_exclude",
        nargs="+",
        default=[],
        help="Specify groups of functional capabilities to not include in the analysis."
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

    scatterplot_parser.add_argument(
        "--by",
        type=str,
        default='benchmarks',
        choices=["benchmarks", "models"],
        help="Choose whether you wish to compute the correlation by benchmark or model."
    )

    barchart_parser = sub_parsers.add_parser("barchart", formatter_class=argparse.RawTextHelpFormatter)
    barchart_parser.add_argument(
        "-s", "--src_path",
        type=str,
        default="results/playpen_eval",
        help="Path to the folder containing the results from which to extract data for the bar charts."
    )
    barchart_parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="results/barcharts",
        help="Path to the folder where to save the bar charts."
    )

    barchart_parser.add_argument(
        "--ignore_groups",
        nargs="+",
        default=[],
        help="Specify groups of tasks to ignore in the bar charts."
    )

    barchart_parser.add_argument(
        "--by",
        type=str,
        default='benchmarks',
        choices=["benchmarks", "models"],
        help="Choose whether you wish to create the bar charts by benchmark or model."
    )

    args = parser.parse_args()
    main(args)