import argparse
import logging

from eval.playpen_evaluator import PlaypenEvaluator

playpen_eval_logger = logging.getLogger("playpen_eval_logger")

def main(args: argparse.Namespace) -> None:
    if args.command_name == "ls":
        PlaypenEvaluator.list_tasks()
    if args.command_name == "run":
        PlaypenEvaluator.run(
            model_backend=args.model_backend,
            model_args=args.model_args,
            tasks=args.tasks,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            results_path=args.results_path,
        )
    if args.command_name == "build_model_report":
        PlaypenEvaluator.model_report()
    if args.command_name == "build_benchmark_report":
        PlaypenEvaluator.benchmark_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    sub_parsers.add_parser("ls")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)
    run_parser.add_argument(
        "-m", "--model_args",
        type=str,
        required=True,
        help="Model args, such as name, e.g., 'pretrained=model-name'."
    )
    run_parser.add_argument(
        "--tasks",
        nargs="+",
        default=["remaining"],
        help="List of tasks, e.g., 'task1 task2 task3'. "
             "Type: 'remaining' to evaluate on the tasks for which the model has yet to be evaluated. "
             "Type 'all' to evaluate on all tasks."
             "This won't work on tasks that are not in the Playpen pipeline"


    )
    run_parser.add_argument(
        "--model_backend",
        type=str,
        default="hf",
        help="Model type, default is 'hf'."
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g., 'cuda:0' or 'cpu'. Default is 'cuda:0'."
    )
    run_parser.add_argument(
        "--trust_remote_code",
        action="store_true",  # This makes it a flag
        default=True,  # Default is True if the flag is not passed
        help="Whether to trust remote code. Default is True."
    )

    run_parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="Output path for results. Default is 'results/by_model'."
    )

    model_report_parser = sub_parsers.add_parser("build_model_report", formatter_class=argparse.RawTextHelpFormatter)
    model_report_parser.add_argument(
        "-m", "--model_name",
        type=str,
        required=True,
        help="Model name, e.g., 'pretrained=model_name'."
    )
    model_report_parser.add_argument(
        "--results_path",
        type=str,
        default="results/by_model",
        help="Output path for results. Default is 'results/by_model'."
    )

    benchmark_report_parser = sub_parsers.add_parser("build_benchmark_report", formatter_class=argparse.RawTextHelpFormatter)
    benchmark_report_parser.add_argument(
        "-m", "--benchmark_name",
        type=str,
        required=True,
        help="Model name, e.g., 'pretrained=model_name'."
    )
    benchmark_report_parser.add_argument(
        "--results_path",
        type=str,
        default="results/by_benchmark",
        help="Output path for results. Default is 'results/by_benchmark'."
    )

    args = parser.parse_args()
    main(args)