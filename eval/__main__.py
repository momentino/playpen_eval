import argparse
import logging

from eval import playpen_evaluator

playpen_eval_logger = logging.getLogger("playpen_eval_logger")

def main(args: argparse.Namespace) -> None:
    if args.command_name == "ls":
        playpen_evaluator.list_tasks()
    if args.command_name == "run":
        playpen_evaluator.run(
            model_backend=args.model_backend,
            model_args=args.model_args,
            gen_kwargs=args.gen_kwargs,
            tasks=args.tasks,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            parallelize=args.parallelize,
            results_path=args.results_path,
        )

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
        "--parallelize",
        action="store_true",  # This makes it a flag
        default=False,  # Default is True if the flag is not passed
        help="Whether to run big models on multiple GPUs."
    )

    run_parser.add_argument(
        "--gen_kwargs",
        type=str,
        default="do_sample=False",
        help="Kwargs for generation. Same format as --model_args"
    )

    run_parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="Output path for results. Default is 'results/by_model'."
    )

    args = parser.parse_args()
    main(args)