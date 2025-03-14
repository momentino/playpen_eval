import argparse
import evaluate
from pathlib import Path
from evaluate import playpen_evaluator

stdout_logger = evaluate.get_logger("evaluate.run")

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
            device_map_option=args.device_map_option,
            num_fewshot=args.num_fewshot,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            apply_chat_template=args.apply_chat_template,
            batch_size=args.batch_size
        )
    if args.command_name == "report_costs":
        playpen_evaluator.report_costs(
            results_path = Path(args.results_path),
            output_path=Path(args.output_path),
            models= args.models,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")


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
        action="store_true",
        default=True,
        help="Whether to trust remote code. Default is True."
    )

    run_parser.add_argument(
        "--parallelize",
        action="store_true",
        default=False,
        help="Whether to run big models on multiple GPUs."
    )


    run_parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="The number of few-shot samples to use for evaluation"
    )

    run_parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="Whether we wish to have fewshot example spanning across turns"
    )

    run_parser.add_argument(
        "--batch_size",
        type=str,
        default='auto',
        help="Batch size."
    )

    run_parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Whether to apply the chat template."
    )

    run_parser.add_argument(
        "--gen_kwargs",
        type=str,
        default="do_sample=False",
        help="Kwargs for generation. Same format as --model_args"
    )

    run_parser.add_argument(
        "--device_map_option",
        type=str,
        default="",
        help="Output path for the reports for the cost estimates. Default is 'results/cost_reports'."
    )

    report_costs_parser = sub_parsers.add_parser("report_costs", formatter_class=argparse.RawTextHelpFormatter)
    report_costs_parser.add_argument(
        "--results_path",
        type=str,
        default="results/playpen_eval",
        help="Output path for results. Default is 'results/playpen_eval'."
    )
    report_costs_parser.add_argument(
        "--output_path",
        type=str,
        default="results/cost_reports",
        help="Output path for the reports for the cost estimates. Default is 'results/cost_reports'."
    )
    report_costs_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="""
            List of models for which we want to estimate costs
        """
    )



    args = parser.parse_args()
    main(args)